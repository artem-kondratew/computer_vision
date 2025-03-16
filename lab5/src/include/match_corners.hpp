#ifndef MATCH_CORNERS_HPP
#define MATCH_CORNERS_HPP


#include <iostream>
#include <string>

#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/imgproc.hpp>
#include <opencv4/opencv2/highgui.hpp>


namespace match_corners {

using match = std::pair<cv::Point, cv::Point>;


std::vector<cv::Mat> createGaussianPyramid(const cv::Mat& im, int nlevels) {
    cv::Mat im1;
    im.convertTo(im1, CV_8UC1);
    std::vector<cv::Mat> pyr{im};

    nlevels -= 1;

    while (std::min(im.rows, im.cols) > 2 && nlevels != 0) {
        cv::Mat im2;
        cv::pyrDown(im1, im2);
        pyr.push_back(im2);
        im1 = im2;
        nlevels--;
    }

    return pyr;
}


void draw(const std::vector<match>& matches, cv::Mat im1, cv::Mat im2) {
    cv::Mat im(im1.rows, im1.cols * 2, im1.type(), cv::Scalar{128});
    im1.copyTo(im(cv::Rect(0, 0, im1.cols, im1.rows)));
    im2.copyTo(im(cv::Rect(im1.cols, 0, im2.cols, im2.rows)));
    cv::cvtColor(im, im, cv::COLOR_GRAY2BGR);

    for (const auto& m  : matches) {
        cv::Point second{m.second.x + im1.cols, m.second.y};
        cv::line(im, m.first, second, cv::Scalar{255, 0, 255}, 1);
    }

    cv::Mat im1_rgb, im2_rgb;
    cv::cvtColor(im1, im1_rgb, cv::COLOR_GRAY2BGR);
    cv::cvtColor(im2, im2_rgb, cv::COLOR_GRAY2BGR);

    for (auto m : matches) {
        cv::circle(im1_rgb, m.first, 1, cv::Scalar{255, 0, 255}, -1);
        cv::circle(im2_rgb, m.second, 1, cv::Scalar{255, 0, 255}, -1);
    }

    cv::imshow("matching", im);
    cv::imshow("corners1", im1_rgb);
    cv::imshow("corners2", im2_rgb);
    cv::waitKey(0);
}


void findMatchedCorners(const std::vector<cv::Point>& corners,
                        const std::vector<cv::Mat>& pyr1,
                        const std::vector<cv::Mat>& pyr2,
                        std::vector<match>& matches,
                        std::vector<bool>& mask) {

    match empty_match(cv::Point{-1, -1}, cv::Point{-1, -1});
    matches.resize(corners.size());
    mask.resize(corners.size(), true);

    for (auto i = 0; i < matches.size(); i++) {
        matches[i].first = corners[i];
        matches[i].second = corners[i] / static_cast<float>(std::pow(2, pyr1.size()));
    }

    int sz = 5;
    int sz_half = (sz - 1) / 2;
    int w = 5;

    int roi_sz = w * 2 + sz;

    for (size_t i = pyr1.size(); i-- > 0;) {
        const cv::Mat& p1 = pyr1[i];
        const cv::Mat& p2 = pyr2[i];

        float divider = std::pow(2, i);

        for (auto j = 0; j < corners.size(); j++) {
            if (!mask[j]) {
                continue;
            }

            cv::Point pt(corners[j] / divider);
            cv::Point pt2(matches[j].second.x * 2, matches[j].second.y * 2);

            if (pt.x < sz_half || pt.x > p1.cols - sz_half - 1 || pt.y < sz_half || pt.y > p1.rows - sz_half - 1) {
                matches[j] = empty_match;
                mask[j] = false;
                continue;
            }
            if (pt2.x < w + sz_half || pt2.x > p2.cols - w - sz_half - 1 || pt2.y < w + sz_half || pt2.y > p2.rows - w - sz_half - 1) {
                matches[j] = empty_match;
                mask[j] = false;
                continue;
            }

            cv::Mat templ = p1(cv::Rect{pt.x - sz_half, pt.y - sz_half, sz, sz});
            cv::Mat roi = p2(cv::Rect{pt2.x - roi_sz/2, pt2.y - roi_sz/2, roi_sz, roi_sz});

            cv::Mat res;
            cv::matchTemplate(roi, templ, res, cv::TM_CCOEFF_NORMED);

            double min, max;
            cv::Point min_pt, max_pt;
            cv::minMaxLoc(res, &min, &max, &min_pt, &max_pt);
            cv::Point& match_pt = max_pt;

            match_pt.x += pt2.x - w;
            match_pt.y += pt2.y - w;

            if (cv::norm(pt - match_pt) > std::min(p1.cols, p1.rows)) {
                matches[j] = empty_match;
                mask[j] = false;
                continue;
            }

            matches[j] = match(pt, match_pt);
        }
    }
}


void matchCorners(const cv::Mat& im1, const cv::Mat& im2, const std::vector<cv::Point>& corners, std::vector<match>& matches, std::vector<bool>& mask) {
    int nlevels = 4;

    std::vector<cv::Mat> pyr1 = createGaussianPyramid(im1, nlevels);
    std::vector<cv::Mat> pyr2 = createGaussianPyramid(im2, nlevels);

    findMatchedCorners(corners, pyr1, pyr2, matches, mask);
}

}  // match_corners


#endif  // MATCH_CORNERS_HPP
