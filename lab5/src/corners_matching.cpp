#include <iostream>
#include <random>

#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/imgproc.hpp>
#include <opencv4/opencv2/highgui.hpp>

#include "include/corners_matching.hpp"


namespace match_corners {

std::vector<cv::Mat> createGaussianPyramid(const cv::Mat& im, int nlevels) {
    cv::Mat im1;
    im.convertTo(im1, CV_8UC1);
    std::vector<cv::Mat> pyr{im};

    nlevels -= 1;

    while (std::min(im1.rows, im1.cols) > 2 && nlevels != 0) {
        cv::Mat im2;
        cv::Size downSize((im1.cols + 1) / 2, (im1.rows + 1) / 2);
        cv::pyrDown(im1, im2, downSize);
        pyr.push_back(im2);
        im1 = im2;
        nlevels--;
    }

    return pyr;
}


void findMatchedCorners(const std::vector<cv::Point2f>& corners,
                        const std::vector<cv::Mat>& pyr1,
                        const std::vector<cv::Mat>& pyr2,
                        std::vector<match>& matches,
                        std::vector<bool>& mask) {

    match empty_match(cv::Point2f{-1, -1}, cv::Point2f{-1, -1});
    matches.resize(corners.size());
    mask.resize(corners.size(), true);

    for (auto i = 0; i < matches.size(); i++) {
        matches[i].first = corners[i];
        matches[i].second = corners[i] / static_cast<float>(std::pow(2, pyr1.size()));
    }

    int sz = 5;
    int sz_half = (sz - 1) / 2;

    int roi_sz = 9;
    int roi_sz_half = (roi_sz - 1) / 2;

    for (size_t i = pyr1.size(); i-- > 0;) {
        const cv::Mat& p1 = pyr1[i];
        const cv::Mat& p2 = pyr2[i];

        float divider = std::pow(2, i);

        for (auto j = 0; j < corners.size(); j++) {
            if (!mask[j]) {
                continue;
            }

            cv::Point2f pt(corners[j] / divider);
            cv::Point2f pt2(matches[j].second.x * 2, matches[j].second.y * 2);

            int templ_x = std::clamp(int(std::round(pt.x)) - sz_half, 0, p1.cols - 1);
            int templ_y = std::clamp(int(std::round(pt.y)) - sz_half, 0, p2.rows - 1);
            int templ_sz_x = std::clamp(sz, 1, p1.cols - 1 - templ_x);
            int templ_sz_y = std::clamp(sz, 1, p1.rows - 1 - templ_y);

            int roi_x = std::clamp(int(std::round(pt2.x)) - roi_sz_half, 0, p2.cols - 1);
            int roi_y = std::clamp(int(std::round(pt2.y)) - roi_sz_half, 0, p2.rows - 1);
            int roi_sz_x = std::clamp(roi_sz, 1, p2.cols - 1 - roi_x);
            int roi_sz_y = std::clamp(roi_sz, 1, p2.rows - 1 - roi_y);
                    
            cv::Mat templ = p1(cv::Rect{templ_x, templ_y, templ_sz_x, templ_sz_y});
            cv::Mat roi = p2(cv::Rect{roi_x, roi_y, roi_sz_x, roi_sz_y});

            cv::Mat res;
            cv::matchTemplate(roi, templ, res, cv::TM_CCOEFF_NORMED);

            double min, max;
            cv::Point min_pt, max_pt;
            cv::minMaxLoc(res, &min, &max, &min_pt, &max_pt);
            cv::Point2f match_pt = max_pt;

            match_pt.x += roi_x + templ_sz_x/2;
            match_pt.y += roi_y + templ_sz_y/2;

            if (cv::norm(pt - match_pt) > std::min(p1.cols, p1.rows)) {
                matches[j] = empty_match;
                mask[j] = false;
                continue;
            }

            matches[j] = match(pt, match_pt);
        }
    }
}


void matchCorners(const cv::Mat& im1,
                  const cv::Mat& im2,
                  const std::vector<cv::Point2f>& corners,
                  std::vector<match>& matches,
                  std::vector<bool>& mask) {

    std::vector<cv::Mat> pyr1 = createGaussianPyramid(im1);
    std::vector<cv::Mat> pyr2 = createGaussianPyramid(im2);

    findMatchedCorners(corners, pyr1, pyr2, matches, mask);
}


void draw(const std::vector<match>& matches, cv::Mat im1, cv::Mat im2, bool wait) {
    cv::Mat im(im1.rows, im1.cols * 2, im1.type(), cv::Scalar{128});
    im1.copyTo(im(cv::Rect(0, 0, im1.cols, im1.rows)));
    im2.copyTo(im(cv::Rect(im1.cols, 0, im2.cols, im2.rows)));
    cv::cvtColor(im, im, cv::COLOR_GRAY2BGR);

    cv::Mat im1_rgb, im2_rgb;
    cv::cvtColor(im1, im1_rgb, cv::COLOR_GRAY2BGR);
    cv::cvtColor(im2, im2_rgb, cv::COLOR_GRAY2BGR);

    for (const auto& m  : matches) {
        cv::Scalar color{double(random() % 256),
                         double(random() % 256),
                         double(random() % 256)};
        cv::Point2f second{m.second.x + im1.cols, m.second.y};
        cv::line(im, m.first, second, color, 1);
        cv::circle(im1_rgb, m.first, 1, color, -1);
        cv::circle(im2_rgb, m.second, 1, color, -1);
    }

    cv::imshow("matching", im);
    cv::imshow("corners1", im1_rgb);
    cv::imshow("corners2", im2_rgb);
    cv::imwrite("matching.png", im);
    cv::imwrite("corners1.png", im1_rgb);
    cv::imwrite("corners2.png", im2_rgb);

    if (wait) {
        cv::waitKey(0);
    }
}

}