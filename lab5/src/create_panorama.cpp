#include <algorithm>
#include <random>

#include <opencv4/opencv2/imgproc.hpp>
#include <opencv4/opencv2/calib3d.hpp>
#include <opencv4/opencv2/features2d.hpp>

#include "include/corners_matching.hpp"
#include "include/homography.hpp"


std::vector<cv::Mat> createLaplacianPyramid(const cv::Mat& im, int nlevels=-1) {
    cv::Mat im1;
    im.convertTo(im1, CV_16SC1);

    std::vector<cv::Mat> pyr;
    nlevels -= 1;

    while (std::min(im1.rows, im1.cols) > 2 && nlevels != 0) {
        cv::Mat im2;
        cv::Size downSize((im1.cols + 1) / 2, (im1.rows + 1) / 2);
        cv::pyrDown(im1, im2, downSize);
        cv::Mat im3;
        cv::pyrUp(im2, im3, im1.size());
        cv::Mat layer = im1 - im3;
        pyr.push_back(layer);
        im1 = im2;
        nlevels--;
    }

    pyr.push_back(im1);

    return pyr;
}


cv::Mat pyramidalMerge(const cv::Mat& im1, const cv::Mat& im2, const cv::Mat& mask, int nlevels) {
    auto pyrm = match_corners::createGaussianPyramid(mask, nlevels);
    auto pyr1 = createLaplacianPyramid(im1, nlevels);
    auto pyr2 = createLaplacianPyramid(im2, nlevels);

    cv::Mat u1, u2, m;
    pyr1.back().convertTo(u1, CV_32SC1);
    pyr2.back().convertTo(u2, CV_32SC1);
    pyrm.back().convertTo(m, CV_32SC1);

    cv::Mat mul1, mul2;
    cv::multiply(u1, 255 - m, mul1);
    cv::multiply(u2, m, mul2);
    cv::Mat u = (mul1 + mul2) / 255;
    u.convertTo(u, CV_16SC1);

    nlevels -= 1;

    for (int i = nlevels - 1; i >= 0; i--) {
        cv::pyrUp(u, u  , pyr1[i].size());
        cv::Mat lap1, lap2, m;
        pyr1[i].convertTo(lap1, CV_32SC1);
        pyr2[i].convertTo(lap2, CV_32SC1);
        pyrm[i].convertTo(m, CV_32SC1);

        cv::Mat mul1, mul2;
        cv::multiply(lap1, 255 - m, mul1);
        cv::multiply(lap2, m, mul2);
        cv::Mat lap = (mul1 + mul2) / 255;
        lap.convertTo(lap, CV_16SC1);
        u = u + lap;
    }

    u.convertTo(u, CV_8UC1);
    return u;
}


cv::Mat overlapImages(const cv::Mat& result, const cv::Mat& src, const cv::Mat& H, int overlap=25) {   
    cv::Mat mask(src.size(), CV_8UC1, cv::Scalar{255});
    mask(cv::Rect(0, 0, overlap, mask.rows)).setTo(0);
    mask(cv::Rect(mask.cols - overlap, 0, overlap, mask.rows)).setTo(0);

    cv::Mat mask_warped;
    cv::warpPerspective(mask, mask_warped, H, result.size());
    cv::GaussianBlur(mask_warped, mask_warped, cv::Size{2*overlap - 1, 2*overlap - 1}, 0, 0);

    cv::Mat src_warped;
    cv::warpPerspective(src, src_warped, H, result.size());

    int nlevels = 7;
    return pyramidalMerge(result, src_warped, mask_warped, nlevels);
}


cv::Mat createPanorama(const cv::Mat& im1, const cv::Mat im2, const cv::Mat& H21) {
    cv::Mat result(im1.rows * 3 / 2, im1.cols * 3 / 2, CV_8UC1, cv::Scalar{0});
    cv::Mat H = cv::Mat::eye(cv::Size{3, 3}, CV_32FC1);
    H.at<float>(0, 2) = im1.cols / 6;
    H.at<float>(1, 2) = im1.rows / 6;

    result = overlapImages(result, im1, H);

    H = H * H21;

    result = overlapImages(result, im2, H);

    return result;
}


int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "wrong usage" << std::endl;
        exit(1);
    }

    cv::Mat im1 = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
    cv::Mat im2 = cv::imread(argv[2], cv::IMREAD_GRAYSCALE);

    if (im1.empty() || im2.empty()) {
        std::cerr << "empty image" << std::endl;
        return -1;
    }

    if (im1.cols != 640 || im1.rows != 480) {
        if (im1.cols > im1.rows) {
            cv::resize(im1, im1, cv::Size{640, 640 * im1.rows / im1.cols});
        }
        else {
            cv::resize(im1, im1, cv::Size{480 * im1.cols / im1.rows, 480});
        }
        cv::resize(im2, im2, im1.size());
    }

    cv::equalizeHist(im1, im1);
    cv::equalizeHist(im2, im2);

    std::vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(im1, corners, 0, 0.02, 4);

    std::vector<bool> mask;
    std::vector<match_corners::match> matches;
    match_corners::matchCorners(im1, im2, corners, matches, mask);

    std::vector<match_corners::match> good_matches;
    for (auto i = 0; i < matches.size(); i++) {
        if (mask[i]) {
            good_matches.push_back(matches[i]);
        }
    }

    std::vector<cv::Point2f> points1, points2;
    for (const auto& m : good_matches) {
        points1.push_back(m.first);
        points2.push_back(m.second);
    }

    std::cout << good_matches.size() << " of " << corners.size() << " matched" << std::endl;

    match_corners::draw(good_matches, im1, im2, false);

    cv::Mat H = findHomographyMatrixRANSAC(points1, points2, 0.999, 6., 2000, false, false);

    cv::Mat panorama = createPanorama(im1, im2, H);
    
    cv::imshow("panorama", panorama);
    cv::waitKey(0);

    return 0;
}
