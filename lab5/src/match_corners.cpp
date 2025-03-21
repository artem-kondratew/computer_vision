#include "include/corners_matching.hpp"


int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "wrong usage" << std::endl;
        exit(1);
    }

    cv::Mat im1 = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
    cv::Mat im2 = cv::imread(argv[2], cv::IMREAD_GRAYSCALE);

    if (im1.empty() || im2.empty()) {
        std::cerr << "empty image" << std::endl;
        exit(2);
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

    std::cout << good_matches.size() << " of " << corners.size() << " matched" << std::endl;    

    match_corners::draw(good_matches, im1, im2);

    return 0;
}