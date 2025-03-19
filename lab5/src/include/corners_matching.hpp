#ifndef MATCH_CORNERS_HPP
#define MATCH_CORNERS_HPP


#include <iostream>
#include <random>

#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/imgproc.hpp>
#include <opencv4/opencv2/highgui.hpp>


namespace match_corners {

using match = std::pair<cv::Point2f, cv::Point2f>;

std::vector<cv::Mat> createGaussianPyramid(const cv::Mat& im, int nlevels=-1);

void findMatchedCorners(const std::vector<cv::Point2f>& corners,
                        const std::vector<cv::Mat>& pyr1,
                        const std::vector<cv::Mat>& pyr2,
                        std::vector<match>& matches,
                        std::vector<bool>& mask);

void matchCorners(const cv::Mat& im1,
                  const cv::Mat& im2,
                  const std::vector<cv::Point2f>& corners,
                  std::vector<match>& matches,
                  std::vector<bool>& mask);

void draw(const std::vector<match>& matches, cv::Mat im1, cv::Mat im2, bool wait=true);

}  // match_corners


#endif  // MATCH_CORNERS_HPP
