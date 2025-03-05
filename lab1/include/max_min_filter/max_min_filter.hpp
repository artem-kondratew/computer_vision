#ifndef MAX_MIN_FILTER_HPP
#define MAX_MIN_FILTER_HPP


#include <opencv4/opencv2/opencv.hpp>


cv::Mat maxMinFilter(const cv::Mat& src);
cv::Mat maxMinFilterVectorized(const cv::Mat& src);


#endif  // MAX_MIN_FILTER_HPP
