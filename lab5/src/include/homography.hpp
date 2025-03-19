#ifndef FIND_HOMOGRAPHY_MATRIX_HPP
#define FIND_HOMOGRAPHY_MATRIX_HPP


#include <opencv4/opencv2/imgproc.hpp>
#include <opencv4/opencv2/calib3d.hpp>
#include <opencv4/opencv2/features2d.hpp>

#include "corners_matching.hpp"


cv::Vec3f getHomogeneousCoords(const cv::Point2f& pt);
cv::Point2f getPointFromMat(const cv::Mat& mat);


std::pair<std::vector<cv::Point2f>, std::vector<cv::Point2f>> getRandomSamples(const std::vector<cv::Point2f>& pts1,
                                                                               const std::vector<cv::Point2f>& pts2,
                                                                               const std::vector<size_t>& range,
                                                                               int num);

cv::Mat getMatrixA(const std::vector<cv::Point2f>& pts1, const std::vector<cv::Point2f>& pts2);

void getGoodPoints(const std::vector<cv::Point2f>& points1,
                   const std::vector<cv::Point2f>& points2,
                   std::vector<cv::Point2f>& good_points1,
                   std::vector<cv::Point2f>& good_points2,
                   const std::vector<bool>& mask);

cv::Mat findHomographyMatrix(const std::vector<cv::Point2f>& pts1,
                             const std::vector<cv::Point2f>& pts2);

cv::Mat findHomographyMatrixLSM(const std::vector<cv::Point2f>& pts1,
                                const std::vector<cv::Point2f>& pts2,
                                const std::vector<bool>& mask);

inline float calcRatio(size_t inliers, size_t points);

cv::Mat findHomographyMatrixRANSAC(const std::vector<cv::Point2f>& points1,
                                   const std::vector<cv::Point2f>& points2,
                                   float confidence,
                                   float e,
                                   size_t max_iters,
                                   bool use_confidence,
                                   bool reestimation);

#endif  // FIND_HOMOGRAPHY_MATRIX_HPP
