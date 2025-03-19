#include <algorithm>
#include <random>

#include <opencv4/opencv2/imgproc.hpp>
#include <opencv4/opencv2/calib3d.hpp>
#include <opencv4/opencv2/features2d.hpp>

#include "include/homography.hpp"


cv::Vec3f getHomogeneousCoords(const cv::Point2f& pt) {
    return cv::Vec3f{pt.x, pt.y, 1};
}


cv::Point2f getPointFromMat(const cv::Mat& mat) {
    return cv::Point2f{mat.at<float>(0, 0), mat.at<float>(1, 0)} / mat.at<float>(2, 0);
}


std::pair<std::vector<cv::Point2f>, std::vector<cv::Point2f>> getRandomSamples(const std::vector<cv::Point2f>& pts1,
                                                                               const std::vector<cv::Point2f>& pts2,
                                                                               const std::vector<size_t>& range,
                                                                               int num) {

    std::vector<size_t> indices;
    std::sample(range.begin(), range.end(), std::back_inserter(indices), num, std::mt19937(std::random_device{}()));

    std::vector<cv::Point2f> sample1, sample2;
    for (auto idx : indices) {
        sample1.push_back(pts1[idx]);
        sample2.push_back(pts2[idx]);
    }

    return std::pair(sample1, sample2);
}


cv::Mat getMatrixA(const std::vector<cv::Point2f>& pts1, const std::vector<cv::Point2f>& pts2) {
    cv::Mat A(2 * pts1.size(), 9, CV_32FC1, cv::Scalar{0});
    for (auto i = 0; i < pts1.size(); i++) {
        const cv::Point2f& p = pts1[i];
        const cv::Point2f& q = pts2[i];

        auto a0 = A.ptr<float>(2 * i);
        a0[0] = q.x;
        a0[1] = q.y;
        a0[2] = 1;
        a0[6] = -p.x * q.x;
        a0[7] = -p.x * q.y;
        a0[8] = -p.x;

        auto a1 = A.ptr<float>(2 * i + 1);
        a1[3] = q.x;
        a1[4] = q.y;
        a1[5] = 1;
        a1[6] = -p.y * q.x;
        a1[7] = -p.y * q.y;
        a1[8] = -p.y;
    }

    return A;
}


void getGoodPoints(const std::vector<cv::Point2f>& points1,
                   const std::vector<cv::Point2f>& points2,
                   std::vector<cv::Point2f>& good_points1,
                   std::vector<cv::Point2f>& good_points2,
                   const std::vector<bool>& mask) {

    for (auto i = 0; i < points1.size(); i++) {
        if (mask[i]) {
            good_points1.push_back(points1[i]);
            good_points2.push_back(points2[i]);
        }
    }
}


cv::Mat findHomographyMatrix(const std::vector<cv::Point2f>& pts1,
                             const std::vector<cv::Point2f>& pts2) {

    cv::Mat A = getMatrixA(pts1, pts2);

    cv::Mat W, U, Vt;
    cv::SVD().compute(A, W, U, Vt, cv::SVD::FULL_UV);

    cv::Vec<float, 9> v = Vt.at<cv::Vec<float, 9>>(Vt.rows-1);

    cv::Mat H(1, 9, CV_32F, v.val);
    H = H.reshape(1, 3) / H.at<float>(H.rows-1, H.cols-1);

    return H;
}


cv::Mat findHomographyMatrixLSM(const std::vector<cv::Point2f>& pts1,
                                const std::vector<cv::Point2f>& pts2,
                                const std::vector<bool>& mask) {

    std::vector<cv::Point2f> good_pts1, good_pts2;
    getGoodPoints(pts1, pts2, good_pts1, good_pts2, mask);

    cv::Mat A = getMatrixA(good_pts1, good_pts2);

    cv::Mat W, U, Vt;
    cv::SVD().compute(A, W, U, Vt, cv::SVD::FULL_UV);

    cv::Vec<float, 9> v = Vt.at<cv::Vec<float, 9>>(Vt.rows-1);

    cv::Mat H(1, 9, CV_32F, v.val);
    H = H.reshape(1, 3) / H.at<float>(2, 2);

    return H;
}


inline float calcRatio(size_t inliers, size_t points) {
    return inliers / static_cast<float>(points);
}


cv::Mat findHomographyMatrixRANSAC(const std::vector<cv::Point2f>& points1,
                                   const std::vector<cv::Point2f>& points2,
                                   float confidence,
                                   float e,
                                   size_t max_iters,
                                   bool use_confidence,
                                   bool reestimation) {  
    size_t sample_size = 4;
    std::vector<bool> mask(points1.size());

    size_t inliers_best = 0;
    cv::Mat H_best;
    std::vector<bool> mask_best;

    std::vector<size_t> range(points1.size());
    std::iota(range.begin(), range.end(), 0);

    for (auto i = 0; i < max_iters; i++) {
        auto samples = getRandomSamples(points1, points2, range, sample_size);

        cv::Mat H = findHomographyMatrix(samples.first, samples.second);

        size_t inliers = 0;

        for (auto j = 0; j < points1.size(); j++) {
            cv::Point2f p = points1[j];
            cv::Vec3f q = getHomogeneousCoords(points2[j]);

            cv::Point2f p_est = getPointFromMat(H * q);

            if (cv::norm(p - p_est) > e) {
                mask[j] = false;
                continue;
            }

            mask[j] = true;
            inliers++;
        }

        if (use_confidence) {
            float ratio = calcRatio(inliers, points1.size());
            if (ratio >= confidence) {
                return reestimation ? findHomographyMatrixLSM(points1, points2, mask) : H;
            }
        }
        
        if (inliers > inliers_best) {
            inliers_best = inliers;
            H_best = H;
            mask_best = mask;
        }
    }

    return reestimation ? findHomographyMatrixLSM(points1, points2, mask_best) : H_best;
}
