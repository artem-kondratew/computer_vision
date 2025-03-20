#include <iostream>

#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/imgproc.hpp>
#include <opencv4/opencv2/highgui.hpp>

#include "filter.hpp"


cv::Mat apply_filter(const cv::Mat& test_image, const Filter& filter, double threshold) {
    cv::Mat data = test_image.reshape(1, test_image.rows * test_image.cols);
    data.convertTo(data, CV_64FC1);

    cv::Mat d = data - filter.p0;

    cv::Mat t = d * filter.v;

    cv::Mat dt = cv::abs(t - (filter.t1 + filter.t2) / 2) - (filter.t2 - filter.t1) / 2;
    dt = cv::max(dt, 0);

    cv::Mat dp(d.rows, 1, CV_64FC1);
    for (auto i = 0; i < d.rows; i++) {
        dp.at<double>(i, 0) = cv::norm(d.at<cv::Vec3d>(i, 0).cross(filter.v)) / std::pow(cv::norm(filter.v), 2);
    }
    dp = cv::max(dp - filter.r, 0);

    cv::Mat err = dp + dt;

    cv::Mat mask = err < threshold;
    mask.convertTo(mask, CV_8UC1);
    mask = mask.reshape(1, {test_image.rows, test_image.cols});

    cv::Mat result(test_image.size(), CV_8UC1, cv::Scalar{0});
    result.setTo(255, mask);
    
    return result;
}


int main(int argc, char* argv[]) {
    if (argc != 4) {
        std::cerr << "wrong usage. correct usage: apply_rgb_filter <path_to_test_image> <path_to_json> <result_path>" << std::endl;
        exit(1);
    }

    cv::Mat test_image = cv::imread(argv[1], cv::IMREAD_COLOR);

    Filter filter = Filter::readFromJson(argv[2]);

    double threshold = 12;
    cv::Mat result = apply_filter(test_image, filter, threshold);

    cv::imwrite(argv[3], result);

    cv::imshow("result", result);
    cv::waitKey(0);

    return 0;
}
