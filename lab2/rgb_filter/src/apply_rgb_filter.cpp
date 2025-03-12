#include <iostream>

#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/imgproc.hpp>
#include <opencv4/opencv2/highgui.hpp>

#include "filter.hpp"


template <typename T>
cv::Mat norm(T src) {
    cv::Mat res;
    cv::multiply(src, src, res);
    cv::reduce(res, res, 1, cv::REDUCE_SUM);
    cv::sqrt(res, res);
    return res;
}


cv::Mat apply_filter(cv::Mat test_image, Filter filter, double threshold) {
    cv::Mat data = test_image.reshape(1, test_image.rows * test_image.cols);
    data.convertTo(data, CV_64FC1);

    cv::Mat t = (data - filter.p0) * filter.v;

    cv::Mat dt = cv::abs(t - (filter.t1 + filter.t2) / 2) - (filter.t2 - filter.t1) / 2;
    dt = cv::max(dt, 0);

    cv::Mat dp = norm(t * filter.v.t() - (data - filter.p0));
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

    return 0;
}
