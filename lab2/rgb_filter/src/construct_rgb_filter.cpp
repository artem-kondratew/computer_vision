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


double percentile(cv::Mat t, int percentile) {
    cv::Mat sorted;
    cv::sort(t, sorted, cv::SORT_EVERY_COLUMN + cv::SORT_ASCENDING);

    int idx = std::round(t.rows * percentile / 100.);

    return sorted.at<double>(idx, 0);
}


Filter construct_filter(cv::Mat train_image, cv::Mat mask) {
    int num_nonzero = cv::countNonZero(mask);

    cv::Mat data(num_nonzero, 3, CV_64FC1);

    int idx = 0;
    for (auto y = 0; y < mask.rows; y++) {
        for (auto x = 0; x < mask.cols; x++) {
            if (mask.at<uint8_t>(y, x) == 255) {
                cv::Vec3b& pixel = train_image.at<cv::Vec3b>(y, x);
                data.at<double>(idx, 0) = pixel[0];
                data.at<double>(idx, 1) = pixel[1];
                data.at<double>(idx, 2) = pixel[2];
                idx++;
            }
        }
    }

    cv::Scalar p0{
        cv::mean(data.col(0))[0],
        cv::mean(data.col(1))[0],
        cv::mean(data.col(2))[0],
        };

    cv::Mat d = data - p0;
    cv::Mat d_T;
    cv::transpose(d, d_T);

    cv::Mat D = d_T * d;

    cv::Mat eigenvalues, eigenvectors;
    cv::eigen(D, eigenvalues, eigenvectors);

    cv::Vec3d v = eigenvectors.at<cv::Vec3d>(0);

    cv::Mat t = d * v;

    double t1 = percentile(t, 5);
    double t2 = percentile(t, 95);

    cv::Mat dp = norm(t * v.t() - d);

    double r = percentile(dp, 95);

    return Filter{v, p0, t1, t2, r};
}


int main(int argc, char* argv[]) {
    if (argc != 4) {
        std::cerr << "wrong usage. correct usage: construct_rgb_filter <path_to_train_image> <path_to_mask> <path_to_json>" << std::endl;
        exit(1);
    }
    cv::Mat train_image = cv::imread(argv[1], cv::IMREAD_COLOR);
    cv::Mat mask = cv::imread(argv[2], cv::IMREAD_GRAYSCALE);

    Filter filter = construct_filter(train_image, mask);
    filter.writeToJson(argv[3]);

    return 0;
}
