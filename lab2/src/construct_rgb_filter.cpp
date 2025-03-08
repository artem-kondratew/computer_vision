#include <iostream>

#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/imgproc.hpp>
#include <opencv4/opencv2/highgui.hpp>


struct Filter {
    cv::Vec3d v;
    cv::Scalar p0;
    double t1, t2;
    double r;

    Filter(cv::Vec3d v, cv::Scalar p0, double t1, double t2, double r) : v{v}, p0{p0}, t1{t1}, t2{t2}, r{r} {}
};


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
    cv::Mat train_image = cv::imread("/home/user/computer_vision/lab2/train.png", cv::IMREAD_COLOR);
    cv::Mat mask = cv::imread("/home/user/computer_vision/lab2/mask.png", cv::IMREAD_GRAYSCALE);

    Filter filter = construct_filter(train_image, mask);

    cv::Mat test_image = cv::imread("/home/user/computer_vision/lab2/test.png", cv::IMREAD_COLOR);

    double threshold = 12;
    cv::Mat result = apply_filter(test_image, filter, threshold);

    cv::imshow("test_image", test_image);
    cv::imshow("result", result);
    cv::waitKey(0);

    return 0;
}
