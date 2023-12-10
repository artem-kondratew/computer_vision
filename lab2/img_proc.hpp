#ifndef LAB2_IMG_PROC_HPP
#define LAB2_IMG_PROC_HPP


#include <iostream>

#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/imgproc.hpp>
#include <opencv4/opencv2/highgui.hpp>


namespace ImgProc {

using filter_type = int;

const filter_type box_filter = 0;
const filter_type gauss_filter = 1;
const filter_type laplacian_filter = 2;

const int HIGH = 255;


cv::Mat convolution(const cv::Mat img, const float* kernel, const int dx, const int dy) {

    if (!(dx % 2) || !(dy % 2)) {
        return cv::Mat({0, 0});
    }

    const int nx = (dx - 1) / 2;
    const int ny = (dy - 1) / 2;

    const int w = img.cols;
    const int h = img.rows;

    cv::Mat empty = cv::Mat::zeros({w, h}, CV_8UC1);

    int kernel_size = dx * dy;
    uint8_t submatrix[kernel_size];

    for (int y = ny; y < h - ny; y++) {
        for (int x = nx; x < w - nx; x++) {
            int cell = 0;
            float sum = 0;
            for (int i = y - ny; i < y + ny + 1; i++) {
                for (int j = x - nx; j < x + nx + 1; j++) {
                    sum += img.data[i*w+j] * kernel[cell];
                    cell++;
                }
            }
            int res = static_cast<int>(std::abs(sum));
            empty.data[y*w+x] = res > HIGH ? HIGH : res;
        }
    }

    return empty;
}


cv::Mat boxFilter(const cv::Mat img, const int dx, const int dy) {
    int kernel_size = dx * dy;
    float kernel[kernel_size];

    float value = 1 / static_cast<float>(kernel_size);

    for (int i = 0; i < kernel_size; i++) {
        kernel[i] = value;
    }

    return convolution(img, kernel, dx, dy);
}


cv::Mat laplacian(const cv::Mat img) {
    const float kernel[] = {
        0,  1, 0,
        1, -4, 1,
        0,  1, 0,
    };

    return convolution(img, kernel, 3, 3);
}


cv::Mat findDiff(const cv::Mat img1, const cv::Mat img2) {
    if (img1.rows != img2.rows || img1.cols != img2.cols || img1.channels() != img2.channels()) {
        return cv::Mat({0, 0});
    }

    size_t cnt = 0;

    cv::Mat diff = cv::Mat::zeros({img1.cols, img1.rows}, CV_8UC1);

    for (size_t i = 0; i < diff.rows * diff.cols; i++) {
        float df = std::abs(img1.data[i] - img2.data[i]);
        if (df < 1.1) {
            cnt++;
        }
        else {
            diff.data[i] = df;
        }
    }

    float accuracy = cnt / static_cast<float>(img1.rows * img1.cols);
    std::cout << "accuracy = " << accuracy << std::endl;

    return diff;
}


cv::Mat findLogDiff(const cv::Mat img1, const cv::Mat img2) {
    if (img1.rows != img2.rows || img1.cols != img2.cols || img1.channels() != img2.channels()) {
        return cv::Mat({0, 0});
    }

    cv::Mat diff = cv::Mat::zeros({img1.cols, img1.rows}, CV_8UC1);

    for (size_t i = 0; i < diff.rows * diff.cols; i++) {
        float df = std::abs(img1.data[i] - img2.data[i]);
        float df_log = 100 * std::log(1 + df);
        diff.data[i] = df_log > HIGH ? HIGH : df_log;
    }

    return diff;
}


cv::Mat unsharpMask(const cv::Mat img, const float alpha, const filter_type ft) {
    cv::Mat blur;
    if (ft == box_filter) {
        blur = boxFilter(img, 3, 3);
    }
    else if (ft == gauss_filter) {
        cv::GaussianBlur(img, blur, {3, 3}, 0);
    }
    else if (ft == laplacian_filter) {
        blur = laplacian(img);
    }
    else {
        return cv::Mat({0, 0});
    }

    cv::Mat empty = cv::Mat::zeros({img.cols, img.rows}, CV_8UC1);

    for (int i = 0; i < empty.rows * empty.cols; i++) {
        int value  = (1 + alpha) * img.data[i] - alpha * blur.data[i];
        value = value > HIGH ? HIGH : value;
        value = value < 0 ? 0 : value;
        empty.data[i] = value;
    }

    return empty;
}

}


#endif // LAB2_IMG_PROC_HPP
