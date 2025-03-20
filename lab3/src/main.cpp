#include <iostream>
#include <numbers>

#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/imgproc.hpp>
#include <opencv4/opencv2/highgui.hpp>


cv::Mat apply_effect(const cv::Mat& f, const cv::Mat& m, float lambda) {
    int w = f.cols;
    int h = f.rows;

    cv::Mat f_float, m_float;
    f.convertTo(f_float, CV_32F);
    m.convertTo(m_float, CV_32F);

    cv::Mat gradx, grady;
    cv::Sobel(f_float, gradx, CV_32F, 1, 0);
    cv::Sobel(f_float, grady, CV_32F, 0, 1);

    cv::Mat gx, gy;
    cv::multiply(gradx, m_float / 100., gx);
    cv::multiply(grady, m_float / 100., gy);

    cv::Mat f_complex;

    cv::merge(std::vector{f_float, cv::Mat(f_float.size(), f_float.type(), cv::Scalar{0})}, f_complex);
    cv::merge(std::vector{gx, cv::Mat(gx.size(), gx.type(), cv::Scalar{0})}, gx);
    cv::merge(std::vector{gy, cv::Mat(gy.size(), gy.type(), cv::Scalar{0})}, gy);
    
    cv::Mat Gx, Gy, F;
    cv::dft(gx, Gx);
    cv::dft(gy, Gy);
    cv::dft(f_complex, F);

    cv::Mat Dx(h, w, CV_32FC2, cv::Scalar(0, 0));
    cv::Mat Dy(h, w, CV_32FC2, cv::Scalar(0, 0));

    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            float imx = std::sin(2 * std::numbers::pi * x / static_cast<float>(w));
            float imy = std::sin(2 * std::numbers::pi * y / static_cast<float>(h));
            Dx.at<cv::Vec2f>(y, x) = cv::Vec2f(0, imx);
            Dy.at<cv::Vec2f>(y, x) = cv::Vec2f(0, imy);
        }
    }

    cv::Mat mul1, mul2, mul3, mul4, U;
    cv::Mat real_ones(h, w, CV_32FC2, cv::Scalar(1, 0));

    cv::mulSpectrums(Dx, Gx, mul1, 0);
    cv::mulSpectrums(Dy, Gy, mul2, 0);
    cv::mulSpectrums(Dx, Dx, mul3, 0); 
    cv::mulSpectrums(Dy, Dy, mul4, 0);

    cv::divSpectrums(F - lambda * (mul1 + mul2), real_ones - lambda * (mul3 + mul4), U, 0);

    cv::Mat u;
    cv::idft(U, u, cv::DFT_SCALE | cv::DFT_REAL_OUTPUT);
    u.convertTo(u, CV_8U);
    
    return u;
}


int main(int argc, char const* argv[]) {
    if (argc != 3) {
        std::cerr << "wrong usage" << std::endl;
        return -1;
    }

    cv::Mat image = cv::imread(argv[1], cv::IMREAD_COLOR);
    cv::Mat mask = cv::imread(argv[2], cv::IMREAD_GRAYSCALE);

    if (image.empty() || mask.empty()) {
        std::cerr << "empty input" << std::endl;
        return -2;
    }

    std::vector<cv::Mat> channels;
    cv::split(image, channels);
    cv::Mat r = channels[0];
    cv::Mat g = channels[1];
    cv::Mat b = channels[2];

    float lambda = 0.5;
    r = apply_effect(r, mask, lambda);
    g = apply_effect(g, mask, lambda);
    b = apply_effect(b, mask, lambda);
    cv::Mat result;
    cv::merge(std::vector{r, g, b}, result);

    cv::imshow("image", image);
    cv::imshow("result", result);
    cv::waitKey(0);
    
    return 0;
}
