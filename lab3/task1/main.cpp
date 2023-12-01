#include <iostream>

#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/imgproc.hpp>
#include <opencv4/opencv2/highgui.hpp>


std::vector<std::vector<cv::Point>> findContours(cv::Mat gray, int thr) {
    cv::Mat thresh;
    cv::threshold(gray, thresh, thr, 255, cv::THRESH_BINARY);

    cv::Mat erode;
    cv::Mat erode_kernel = cv::getStructuringElement(cv::MORPH_ERODE, {3, 3});
    cv::erode(thresh, erode, erode_kernel);

    cv::Mat dilate;
    cv::Mat dilate_kernel = cv::getStructuringElement(cv::MORPH_DILATE, {3, 3});
    cv::dilate(erode, dilate, dilate_kernel);

    cv::Mat canny;
    cv::Canny(dilate, canny, 10, 50);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(canny.clone(), contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

    return contours;
}


cv::Mat findTarget(const cv::Mat src) {
    cv::Scalar magenta(255, 0, 255);
    cv::Scalar red(0, 0, 255);

    cv::Mat gray;
    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);

    int sum = 0;
    for (int i = 0; i < src.rows * src.cols; i++) {
        sum += gray.data[i];
    }
    int thr = sum / (src.rows * src.cols);

    int cnt = 2;
    std::vector<std::vector<cv::Point>> contours;
    
    while (cnt > 1) {
        contours = findContours(gray, thr);
        cnt = contours.size();
        thr++;
    }

    cv::Moments moments = cv::moments(contours[0]);
    cv::Point2f center = cv::Point2f(static_cast<float>(moments.m10 / (moments.m00 + 1e-5)), static_cast<float>(moments.m01 / (moments.m00 + 1e-5)));

    cv::Mat copy = src.clone();
    cv::drawContours(copy, contours, -1, magenta);
    circle(copy, center, 4, red, -1);

    return copy;
}


int main() {

    cv::Mat img0 = cv::imread("/home/user/Projects/computer_vision/lab3/task1/src/img0.jpg");
    cv::Mat img1 = cv::imread("/home/user/Projects/computer_vision/lab3/task1/src/img1.jpg");
    cv::Mat img2 = cv::imread("/home/user/Projects/computer_vision/lab3/task1/src/img2.jpg");

    cv::Mat t0 = findTarget(img0);
    cv::Mat t1 = findTarget(img1);
    cv::Mat t2 = findTarget(img2);

    cv::imshow("target0", t0);
    cv::imshow("target1", t1);
    cv::imshow("target2", t2);

    cv::waitKey(0);

    return 0;
}
