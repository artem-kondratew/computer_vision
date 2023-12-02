#include <iostream>

#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/imgproc.hpp>
#include <opencv4/opencv2/highgui.hpp>


cv::Scalar MAGENTA(255, 0, 255);
cv::Scalar RED(0, 0, 255);
cv::Scalar BLACK(0, 0, 0);
cv::Scalar WHITE(255, 255, 255);


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
    cv::drawContours(copy, contours, -1, MAGENTA);
    cv::circle(copy, center, 4, RED, -1);

    return copy;
}


void task1() {
    cv::Mat img0 = cv::imread("/home/user/Projects/computer_vision/lab3/src1/img0.jpg");
    cv::Mat img1 = cv::imread("/home/user/Projects/computer_vision/lab3/src1/img1.jpg");
    cv::Mat img2 = cv::imread("/home/user/Projects/computer_vision/lab3/src1/img2.jpg");

    cv::Mat t0 = findTarget(img0);
    cv::Mat t1 = findTarget(img1);
    cv::Mat t2 = findTarget(img2);

    cv::imshow("target0", t0);
    cv::imshow("target1", t1);
    cv::imshow("target2", t2);

    cv::waitKey(0);
}


cv::Mat findRed(cv::Mat src) {
    cv::Scalar low_red(0, 100, 100);
    cv::Scalar high_red(20, 255, 255);

    cv::Mat hsv;
    cv::cvtColor(src, hsv, cv::COLOR_BGR2HSV);

    cv::Mat red;
    cv::inRange(hsv, low_red, high_red, red);

    return red;
}


int findMaxContour(std::vector<std::vector<cv::Point>> contours) {
    double p = 0;
    int idx = 0;

    for (int i = 0; i < contours.size(); i++) {
        cv::Moments m = cv::moments(contours[i]);
        if (m.m00 > p) {
            p = m.m00;
            idx = i;
        }
    }

    return idx;
}


cv::Mat findMotor(cv::Mat src) {
    cv::Mat red = findRed(src);
    
    cv::Mat dilate;
    cv::Mat dilate_kernel = cv::getStructuringElement(cv::MORPH_DILATE, {3, 3});
    cv::dilate(red, dilate, dilate);

    std::vector<std::vector<cv::Point>> contours = findContours(dilate, 128);
    int max_contour_idx = findMaxContour(contours);
    cv::Moments moments = cv::moments(contours[max_contour_idx]);

    cv::Mat copy = src.clone();
    cv::drawContours(copy, contours, max_contour_idx, BLACK, 2);
    
    cv::Point2f center = cv::Point2f(static_cast<float>(moments.m10 / (moments.m00 + 1e-5)), static_cast<float>(moments.m01 / (moments.m00 + 1e-5)));
    circle(copy, center, 4, WHITE, -1);

    return copy;
}


void task2() {
    cv::Mat img0 = cv::imread("/home/user/Projects/computer_vision/lab3/src2/img0.jpg");
    cv::Mat img1 = cv::imread("/home/user/Projects/computer_vision/lab3/src2/img1.jpg");
    cv::Mat img2 = cv::imread("/home/user/Projects/computer_vision/lab3/src2/img2.jpg");
    cv::Mat img3 = cv::imread("/home/user/Projects/computer_vision/lab3/src2/img3.jpg");
    cv::Mat img4 = cv::imread("/home/user/Projects/computer_vision/lab3/src2/img4.jpg");

    cv::Mat t0 = findMotor(img0);
    cv::Mat t1 = findMotor(img1);
    cv::Mat t2 = findMotor(img2);
    cv::Mat t3 = findMotor(img3);
    cv::Mat t4 = findMotor(img4);

    cv::imshow("target0", t0);
    cv::imshow("target1", t1);
    cv::imshow("target2", t2);
    cv::imshow("target3", t3);
    cv::imshow("target4", t4);

    cv::waitKey(0);
}


int main() {

    // task1();
    task2();

    return 0;
}
