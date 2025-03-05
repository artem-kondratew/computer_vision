#include <iostream>

#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/imgproc.hpp>
#include <opencv4/opencv2/highgui.hpp>

#include "hsv_picker.hpp"


const cv::Scalar MAGENTA(255, 0, 255);

const cv::Scalar BLUE(255, 0, 0);
const cv::Scalar GREEN(0, 255, 0);
const cv::Scalar RED(0, 0, 255);

const cv::Scalar BLACK(0, 0, 0);
const cv::Scalar WHITE(255, 255, 255);


std::vector<std::vector<cv::Point>> findContours(const cv::Mat gray, const int thr) {
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


cv::Mat findRed(const cv::Mat src) {
    cv::Scalar low_red(0, 100, 100);
    cv::Scalar high_red(20, 255, 255);

    cv::Mat hsv;
    cv::cvtColor(src, hsv, cv::COLOR_BGR2HSV);

    cv::Mat red;
    cv::inRange(hsv, low_red, high_red, red);

    return red;
}


int findMaxContour(const std::vector<std::vector<cv::Point>> contours) {
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


cv::Mat findMotor(const cv::Mat src) {
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


cv::Point findLamp(cv::Mat src) {
    cv::Mat gray;
    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);

    int thr = 250;

    std::vector<std::vector<cv::Point>> contours = findContours(gray, thr);

    cv::drawContours(src, contours, -1, MAGENTA, 2, 2);

    cv::Moments m = cv::moments(contours[0]);
    int x = m.m10 / m.m00;
    int y = m.m01 / m.m00;
    
    return cv::Point(x, y);
}


cv::Mat morph(const cv::Mat src) {
    cv::Mat morph;
    cv::Mat erode_kernel = cv::getStructuringElement(cv::MORPH_ERODE, {3, 3});
    cv::Mat dilate_kernel = cv::getStructuringElement(cv::MORPH_ERODE, {3, 3});

    cv::erode(src, morph, erode_kernel);
    cv::dilate(morph, morph, dilate_kernel);

    return morph;
}


std::vector<std::vector<cv::Point>> eraseSmallContours(std::vector<std::vector<cv::Point>> contours, int area) {
    for (int i = 0; i < contours.size(); i++) {
        if (cv::contourArea(contours[i]) < area) {
            contours.erase(contours.begin() + i);
            i--;
        }
    }

    return contours;
}


std::vector<cv::Point> findCenters(std::vector<std::vector<cv::Point>> contours) {
    std::vector<cv::Point> centers;

    for (int i = 0; i < contours.size(); i++) {
        cv::Moments m = cv::moments(contours[i]);
        int x = m.m10 / m.m00;
        int y = m.m01 / m.m00;
        centers.push_back(cv::Point(x, y));
    }

    return centers;
}


std::vector<float> findDistances(std::vector<cv::Point> centers, cv::Point lamp_center) {
    std::vector<float> dist;

    for (int i = 0; i < centers.size(); i++) {
        float d = cv::norm(centers[i] - lamp_center);
        dist.push_back(d);
    }

    return dist;
}


cv::Point findMinDist(std::vector<std::vector<cv::Point>> contours, cv::Point lamp_center) {

    std::vector<cv::Point> centers = findCenters(contours);
    std::vector<float> dist = findDistances(centers, lamp_center);

    int idx = -1;
    float min_dist = FLT_MAX;

    for (int i = 0; i < dist.size(); i++) {
        if (dist[i] < min_dist && dist[i] > 50) {
            idx = i;
            min_dist = dist[i];
        }
    }

    return centers[idx];
}


void task3() {
    cv::Mat src = cv::imread("/home/user/Projects/computer_vision/lab3/src3/img0.jpg");

    cv::Mat hsv;
    cv::cvtColor(src, hsv, cv::COLOR_BGR2HSV);

    cv::Scalar red_lowerb(0, 50, 50);
    cv::Scalar red_upperb(10, 255, 255);

    cv::Scalar red_lowerb_end(170, 50, 50);
    cv::Scalar red_upperb_end(179, 255, 255);

    cv::Scalar green_lowerb(70, 50, 50);
    cv::Scalar green_upperb(80, 255, 255);

    cv::Scalar blue_lowerb(90, 50, 50);
    cv::Scalar blue_upperb(130, 255, 255);

    cv::Mat red, red_0, red_1, green, blue;

    cv::inRange(hsv, red_lowerb, red_upperb, red_0);
    cv::inRange(hsv, red_lowerb_end, red_upperb_end, red_1);
    cv::bitwise_or(red_0, red_1, red);
    cv::inRange(hsv, green_lowerb, green_upperb, green);
    cv::inRange(hsv, blue_lowerb, blue_upperb, blue);

    std::vector<std::vector<cv::Point>> red_cont, green_cont, blue_cont;
    cv::findContours(red, red_cont, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
    cv::findContours(green, green_cont, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
    cv::findContours(blue, blue_cont, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

    red_cont = eraseSmallContours(red_cont, 100);
    green_cont = eraseSmallContours(green_cont, 200);
    blue_cont = eraseSmallContours(blue_cont, 100);

    cv::drawContours(src, red_cont, -1, RED, 1);
    cv::drawContours(src, green_cont, -1, GREEN, 1);
    cv::drawContours(src, blue_cont, -1, BLUE, 1);

    cv::Point lamp_center = findLamp(src);

    // cv::imshow("red", red);
    // cv::imshow("blue", blue);
    // cv::imshow("green", green);

    cv::Point red_center = findMinDist(red_cont, lamp_center);
    cv::Point green_center = findMinDist(green_cont, lamp_center);
    cv::Point blue_center = findMinDist(blue_cont, lamp_center);

    cv::line(src, lamp_center, red_center, MAGENTA, 2);
    cv::line(src, lamp_center, green_center, MAGENTA, 2);
    cv::line(src, lamp_center, blue_center, MAGENTA, 2);

    cv::circle(src, lamp_center, 4, MAGENTA, 2);
    cv::circle(src, red_center, 4, MAGENTA, 2);
    cv::circle(src, green_center, 4, MAGENTA, 2);
    cv::circle(src, blue_center, 4, MAGENTA, 2);

    cv::imshow("result", src);

    cv::waitKey(0);
}


void task4() {
    cv::Mat src = cv::imread("/home/user/Projects/computer_vision/lab3/src4/src.jpg");

    cv::Mat gray;
    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);

    cv::Mat threshold;
    cv::threshold(gray, threshold, 250, 255, cv::THRESH_BINARY_INV);

    cv::Mat tmpl = cv::imread("/home/user/Projects/computer_vision/lab3/src4/template.jpg");

    cv::Mat tmpl_gray;
    cv::cvtColor(tmpl, tmpl_gray, cv::COLOR_BGR2GRAY);

    cv::threshold(tmpl_gray, tmpl_gray, 220, 255, cv::THRESH_BINARY);   

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(threshold, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

    std::vector<std::vector<cv::Point>> tmpl_contours;
    cv::findContours(tmpl_gray, tmpl_contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
    cv::drawContours(tmpl, tmpl_contours, -1, MAGENTA, 1);
    

    double tmpl_area = cv::contourArea(tmpl_contours[0]);
    std::cout << tmpl_area << std::endl;

    for (int i = 0; i < contours.size(); i++) {
        if (cv::contourArea(contours[i]) < 50) {
            contours.erase(contours.begin() + i);
            i--;
            continue;
        }

        cv::Mat key = cv::Mat::zeros({src.cols, src.rows}, CV_8UC3);
        cv::drawContours(key, contours, i, WHITE, -1);

        double area = cv::contourArea(contours[i]);

        double diff = area / tmpl_area;

        cv::Moments m = cv::moments(contours[i]);
        int x = m.m10 / m.m00;
        int y = m.m01 / m.m00;

        cv::Scalar color = 0.9 < diff && diff < 1.07 ? GREEN : RED;
        cv::circle(src, {x, y}, 10, color, -1);
    }

    cv::imshow("src", src);

    cv::waitKey(0);
}


int main() {

    // task1();
    // task2();
    // task3();
    task4();

    return 0;
}
