#ifndef LAB3_HSV_PICKER_HPP
#define LAB3_HSV_PICKER_HPP


#include <iostream>

#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/imgproc.hpp>
#include <opencv4/opencv2/highgui.hpp>


namespace HsvPicker {


void nothing(int x) {
    
}


void picK(cv::Mat image) {

    cv::namedWindow("image");

    // Create trackbars for color change
    // Hue is from 0-179 for Opencv
    cv::createTrackbar("HMin", "image", 0, 179);
    cv::createTrackbar("SMin", "image", 0, 255);
    cv::createTrackbar("VMin", "image", 0, 255);
    cv::createTrackbar("HMax", "image", 0, 179);
    cv::createTrackbar("SMax", "image", 0, 255);
    cv::createTrackbar("VMax", "image", 0, 255);

    // Set default value for Max HSV trackbars
    cv::setTrackbarPos("HMax", "image", 179);
    cv::setTrackbarPos("SMax", "image", 255);
    cv::setTrackbarPos("VMax", "image", 255);

    // Initialize HSV min/max values
    int hMin{}, sMin{}, vMin{}, hMax{}, sMax{}, vMax{};
    int phMin{}, psMin{}, pvMin{}, phMax{}, psMax{}, pvMax{};

    while (true) {

        // Get current positions of all trackbars
        hMin = cv::getTrackbarPos("HMin", "image");
        sMin = cv::getTrackbarPos("SMin", "image");
        hMax = cv::getTrackbarPos("HMax", "image");
        vMin = cv::getTrackbarPos("VMin", "image");
        sMax = cv::getTrackbarPos("SMax", "image");
        vMax = cv::getTrackbarPos("VMax", "image");

        // Set minimum and maximum HSV values to display
        cv::Scalar lower(hMin, sMin, vMin);
        cv::Scalar upper(hMax, sMax, vMax);

        // Convert to HSV format and color threshold
        cv::Mat hsv;
        cv::cvtColor(image, hsv, cv::COLOR_BGR2HSV);
        cv::Mat mask;
        cv::inRange(hsv, lower, upper, mask);
        cv::Mat result;
        cv::bitwise_and(image, image, result, mask);

        // Print if there is a change in HSV value
        if ((phMin != hMin) | (psMin != sMin) | (pvMin != vMin) | (phMax != hMax) | (psMax != sMax) | (pvMax != vMax)) {
            phMin = hMin;
            psMin = sMin;
            pvMin = vMin;
            phMax = hMax;
            psMax = sMax;
            pvMax = vMax;
        }
            

        // Display result image
        cv::imshow("image", result);
        if (cv::waitKey(10) == 27) {
            break;
        }
    }

    cv::Scalar minimum(phMin, psMin, pvMin);
    cv::Scalar maximum(phMax, psMax, pvMax);
    std::cout << "min: " << minimum << std::endl;
    std::cout << "max: " << maximum << std::endl;

    cv::destroyAllWindows();
}

}


#endif // LAB3_HSV_PICKER_HPP
