#ifndef TASK2_TANK_HPP
#define TASK2_TANK_HPP


#include <iostream>

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>


class Tank {
private:
    static const int KEY_UP = 82;
    static const int KEY_DOWN = 84;
    static const int KEY_LEFT = 81;
    static const int KEY_RIGHT = 83;
    static const int KEY_SPACE = 32;
    static const int KEY_ESC = 27;

    cv::Mat tank;
    int w;
    int h;
    cv::Point tank_center;
    cv::Mat back;
    cv::Point back_center;
    int orientation;

public:
    Tank(cv::Mat tank_model);
    void spawn(cv::Mat frame, cv::Point spawn_pt);
    void play();
};


#endif // TASK2_TANK_HPP
