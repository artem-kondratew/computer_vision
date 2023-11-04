#ifndef LAB1_ROBOT_HPP
#define LAB1_ROBOT_HPP


#include <cstdint>
#include <iostream>
#include <string>

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>


class Robot {
private:
    cv::Mat back;
    double* raw_path;
    std::string path_to_save;

public:
    Robot(cv::Mat background, std::string path_to_save);
    ~Robot();
    Robot(const Robot& other) = delete;
    Robot(Robot&& other) = delete;

    Robot& operator=(const Robot& other) = delete;
    Robot& operator=(Robot&& other) = delete;

private:
    bool saveTrajectory(cv::Mat frame);
    void calcPath();
    uint64_t calc_y(double raw_path_x);
    void drawPath(uint64_t x);
    cv::Point rotate(cv::Point pt, uint64_t tx);
    void drawRobot(cv::Mat frame, uint64_t x);
    
public:
    void draw();
};


#endif // LAB1_ROBOT_HPP
