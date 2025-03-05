#ifndef TASK2_TANK_HPP
#define TASK2_TANK_HPP


#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include "keys.hpp"
#include "shell.hpp"


class Tank {
private:
    static const int step = 5;

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
    bool checkMoveUp(cv::Point spawn_pt, int new_w, int new_h);
    bool checkMoveDown(cv::Point spawn_pt, int new_w, int new_h);
    bool checkMoveLeft(cv::Point spawn_pt, int new_w, int new_h);
    bool checkMoveRight(cv::Point spawn_pt, int new_w, int new_h);
    bool checkSpawnPosition(cv::Point spawn_pt, int new_w, int new_h);
    bool rotate(cv::Point spawn_pt, int r);
    void createShell(cv::Point spawn_pt);
    void play();
};


#endif // TASK2_TANK_HPP
