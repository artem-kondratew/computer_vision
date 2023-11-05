#ifndef TASK2_SHELL_HPP
#define TASK2_SHELL_HPP


#include <cstdint>
#include <iostream>
#include <list>

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include "keys.hpp"


class Shell {
private:
    cv::Point pt;
    static const int MAIN_STEP = 10;
    int step;
    bool is_x;
    inline static std::list<Shell*> list;

public:
    Shell(cv::Point shell_pt, int orientation);
    static void createShell(cv::Point shell_pt, int orientation);

    static void increaseSteps(cv::Mat frame);
    static void checkShells();
    static void draw(cv::Mat frame);
};


#endif // TASK2_SHELL_HPP
