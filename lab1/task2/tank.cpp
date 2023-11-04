#include "tank.hpp"


Tank::Tank(cv::Mat tank_model) : tank{tank_model} {
    w = tank.cols;
    h = tank.rows;
    tank_center = cv::Point(w / 2, h / 2);
    orientation = KEY_UP;

    back = cv::Mat::zeros(720, 1280, CV_8UC3);
    back_center = cv::Point(back.cols / 2, back.rows / 2);
}


void Tank::spawn(cv::Mat frame, cv::Point spawn_pt) {
    tank.copyTo(frame(cv::Rect(spawn_pt.x - w / 2, spawn_pt.y - h / 2, w, h)));
}


void Tank::play() {
    int step = 5;
    cv::Mat frame;
    back.copyTo(frame);
    cv::Point spawn_pt = back_center;
    spawn(frame, spawn_pt);
    bool change_flag = false;


    while (true) {
        
        if (change_flag) {
            back.copyTo(frame);
            spawn(frame, spawn_pt);
            change_flag = false;
        }
        imshow("Tanks", frame);
        int key = cv::waitKey(20);

        if (key == KEY_ESC) {
            return;
        }
        if (key == KEY_SPACE) {

        }
        if (key == KEY_UP && spawn_pt.y > h / 2 + step) {
            spawn_pt.y -= step;
            change_flag = true;
        }
        if (key == KEY_DOWN && spawn_pt.y < back.rows - h / 2 - step) {
            spawn_pt.y += step;
            change_flag = true;
        }
        if (key == KEY_LEFT && spawn_pt.x > w / 2 + step) {
            spawn_pt.x -= step;
            change_flag = true;
        }
        if (key == KEY_RIGHT && spawn_pt.x < back.cols - w / 2 - step) {
            spawn_pt.x += step;
            change_flag = true;
        }

        // std::cout << spawn_pt.x << " " << spawn_pt.y << std::endl;
    }
}
