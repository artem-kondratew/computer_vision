#include "robot.hpp"


Robot::Robot(cv::Mat background, std::string path_to_save) : back{background}, path_to_save{path_to_save} {
    calcPath();
}


Robot::~Robot() {
    delete[] raw_path;
}


bool Robot::saveTrajectory(cv::Mat frame) {
    return cv::imwrite(path_to_save, frame);
}


void Robot::calcPath() {
    raw_path = new double[back.cols+20];
    for (uint64_t x = 0; x < back.cols; x++) {
        raw_path[x] = std::cos(0.04 * x);
    }
}


uint64_t Robot::calc_y(double raw_path_x) {
    return back.rows / 2 - 100 * raw_path_x;
}


void Robot::drawPath(uint64_t x) {
        cv::Point prev_pt = x == 0 ? cv::Point(0, calc_y(raw_path[0])) : cv::Point(x - 1, calc_y(raw_path[x-1]));
        cv::Point current_pt(x, calc_y(raw_path[x]));
        cv::line(back, prev_pt, current_pt, {255, 0, 0}, 1);
}


cv::Point Robot::rotate(cv::Point pt, uint64_t tx) {
    uint64_t ty = calc_y(raw_path[tx]);
    double theta = 0.04 * tx;

    if (raw_path[tx] < 0) { 
        theta = 3.14 - theta;
    }

    theta = theta - 3.14 * (theta / 2 / 3.14);

    int x = pt.x * std::cos(theta) - pt.y * std::sin(theta) + tx;
    int y = pt.x * std::sin(theta) + pt.y * std::cos(theta) + ty;
    return cv::Point(x, y);
}


void Robot::drawRobot(cv::Mat frame, uint64_t x) {

    cv::Point base_pt1 = rotate(cv::Point(-16, +12), x);
    cv::Point base_pt2 = rotate(cv::Point(-16, -12), x);
    cv::Point base_pt3 = rotate(cv::Point(+16, +12), x);
    cv::Point base_pt4 = rotate(cv::Point(+16, -12), x);
    
    cv::Point back_wheel1 = rotate(cv::Point(-12, 16), x);
    cv::Point back_wheel2 = rotate(cv::Point( -4, 16), x);
    cv::Point back_wheel3 = rotate(cv::Point(-12, 12), x);
    cv::Point back_wheel4 = rotate(cv::Point( -4, 12), x);

    cv::Point back_wheel5 = rotate(cv::Point(-12, -16), x);
    cv::Point back_wheel6 = rotate(cv::Point( -4, -16), x);
    cv::Point back_wheel7 = rotate(cv::Point(-12, -12), x);
    cv::Point back_wheel8 = rotate(cv::Point( -4, -12), x);
    
    cv::Point front_wheel1 = rotate(cv::Point(12, 16), x);
    cv::Point front_wheel2 = rotate(cv::Point( 4, 16), x);
    cv::Point front_wheel3 = rotate(cv::Point(12, 12), x);
    cv::Point front_wheel4 = rotate(cv::Point( 4, 12), x);

    cv::Point front_wheel5 = rotate(cv::Point(12, -16), x);
    cv::Point front_wheel6 = rotate(cv::Point( 4, -16), x);
    cv::Point front_wheel7 = rotate(cv::Point(12, -12), x);
    cv::Point front_wheel8 = rotate(cv::Point( 4, -12), x);

    cv::line(frame, base_pt1, base_pt2, {0, 255, 0}, 1);
    cv::line(frame, base_pt1, base_pt3, {0, 255, 0}, 1);
    cv::line(frame, base_pt2, base_pt4, {0, 255, 0}, 1);
    cv::line(frame, base_pt3, base_pt4, {0, 255, 0}, 1);

    cv::line(frame, back_wheel1, back_wheel2, {0, 255, 0}, 1);
    cv::line(frame, back_wheel1, back_wheel3, {0, 255, 0}, 1);
    cv::line(frame, back_wheel2, back_wheel4, {0, 255, 0}, 1);
    cv::line(frame, back_wheel3, back_wheel4, {0, 255, 0}, 1);

    cv::line(frame, back_wheel5, back_wheel6, {0, 255, 0}, 1);
    cv::line(frame, back_wheel5, back_wheel7, {0, 255, 0}, 1);
    cv::line(frame, back_wheel6, back_wheel8, {0, 255, 0}, 1);
    cv::line(frame, back_wheel7, back_wheel8, {0, 255, 0}, 1);

    cv::line(frame, front_wheel1, front_wheel2, {0, 255, 0}, 1);
    cv::line(frame, front_wheel1, front_wheel3, {0, 255, 0}, 1);
    cv::line(frame, front_wheel2, front_wheel4, {0, 255, 0}, 1);
    cv::line(frame, front_wheel3, front_wheel4, {0, 255, 0}, 1);

    cv::line(frame, front_wheel5, front_wheel6, {0, 255, 0}, 1);
    cv::line(frame, front_wheel5, front_wheel7, {0, 255, 0}, 1);
    cv::line(frame, front_wheel6, front_wheel8, {0, 255, 0}, 1);
    cv::line(frame, front_wheel7, front_wheel8, {0, 255, 0}, 1);
}


void Robot::draw() {
    cv::Mat frame;
    for (uint64_t x = 0; x < back.cols + 1; x++) {
        drawPath(x);
        back.copyTo(frame);
        if (x != back.cols) {
            drawRobot(frame, x);
        }
        cv::imshow("robot_travelling", frame);
        cv::waitKey(20);
    }
    cv::waitKey(500);
}
