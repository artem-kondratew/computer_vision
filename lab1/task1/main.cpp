#include "robot.hpp"


int main(int argc, char** argv){
    cv::Mat background = cv::imread("/home/user/Projects/computer_vision/lab1/task1/background.jpg");

    std::string path_to_save = "./trajectory.jpg";

    Robot robot(background, path_to_save);

    robot.draw();

    return 0;
}
