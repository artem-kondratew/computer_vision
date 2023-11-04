#include "robot.hpp"


int main(int argc, char** argv){
    cv::Mat background = cv::imread("./background.jpg");

    std::string path_to_save = "./trajectory.jpg";

    Robot robot(background, path_to_save);

    robot.draw();

    return 0;
}
