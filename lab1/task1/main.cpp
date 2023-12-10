#include <iostream>
#include "robot.hpp"


int main(int argc, char** argv){
    if (argc != 3) {
        std::cout << "usage: ./lab1 path_to_background path_to_save" << std::endl;
        return 1;
    }

    cv::Mat background = cv::imread(argv[1]);

    std::string path_to_save = argv[2];

    std::cout << path_to_save << std::endl;

    Robot robot(background, path_to_save);

    robot.draw();

    return 0;
}
