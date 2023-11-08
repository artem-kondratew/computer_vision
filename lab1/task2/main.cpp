#include "tank.hpp"


#if false
void generate_model() {
    cv::Mat model = cv::Mat::zeros(141, 141, CV_8UC3);

    cv::Point base_pt1 = cv::Point(30, 40);
    cv::Point base_pt4 = cv::Point(30+81, 40+101);
    
    cv::Point back_wheel1 = cv::Point(14, 40+8);
    cv::Point back_wheel4 = cv::Point(30, 40+8 + 101 - 16);

    cv::Point back_wheel5 = cv::Point(112+14, 40+8);
    cv::Point back_wheel8 = cv::Point(140-16-14, 40+8 + 101 - 16);
    
    cv::Point front_wheel1 = cv::Point(56-24+14, 40+25+25+1-36);
    cv::Point front_wheel4 = cv::Point(56+24+14, 40+25+25+1+36);

    cv::Point front_wheel5 = cv::Point(56-6+14, 0);
    cv::Point front_wheel8 = cv::Point(56+6+14, 40+25+25+1-35);

    cv::Rect base(base_pt1, base_pt4);
    cv::Rect b1(back_wheel1, back_wheel4);
    cv::Rect b2(back_wheel5, back_wheel8);
    cv::Rect f1(front_wheel1, front_wheel4);
    cv::Rect f2(front_wheel5, front_wheel8);

    int c = 90;

    cv::rectangle(model, base, {37, 169, 37}, -1);
    cv::rectangle(model, b1, {c, c, c}, -1);
    cv::rectangle(model, b2, {c, c, c}, -1);
    cv::rectangle(model, f1, {24, 108, 24}, -1);
    cv::rectangle(model, f2, {24, 108, 24}, -1);

    cv::imshow("model", model);
    cv::waitKey(0);

    cv::imwrite("./tank.jpg", model);
}
#endif


int main(int argc, char** argv) {

    cv::Mat tank_model = cv::imread("/home/user/Projects/computer_vision/lab1/task2/tank.jpg");
    
    Tank tank(tank_model);

    tank.play();

    return 0;
}
