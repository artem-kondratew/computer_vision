#include <chrono>

#include "img_proc.hpp"


int main(int argc, char** argv) {

    cv::Mat src = cv::imread("/home/user/Pictures/lenna.jpg");
    cv::Mat gray;
    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);

    auto box_filter_start = std::chrono::system_clock::now();
    cv::Mat box = ImgProc::boxFilter(gray, 3, 3);
    auto box_filter_stop = std::chrono::system_clock::now();

    auto cv_box_filter_start = std::chrono::system_clock::now();
    cv::Mat cvbox;
    cv::boxFilter(gray, cvbox, -1, {3, 3});
    auto cv_box_filter_stop = std::chrono::system_clock::now();

    cv::Mat boxes_diff = ImgProc::findDiff(cvbox, box);
    auto box_filter_duration = std::chrono::duration_cast<std::chrono::nanoseconds>(box_filter_stop - box_filter_start).count();
    auto cv_box_filter_duration = std::chrono::duration_cast<std::chrono::nanoseconds>(cv_box_filter_stop - cv_box_filter_start).count();

    std::cout << "box_filter_duration = " << box_filter_duration << " ns" << std::endl;
    std::cout << "cv_box_filter_duration = " << cv_box_filter_duration << " ns" << std::endl;
    std::cout << "k = " << static_cast<float>(box_filter_duration) / cv_box_filter_duration << std::endl;
    
    cv::imshow("gray", gray);
    cv::imshow("cvbox", cvbox);
    cv::imshow("box", box);
    cv::imshow("boxes_diff", boxes_diff);

    cv::Mat gauss;
    cv::GaussianBlur(gray, gauss, {3, 3}, 0);
    cv::Mat box_gauss_diff = ImgProc::findDiff(gauss, cvbox);
    cv::Mat box_gauss_diff_log = ImgProc::findLogDiff(gauss, cvbox);

    cv::imshow("gauss", gauss);
    cv::imshow("box_gauss_diff", box_gauss_diff);
    cv::imshow("box_gauss_diff_log", box_gauss_diff_log);

    cv::waitKey(0);
}
