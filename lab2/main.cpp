#include <chrono>

#include "img_proc.hpp"


int main(int argc, char** argv) {

    cv::Mat src = cv::imread("/home/user/Pictures/lenna.jpg");
    cv::Mat gray;
    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);

/*--------------------------------------------------------------------------------------------------------------------------------*/
    
    /* Using custom box filter */
    auto box_filter_start = std::chrono::system_clock::now();
    cv::Mat box = ImgProc::boxFilter(gray, 3, 3);
    auto box_filter_stop = std::chrono::system_clock::now();

    /* Using builtin box filter */
    auto cv_box_filter_start = std::chrono::system_clock::now();
    cv::Mat cvbox;
    cv::boxFilter(gray, cvbox, -1, {3, 3});
    auto cv_box_filter_stop = std::chrono::system_clock::now();

    /* Compare custom box filter and builtin box filter */
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
    
/*--------------------------------------------------------------------------------------------------------------------------------*/

    /* Using builtin gauss filter */
    cv::Mat gauss;
    cv::GaussianBlur(gray, gauss, {3, 3}, 0);

    /* Compare gauss filter and builtin box filter */
    cv::Mat box_gauss_diff = ImgProc::findDiff(gauss, cvbox);
    cv::Mat box_gauss_diff_log = ImgProc::findLogDiff(gauss, cvbox);

    cv::imshow("gauss", gauss);
    cv::imshow("box_gauss_diff", box_gauss_diff);
    cv::imshow("box_gauss_diff_log", box_gauss_diff_log);

/*--------------------------------------------------------------------------------------------------------------------------------*/

    /* Using custom unsharp mask filters with box filter and gauss filter */
    cv::Mat unsharp_mask_box = ImgProc::unsharpMask(gray, 2, ImgProc::box_filter);
    cv::Mat unsharp_mask_gauss = ImgProc::unsharpMask(gray, 2, ImgProc::gauss_filter);

    /* Compare unsharp mask filters */
    cv::Mat unsharp_mask_diff = ImgProc::findDiff(unsharp_mask_box, unsharp_mask_gauss);

    cv::imshow("unsharp_mask_box", unsharp_mask_box);
    cv::imshow("unsharp_mask_gauss", unsharp_mask_gauss);
    cv::imshow("unsharp_mask_diff", unsharp_mask_diff);

/*--------------------------------------------------------------------------------------------------------------------------------*/

    /* Using custom laplace filter */
    cv::Mat laplacian = ImgProc::laplacian(gray);

    cv::imshow("laplacian", laplacian);

/*--------------------------------------------------------------------------------------------------------------------------------*/

    /* Using custom unsharp mask filter with laplace filter */
    cv::Mat laplacian_unsharp_mask = ImgProc::unsharpMask(gray, 2, ImgProc::laplacian_filter);

    cv::imshow("laplacian_unsharp_mask", laplacian_unsharp_mask);

/*--------------------------------------------------------------------------------------------------------------------------------*/

    cv::waitKey(0);
}
