#include <iostream>
#include <filesystem>

#include "max_min_filter/max_min_filter.hpp"


int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "wrong usage. correct usage: ./max_min_filter <src_path>" << std::endl;
        exit(1);
    }

    cv::Mat src = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
    cv::Mat result = maxMinFilterVectorized(src);
    cv::imwrite(std::filesystem::path(argv[1]).parent_path() / std::filesystem::path("result.jpg"), result);

    return 0;
}