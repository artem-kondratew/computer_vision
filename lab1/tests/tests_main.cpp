#include <filesystem>

#include <gtest/gtest.h>

#include "max_min_filter/max_min_filter.hpp"


TEST(maxMinFilterTests, maxMinFilter) {
    std::string image_path = std::filesystem::current_path().parent_path() / std::filesystem::path("image.jpg");
    cv::Mat src = cv::imread(image_path, cv::IMREAD_GRAYSCALE);

    cv::Mat res_custom = maxMinFilter(src);

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
    cv::Mat max, min;
    cv::dilate(src, max, kernel);
    cv::erode(src, min, kernel);
    cv::Mat res_cv = max - min;

    cv::Mat diff;
    cv::compare(res_custom, res_cv, diff, cv::CMP_NE);

    EXPECT_FALSE(cv::countNonZero(diff));
}


TEST(maxMinFilterTests, maxMinFilterVectorized) {
    std::string image_path = std::filesystem::current_path().parent_path() / std::filesystem::path("image.jpg");
    cv::Mat src = cv::imread(image_path, cv::IMREAD_GRAYSCALE);

    cv::Mat res_custom = maxMinFilterVectorized(src);

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
    cv::Mat max, min;
    cv::dilate(src, max, kernel);
    cv::erode(src, min, kernel);
    cv::Mat res_cv = max - min;

    cv::Mat diff;
    cv::compare(res_custom, res_cv, diff, cv::CMP_NE);

    EXPECT_FALSE(cv::countNonZero(diff));
}


int main() {
    testing::InitGoogleTest();
    return RUN_ALL_TESTS();
}
