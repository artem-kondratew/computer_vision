#include <chrono>
#include <iostream>

#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/imgproc.hpp>
#include <opencv4/opencv2/highgui.hpp>
#include <opencv4/opencv2/core/hal/intrin.hpp>
#include <opencv4/opencv2/core/hal/intrin_sse.hpp>


template <typename T>
T vmax(T first, T second) {
    return first > second ? first : second;
}


template <typename T, typename ... Args>
T vmax(T first, T second, Args ... tail) {
    return vmax(first > second ? first : second, tail...);
}


template <typename T>
T vmin(T first, T second) {
    return first < second ? first : second;
}


template <typename T, typename ... Args>
T vmin(T first, T second, Args ... tail) {
    return vmin(first < second ? first : second, tail...);
}


cv::Mat maxMinFilter(const cv::Mat& src) {
    int h = src.rows;
    int w = src.cols;

    cv::Mat dst(h, w, CV_8UC1);

    for (auto y = 0; y < h; y++) {
        auto ps0 = src.ptr<uint8_t>(std::clamp(y - 2, 0, h - 1));
        auto ps1 = src.ptr<uint8_t>(std::clamp(y - 1, 0, h - 1));
        auto ps2 = src.ptr<uint8_t>(std::clamp(y + 0, 0, h - 1));
        auto ps3 = src.ptr<uint8_t>(std::clamp(y + 1, 0, h - 1));
        auto ps4 = src.ptr<uint8_t>(std::clamp(y + 2, 0, h - 1));

        auto pd = dst.ptr<uint8_t>(y);

        for (auto x = 0; x < w; x++) {
            auto x0 = std::clamp(x - 2, 0, w - 1);
            auto x1 = std::clamp(x - 1, 0, w - 1);
            auto x2 = std::clamp(x + 0, 0, w - 1);
            auto x3 = std::clamp(x + 1, 0, w - 1);
            auto x4 = std::clamp(x + 2, 0, w - 1);

            pd[x] = vmax(
                ps0[x0], ps0[x1], ps0[x2], ps0[x3], ps0[x4],
                ps1[x0], ps1[x1], ps1[x2], ps1[x3], ps1[x4],
                ps2[x0], ps2[x1], ps2[x2], ps2[x3], ps2[x4],
                ps3[x0], ps3[x1], ps3[x2], ps3[x3], ps3[x4],
                ps4[x0], ps4[x1], ps4[x2], ps4[x3], ps4[x4]
            ) - vmin(
                ps0[x0], ps0[x1], ps0[x2], ps0[x3], ps0[x4],
                ps1[x0], ps1[x1], ps1[x2], ps1[x3], ps1[x4],
                ps2[x0], ps2[x1], ps2[x2], ps2[x3], ps2[x4],
                ps3[x0], ps3[x1], ps3[x2], ps3[x3], ps3[x4],
                ps4[x0], ps4[x1], ps4[x2], ps4[x3], ps4[x4]
            );
        }
    }

    return dst;
}


cv::Mat maxMinFilterVectorized(const cv::Mat& src) {
    int h = src.rows;
    int w = src.cols;

    cv::Mat dst(h, w, CV_8UC1);

    uint8_t buf_max[w];
    uint8_t buf_min[w];

    for (auto y = 0; y < h; y++) {
        auto ps0 = src.ptr<uint8_t>(std::clamp(y - 2, 0, h - 1));
        auto ps1 = src.ptr<uint8_t>(std::clamp(y - 1, 0, h - 1));
        auto ps2 = src.ptr<uint8_t>(std::clamp(y + 0, 0, h - 1));
        auto ps3 = src.ptr<uint8_t>(std::clamp(y + 1, 0, h - 1));
        auto ps4 = src.ptr<uint8_t>(std::clamp(y + 2, 0, h - 1));

        auto pd = dst.ptr<uint8_t>(y);

        int x = 0;

        for (; x < w - 15; x += 16) {
            cv::v_uint8x16 u0 = cv::v_load(ps0 + x);
            cv::v_uint8x16 max = u0;
            cv::v_uint8x16 min = u0;

            cv::v_uint8x16 u1 = cv::v_load(ps1 + x);
            max = cv::v_max(max, u1);
            min = cv::v_min(min, u1);

            cv::v_uint8x16 u2 = cv::v_load(ps2 + x);
            max = cv::v_max(max, u2);
            min = cv::v_min(min, u2);

            cv::v_uint8x16 u3 = cv::v_load(ps3 + x);
            max = cv::v_max(max, u3);
            min = cv::v_min(min, u3);

            cv::v_uint8x16 u4 = cv::v_load(ps4 + x);
            max = cv::v_max(max, u4);
            min = cv::v_min(min, u4);

            cv::v_store(buf_max + x, max);
            cv::v_store(buf_min + x, min);
        }

        for (; x < w; x++) {
            buf_max[x] = vmax(ps0[x], ps1[x], ps2[x], ps3[x], ps4[x]);
            buf_min[x] = vmin(ps0[x], ps1[x], ps2[x], ps3[x], ps4[x]);
        }

        for (x = 0; x < w; x++) {
            auto x0 = std::clamp(x - 2, 0, w - 1);
            auto x1 = std::clamp(x - 1, 0, w - 1);
            auto x2 = std::clamp(x + 0, 0, w - 1);
            auto x3 = std::clamp(x + 1, 0, w - 1);
            auto x4 = std::clamp(x + 2, 0, w - 1);

            auto max = vmax(buf_max[x0], buf_max[x1], buf_max[x2], buf_max[x3], buf_max[x4]);
            auto min = vmin(buf_min[x0], buf_min[x1], buf_min[x2], buf_min[x3], buf_min[x4]);
            pd[x] = max - min;
        }
    }

    return dst;
}
