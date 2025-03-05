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

    for (auto y = 2; y < h - 2; y++) {
        auto ps0 = src.ptr<uint8_t>(y - 2);
        auto ps1 = src.ptr<uint8_t>(y - 1);
        auto ps2 = src.ptr<uint8_t>(y + 0);
        auto ps3 = src.ptr<uint8_t>(y + 1);
        auto ps4 = src.ptr<uint8_t>(y + 2);

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

    auto ps0 = src.ptr<uint8_t>(0);
    auto ps1 = src.ptr<uint8_t>(1);
    auto ps2 = src.ptr<uint8_t>(2);
    auto ps3 = src.ptr<uint8_t>(3);
    auto pshm1 = src.ptr<uint8_t>(h - 1);
    auto pshm2 = src.ptr<uint8_t>(h - 2);
    auto pshm3 = src.ptr<uint8_t>(h - 3);
    auto pshm4 = src.ptr<uint8_t>(h - 4);

    auto pd0 = dst.ptr<uint8_t>(0);
    auto pd1 = dst.ptr<uint8_t>(1);
    auto pdhm1 = dst.ptr<uint8_t>(h - 1);
    auto pdhm2 = dst.ptr<uint8_t>(h - 2);

    for (auto x = 0; x < w; x++) {
        auto x0 = std::clamp(x - 2, 0, w - 1);
        auto x1 = std::clamp(x - 1, 0, w - 1);
        auto x2 = std::clamp(x + 0, 0, w - 1);
        auto x3 = std::clamp(x + 1, 0, w - 1);
        auto x4 = std::clamp(x + 2, 0, w - 1);

        pd0[x] = vmax(
            ps0[x0], ps0[x1], ps0[x2], ps0[x3], ps0[x4],
            ps1[x0], ps1[x1], ps1[x2], ps1[x3], ps1[x4],
            ps2[x0], ps2[x1], ps2[x2], ps2[x3], ps2[x4]
        ) - vmin(
            ps0[x0], ps0[x1], ps0[x2], ps0[x3], ps0[x4],
            ps1[x0], ps1[x1], ps1[x2], ps1[x3], ps1[x4],
            ps2[x0], ps2[x1], ps2[x2], ps2[x3], ps2[x4]
        );

        pd1[x] = vmax(
                ps0[x0], ps0[x1], ps0[x2], ps0[x3], ps0[x4],
                ps1[x0], ps1[x1], ps1[x2], ps1[x3], ps1[x4],
                ps2[x0], ps2[x1], ps2[x2], ps2[x3], ps2[x4],
                ps3[x0], ps3[x1], ps3[x2], ps3[x3], ps3[x4]
            ) - vmin(
                ps0[x0], ps0[x1], ps0[x2], ps0[x3], ps0[x4],
                ps1[x0], ps1[x1], ps1[x2], ps1[x3], ps1[x4],
                ps2[x0], ps2[x1], ps2[x2], ps2[x3], ps2[x4],
                ps3[x0], ps3[x1], ps3[x2], ps3[x3], ps3[x4]
            );

        pdhm1[x] = vmax(
            pshm1[x0], pshm1[x1], pshm1[x2], pshm1[x3], pshm1[x4],
            pshm2[x0], pshm2[x1], pshm2[x2], pshm2[x3], pshm2[x4],
            pshm3[x0], pshm3[x1], pshm3[x2], pshm3[x3], pshm3[x4]
        ) - vmin(
            pshm1[x0], pshm1[x1], pshm1[x2], pshm1[x3], pshm1[x4],
            pshm2[x0], pshm2[x1], pshm2[x2], pshm2[x3], pshm2[x4],
            pshm3[x0], pshm3[x1], pshm3[x2], pshm3[x3], pshm3[x4]
        );

        pdhm2[x] = vmax(
                pshm1[x0], pshm1[x1], pshm1[x2], pshm1[x3], pshm1[x4],
                pshm2[x0], pshm2[x1], pshm2[x2], pshm2[x3], pshm2[x4],
                pshm3[x0], pshm3[x1], pshm3[x2], pshm3[x3], pshm3[x4],
                pshm4[x0], pshm4[x1], pshm4[x2], pshm4[x3], pshm4[x4]
            ) - vmin(
                pshm1[x0], pshm1[x1], pshm1[x2], pshm1[x3], pshm1[x4],
                pshm2[x0], pshm2[x1], pshm2[x2], pshm2[x3], pshm2[x4],
                pshm3[x0], pshm3[x1], pshm3[x2], pshm3[x3], pshm3[x4],
                pshm4[x0], pshm4[x1], pshm4[x2], pshm4[x3], pshm4[x4]
            );
    }

    return dst;
}


cv::Mat maxMinFilterVectered(const cv::Mat& src) {
    int h = src.rows;
    int w = src.cols;

    cv::Mat dst(h, w, CV_8UC1);

    uint8_t buf_max[w];
    uint8_t buf_min[w];

    for (auto y = 2; y < h - 2; y++) {
        auto ps0 = src.ptr<uint8_t>(y - 2);
        auto ps1 = src.ptr<uint8_t>(y - 1);
        auto ps2 = src.ptr<uint8_t>(y + 0);
        auto ps3 = src.ptr<uint8_t>(y + 1);
        auto ps4 = src.ptr<uint8_t>(y + 2);

        auto pd = dst.ptr<uint8_t>(y);

        int x = 0;

        for (; x < w - 15; x += 16) {
            cv::v_uint8x16 u0 = cv::v_load(ps0 + x);
            cv::v_uint8x16 u1 = cv::v_load(ps1 + x);
            cv::v_uint8x16 u2 = cv::v_load(ps2 + x);
            cv::v_uint8x16 u3 = cv::v_load(ps3 + x);
            cv::v_uint8x16 u4 = cv::v_load(ps4 + x);

            cv::v_uint8x16 max = cv::v_max(cv::v_max(cv::v_max(cv::v_max(u0, u1), u2), u3), u4);
            cv::v_uint8x16 min = cv::v_min(cv::v_min(cv::v_min(cv::v_min(u0, u1), u2), u3), u4);

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

    auto ps0 = src.ptr<uint8_t>(0);
    auto ps1 = src.ptr<uint8_t>(1);
    auto ps2 = src.ptr<uint8_t>(2);
    auto ps3 = src.ptr<uint8_t>(3);
    auto pshm1 = src.ptr<uint8_t>(h - 1);
    auto pshm2 = src.ptr<uint8_t>(h - 2);
    auto pshm3 = src.ptr<uint8_t>(h - 3);
    auto pshm4 = src.ptr<uint8_t>(h - 4);

    auto pd0 = dst.ptr<uint8_t>(0);
    auto pd1 = dst.ptr<uint8_t>(1);
    auto pdhm1 = dst.ptr<uint8_t>(h - 1);
    auto pdhm2 = dst.ptr<uint8_t>(h - 2);

    for (auto x = 0; x < w; x++) {
        auto x0 = std::clamp(x - 2, 0, w - 1);
        auto x1 = std::clamp(x - 1, 0, w - 1);
        auto x2 = std::clamp(x + 0, 0, w - 1);
        auto x3 = std::clamp(x + 1, 0, w - 1);
        auto x4 = std::clamp(x + 2, 0, w - 1);

        pd0[x] = vmax(
            ps0[x0], ps0[x1], ps0[x2], ps0[x3], ps0[x4],
            ps1[x0], ps1[x1], ps1[x2], ps1[x3], ps1[x4],
            ps2[x0], ps2[x1], ps2[x2], ps2[x3], ps2[x4]
        ) - vmin(
            ps0[x0], ps0[x1], ps0[x2], ps0[x3], ps0[x4],
            ps1[x0], ps1[x1], ps1[x2], ps1[x3], ps1[x4],
            ps2[x0], ps2[x1], ps2[x2], ps2[x3], ps2[x4]
        );

        pd1[x] = vmax(
                ps0[x0], ps0[x1], ps0[x2], ps0[x3], ps0[x4],
                ps1[x0], ps1[x1], ps1[x2], ps1[x3], ps1[x4],
                ps2[x0], ps2[x1], ps2[x2], ps2[x3], ps2[x4],
                ps3[x0], ps3[x1], ps3[x2], ps3[x3], ps3[x4]
            ) - vmin(
                ps0[x0], ps0[x1], ps0[x2], ps0[x3], ps0[x4],
                ps1[x0], ps1[x1], ps1[x2], ps1[x3], ps1[x4],
                ps2[x0], ps2[x1], ps2[x2], ps2[x3], ps2[x4],
                ps3[x0], ps3[x1], ps3[x2], ps3[x3], ps3[x4]
            );

        pdhm1[x] = vmax(
            pshm1[x0], pshm1[x1], pshm1[x2], pshm1[x3], pshm1[x4],
            pshm2[x0], pshm2[x1], pshm2[x2], pshm2[x3], pshm2[x4],
            pshm3[x0], pshm3[x1], pshm3[x2], pshm3[x3], pshm3[x4]
        ) - vmin(
            pshm1[x0], pshm1[x1], pshm1[x2], pshm1[x3], pshm1[x4],
            pshm2[x0], pshm2[x1], pshm2[x2], pshm2[x3], pshm2[x4],
            pshm3[x0], pshm3[x1], pshm3[x2], pshm3[x3], pshm3[x4]
        );

        pdhm2[x] = vmax(
                pshm1[x0], pshm1[x1], pshm1[x2], pshm1[x3], pshm1[x4],
                pshm2[x0], pshm2[x1], pshm2[x2], pshm2[x3], pshm2[x4],
                pshm3[x0], pshm3[x1], pshm3[x2], pshm3[x3], pshm3[x4],
                pshm4[x0], pshm4[x1], pshm4[x2], pshm4[x3], pshm4[x4]
            ) - vmin(
                pshm1[x0], pshm1[x1], pshm1[x2], pshm1[x3], pshm1[x4],
                pshm2[x0], pshm2[x1], pshm2[x2], pshm2[x3], pshm2[x4],
                pshm3[x0], pshm3[x1], pshm3[x2], pshm3[x3], pshm3[x4],
                pshm4[x0], pshm4[x1], pshm4[x2], pshm4[x3], pshm4[x4]
            );
    }

    return dst;
}


int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "wrong usage" << std::endl;
        return 1;
    }
    
    cv::Mat src = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);

    auto t1 = std::chrono::system_clock::now();
    cv::Mat res = maxMinFilter(src);
    auto t2 = std::chrono::system_clock::now();
    cv::Mat res_vec = maxMinFilterVectered(src);
    auto t3 = std::chrono::system_clock::now();
    
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));

    cv::Mat max, min;
    cv::dilate(src, max, kernel);
    cv::erode(src, min, kernel);

    cv::Mat res_cv = max - min;
    auto t4 = std::chrono::system_clock::now();

    auto dt1 = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
    auto dt2 = std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count();
    auto dt3 = std::chrono::duration_cast<std::chrono::nanoseconds>(t4 - t3).count();

    std::cout << "basic: " << dt1 << " ms, vectorized: " << dt2 << "ms, opencv: " << dt3 * 1e-6 << " ms" << std::endl;

    bool equal = true;
    for (auto y = 0; y < src.rows; y++) {
        for (auto x = 0; x < src.cols; x++) {
            auto& res1 = res_vec.at<uint8_t>(y, x);
            auto& res2 = res_cv.at<uint8_t>(y, x);
            if (res1 != res2) {
                equal = false;
                std::cout << y << " " << x << " " << int(res1) << " " << int(res2) << std::endl;
            }
        }
    }
    std::cout << "is equal: " << (equal ? "true" : "false") << std::endl;

    cv::imshow("src", src);
    cv::imshow("cv", res_cv);
    cv::imshow("my", res);
    cv::imshow("my_vec", res_vec);
    cv::imshow("diff", cv::abs(res_vec - res));
    cv::imshow("diff_cv", cv::abs(res_cv - res_vec));
    cv::waitKey(0);

    return 0;
}
