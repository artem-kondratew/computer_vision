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
    size_t h = src.rows;
    size_t w = src.cols;

    cv::Mat dst(h, w, CV_8UC1);

    cv::Mat zeros(h + 4, w + 4, CV_8UC1, cv::Scalar(0));
    cv::Mat ones(h + 4, w + 4, CV_8UC1, cv::Scalar(255));

    src.copyTo(zeros(cv::Rect(2, 2, w, h)));
    src.copyTo(ones(cv::Rect(2, 2, w, h)));

    w += 4;
    h += 4;

    for (auto y = 2; y < h - 2; y++) {
        auto zeros_ptr0 = zeros.ptr<uint8_t>(y - 2);
        auto zeros_ptr1 = zeros.ptr<uint8_t>(y - 1);
        auto zeros_ptr2 = zeros.ptr<uint8_t>(y);
        auto zeros_ptr3 = zeros.ptr<uint8_t>(y + 1);
        auto zeros_ptr4 = zeros.ptr<uint8_t>(y + 2);

        auto ones_ptr0 = ones.ptr<uint8_t>(y - 2);
        auto ones_ptr1 = ones.ptr<uint8_t>(y - 1);
        auto ones_ptr2 = ones.ptr<uint8_t>(y);
        auto ones_ptr3 = ones.ptr<uint8_t>(y + 1);
        auto ones_ptr4 = ones.ptr<uint8_t>(y + 2);

        auto dst_ptr = dst.ptr<uint8_t>(y - 2);

        for (auto x = 2; x < w - 2; x++) {
            dst_ptr[x-2] = vmax(
                zeros_ptr0[x-2], zeros_ptr0[x-1], zeros_ptr0[x], zeros_ptr0[x+1], zeros_ptr0[x+2],
                zeros_ptr1[x-2], zeros_ptr1[x-1], zeros_ptr1[x], zeros_ptr1[x+1], zeros_ptr1[x+2],
                zeros_ptr2[x-2], zeros_ptr2[x-1], zeros_ptr2[x], zeros_ptr2[x+1], zeros_ptr2[x+2],
                zeros_ptr3[x-2], zeros_ptr3[x-1], zeros_ptr3[x], zeros_ptr3[x+1], zeros_ptr3[x+2],
                zeros_ptr4[x-2], zeros_ptr4[x-1], zeros_ptr4[x], zeros_ptr4[x+1], zeros_ptr4[x+2]
            ) - vmin(
                ones_ptr0[x-2], ones_ptr0[x-1], ones_ptr0[x], ones_ptr0[x+1], ones_ptr0[x+2],
                ones_ptr1[x-2], ones_ptr1[x-1], ones_ptr1[x], ones_ptr1[x+1], ones_ptr1[x+2],
                ones_ptr2[x-2], ones_ptr2[x-1], ones_ptr2[x], ones_ptr2[x+1], ones_ptr2[x+2],
                ones_ptr3[x-2], ones_ptr3[x-1], ones_ptr3[x], ones_ptr3[x+1], ones_ptr3[x+2],
                ones_ptr4[x-2], ones_ptr4[x-1], ones_ptr4[x], ones_ptr4[x+1], ones_ptr4[x+2]
            );
        }
    }

    return dst;
}


cv::Mat maxMinFilterVectered(const cv::Mat& src) {
    size_t h = src.rows;
    size_t w = src.cols;

    cv::Mat dst(h, w, CV_8UC1);

    cv::Mat zeros(h + 4, w + 4, CV_8UC1, cv::Scalar(0));
    cv::Mat ones(h + 4, w + 4, CV_8UC1, cv::Scalar(255));

    src.copyTo(zeros(cv::Rect(2, 2, w, h)));
    src.copyTo(ones(cv::Rect(2, 2, w, h)));

    w += 4;
    h += 4;

    uint8_t buf_max[w];
    uint8_t buf_min[w];

    for (auto y = 2; y < h - 2; y++) {
        auto zeros_ptr0 = zeros.ptr<uint8_t>(y - 2);
        auto zeros_ptr1 = zeros.ptr<uint8_t>(y - 1);
        auto zeros_ptr2 = zeros.ptr<uint8_t>(y);
        auto zeros_ptr3 = zeros.ptr<uint8_t>(y + 1);
        auto zeros_ptr4 = zeros.ptr<uint8_t>(y + 2);

        auto ones_ptr0 = ones.ptr<uint8_t>(y - 2);
        auto ones_ptr1 = ones.ptr<uint8_t>(y - 1);
        auto ones_ptr2 = ones.ptr<uint8_t>(y);
        auto ones_ptr3 = ones.ptr<uint8_t>(y + 1);
        auto ones_ptr4 = ones.ptr<uint8_t>(y + 2);

        auto dst_ptr = dst.ptr<uint8_t>(y - 2);

        size_t x = 0;

        for (; x < w - 15; x += 16) {
            cv::v_uint8x16 zeros0 = cv::v_load(zeros_ptr0 + x);
            cv::v_uint8x16 zeros1 = cv::v_load(zeros_ptr1 + x);
            cv::v_uint8x16 zeros2 = cv::v_load(zeros_ptr2 + x);
            cv::v_uint8x16 zeros3 = cv::v_load(zeros_ptr3 + x);
            cv::v_uint8x16 zeros4 = cv::v_load(zeros_ptr4 + x);

            cv::v_uint8x16 ones0 = cv::v_load(ones_ptr0 + x);
            cv::v_uint8x16 ones1 = cv::v_load(ones_ptr1 + x);
            cv::v_uint8x16 ones2 = cv::v_load(ones_ptr2 + x);
            cv::v_uint8x16 ones3 = cv::v_load(ones_ptr3 + x);
            cv::v_uint8x16 ones4 = cv::v_load(ones_ptr4 + x);

            cv::v_uint8x16 max = cv::v_max(cv::v_max(cv::v_max(cv::v_max(zeros0, zeros1), zeros2), zeros3), zeros4);
            cv::v_uint8x16 min = cv::v_min(cv::v_min(cv::v_min(cv::v_min(ones0, ones1), ones2), ones3), ones4);

            cv::v_store(buf_max + x, max);
            cv::v_store(buf_min + x, min);
        }

        for (; x < w; x++) {
            buf_max[x] = vmax(zeros_ptr0[x], zeros_ptr1[x], zeros_ptr2[x], zeros_ptr3[x], zeros_ptr4[x]);
            buf_min[x] = vmin(ones_ptr0[x], ones_ptr1[x], ones_ptr2[x], ones_ptr3[x], ones_ptr4[x]);
        }

        for (x = 2; x < w - 2; x++) {
            auto max = vmax(buf_max[x-2], buf_max[x-1], buf_max[x], buf_max[x+1], buf_max[x+2]);
            auto min = vmin(buf_min[x-2], buf_min[x-1], buf_min[x], buf_min[x+1], buf_min[x+2]);
            dst_ptr[x-2] = max - min;                   
        }
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

    cv::imshow("src", src);
    cv::imshow("cv", res_cv);
    cv::imshow("my", res);
    cv::imshow("my_vec", res_vec);
    cv::imshow("diff", cv::abs(res_vec - res));
    cv::imshow("diff_cv", cv::abs(res_cv - res));
    cv::waitKey(0);

    return 0;
}
