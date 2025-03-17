#include <iostream>
#include <limits>

#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/imgproc.hpp>
#include <opencv4/opencv2/highgui.hpp>


bool checkMarker(const std::vector<cv::Point>& points) {
    cv::Point mean{0, 0};
    for (auto pt : points) {
        mean.x += pt.x;
        mean.y += pt.y;
    }
    mean.x /= points.size();
    mean.y /= points.size();
    
    int idx = -1;
    float min_dist = std::numeric_limits<float>::infinity();
    for (auto i = 0; i < points.size(); i++) {
        float dist = cv::norm(mean - points[i]);
        if (dist < min_dist) {
            min_dist = dist;
            idx = i;
        }
    }

    if (idx == -1) {
        return false;
    }

    cv::Mat lengths(points.size(), 1, CV_64FC1);
    for (auto i = 0; i < lengths.rows; i++) {
        int j = (i == 0) ? lengths.rows - 1 : i - 1;
        lengths.at<double>(i, 1) = cv::norm(points[0] - points[j]);
    }

    double L = cv::sum(lengths)[0];

    int err = 0.1 * L;

    const cv::Point& center = points[idx];

    int sz = points.size();
    const cv::Point& d1 = points[(idx - 2 + sz) % sz];
    const cv::Point& d2 = points[(idx - 1 + sz) % sz];
    const cv::Point& d3 = points[(idx + 1 + sz) % sz];
    const cv::Point& d4 = points[(idx + 2 + sz) % sz];
    const cv::Point& d5 = points[(idx + 3 + sz) % sz];

    cv::Point line_center = (d1 + d4) / 2;

    if (cv::norm(line_center - center) > err / 2) {
        return false;
    }

    if (cv::abs(cv::norm(d1 - d2) - cv::norm(center - d3)) > err) {
        return false;
    }

    if (cv::abs(cv::norm(d3 - d4) - cv::norm(center - d2)) > err) {
        return false;
    }

    if (cv::abs(cv::norm(d4 - d5) - 2 * cv::norm(d1 - d2)) > err) {
        return false;
    }

    if (cv::abs(cv::norm(d4 - d5) - 2 * cv::norm(center - d3)) > err) {
        return false;
    }

    if (cv::abs(cv::norm(d1 - d5) - 2 * cv::norm(d3 - d4)) > err) {
        return false;
    }

    if (cv::abs(cv::norm(d1 - d5) - 2 * cv::norm(d3 - d4)) > err) {
        return false;
    }
        
    return true;
}


int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "wrong usage. correct usage: ./marker_detector <input_path> <output_path>" << std::endl;
        exit(1);
    }

    cv::Mat image = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);

    cv::Mat blurred;
    cv::blur(image, blurred, cv::Size(3, 3));

    cv::Mat thresholded;
    int blocksize = int(std::min(image.rows, image.cols) * 0.8);
    blocksize = (blocksize % 2 == 0) ? blocksize - 1 : blocksize;
    blocksize = std::max(blocksize, 3);
    cv::adaptiveThreshold(blurred, thresholded, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, blocksize, 20);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(thresholded, contours, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);

    cv::cvtColor(image, image, cv::COLOR_GRAY2BGR);

    for (auto i = 0; i < contours.size(); i++ ) {
        std::vector<cv::Point> contour;
        cv::approxPolyDP(contours[i], contour, 3, true);
        if (contour.size() != 6) {
            continue;
        }
        if (!checkMarker(contour)) {
            continue;
        }
        cv::Scalar color = cv::Scalar(255, 0, 0);
        cv::drawContours(image, std::vector<std::vector<cv::Point>>{contour}, -1, color, 2);
        for (auto pt : contour) {
            cv::circle(image, pt, 3, {255, 0, 255}, -1);
        }
    }

    cv::imwrite(argv[2], image);

    return 0;
}
