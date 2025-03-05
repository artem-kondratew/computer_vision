#ifndef LAB4_FOURIER_HPP
#define LAB4_FOURIER_HPP


#include <iostream>

#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/imgproc.hpp>
#include <opencv4/opencv2/highgui.hpp>


namespace Fourier {

cv::Mat getOptimalDftSize(cv::Mat src, int type) {
	int w = cv::getOptimalDFTSize(src.cols);
	int h = cv::getOptimalDFTSize(src.rows);
	cv::Mat result = cv::Mat(cv::Size(w, h), type, cv::Scalar(0));

	cv::Rect rectangle(0, 0, src.cols, src.rows);
	src.copyTo(result(rectangle));
    return result;
}


cv::Mat dft(const cv::Mat src) {
    size_t w = src.cols;
    size_t h = src.rows;

    cv::Mat result(w, h, CV_32FC2);

    // std::cout << "start" << std::endl;
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            for (int i = 0; i < h; i++) {
                for (int j = 0; j < w; j++) {
                    float arg = 2 * M_PI * (static_cast<float>(y * i) / h + static_cast<float>(x * j) / w);
                    result.at<cv::Vec2f>(y, x)[0] += src.at<cv::Vec<float, 1>>(i, j)[0] * std::cos(arg);
                    result.at<cv::Vec2f>(y, x)[1] += src.at<cv::Vec<float, 1>>(i, j)[0] * std::sin(arg);
                }
            }
        }
    }
    // std::cout << "end" << std::endl;
    return result;
}


cv::Mat idft(cv::Mat src) {
    size_t w = src.cols;
    size_t h = src.rows;

    cv::Mat result(w, h, CV_32FC1);

    // std::cout << "start" << std::endl;
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            for (int i = 0; i < h; i++) {
                for (int j = 0; j < w; j++) {
                    float arg = 2 * M_PI * (static_cast<float>(y * i) / h + static_cast<float>(x * j) / w);
                    result.at<float>(y, x) += src.at<cv::Vec2f>(i, j)[0] * std::cos(arg) -
                    src.at<cv::Vec2f>(i, j)[1] * std::sin(arg);
                }
            }
            result.at<float>(y, x) /= static_cast<float>(w * h);
        }
    }
    // std::cout << "end" << std::endl;
    return result;
}


void krasivSpektr(cv::Mat& magI) {
	// rearrange the quadrants of Fourier image  so that the origin is at the image center
	int cx = magI.cols / 2;
	int cy = magI.rows / 2;

	cv::Mat q0(magI, cv::Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant
	cv::Mat q1(magI, cv::Rect(cx, 0, cx, cy));  // Top-Right
	cv::Mat q2(magI, cv::Rect(0, cy, cx, cy));  // Bottom-Left
	cv::Mat q3(magI, cv::Rect(cx, cy, cx, cy)); // Bottom-Right

	cv::Mat tmp;                           // swap quadrants (Top-Left with Bottom-Right)
	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);

	q1.copyTo(tmp);                    // swap quadrant (Top-Right with Bottom-Left)
	q2.copyTo(q1);
	tmp.copyTo(q2);
}


void spectrum(cv::Mat& input, cv::Mat& output, bool krasiv=true) {
	std::vector<cv::Mat> temp;
	cv::split(input, temp);

	magnitude(temp[0], temp[1], output);
	if(krasiv) {
        krasivSpektr(output);
    }

	output += cv::Scalar::all(1);
	log(output, output);

	normalize(output, output, 0.0f, 1.0f, cv::NormTypes::NORM_MINMAX);
}


void show(std::string window, cv::Mat dft, bool krasiv=true, bool colored=true) {
	cv::Mat show;
	spectrum(dft, show, krasiv);
	show.convertTo(show, CV_8UC1, 255);
	if (colored) {
        cv::applyColorMap(show, show, cv::COLORMAP_JET);
    }
	imshow(window, show);
}


bool isPowerOfTwo(size_t n) {
	if (n <= 0)
	{
		return false;
	}

	return (n & (n - 1)) == 0;
}


std::vector<std::complex<double>> mat2vec(cv::Mat image) {
	std::vector<std::complex<double>> result;

	for (int i = 0; i < image.rows; i++) {
		for (int j = 0; j < image.cols; j++) {
			float real = image.at<float>(i, j);
			result.push_back(std::complex<double>(real, 0.0));
		}
	}

	return result;
}


cv::Mat vec2mat(std::vector<std::complex<double>> data, int rows, int cols, int format) {
	cv::Mat result(rows, cols, CV_32FC2);

	if (data.size() != rows * cols) {
		std::cerr << "Input vector size does not match the specified rows and cols" << std::endl;
		return result;
	}

	int k = 0;
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			result.at<cv::Vec2f>(i, j)[1] = data[k].imag();
			result.at<cv::Vec2f>(i, j)[0] = data[k].real();
			k++;
		}
	}

	if (format == CV_32FC1) {
		cv::Mat realPart(rows, cols, CV_32FC1);
		cv::extractChannel(result, realPart, 0);
		return realPart;
	}

    cv::rotate(result, result, cv::ROTATE_90_COUNTERCLOCKWISE);
    cv::rotate(result, result, cv::ROTATE_180);
	return result;
}


void radixTransform(std::vector<std::complex<double>>& input, bool invert)
{
	int n = input.size();
	if (n <= 1)
		return;
	if (!isPowerOfTwo(n))
		return;

	std::vector<std::complex<double>> a0(n / 2), a1(n / 2);
	for (int i = 0, j = 0; i < n; i += 2, ++j)
	{
		a0[j] = input[i];
		a1[j] = input[i + 1];
	}

	radixTransform(a0, invert);
	radixTransform(a1, invert);

	double angle = 2 * CV_PI / n * (invert ? -1 : 1);
	std::complex<double> w(1), wn(cos(angle), sin(angle));

	for (int i = 0; i < n / 2; ++i)
	{
		std::complex<double> t = w * a1[i];
		input[i] = a0[i] + t;
		input[i + n / 2] = a0[i] - t;
		if (invert)
		{
			input[i] /= 2;
			input[i + n / 2] /= 2;
		}
		w *= wn;
	}
}


cv::Mat convolution(cv::Mat src, cv::Mat kernel) {
    cv::Mat dft_img;
    cv::Mat result;

	cv::Rect rect(0, 0, kernel.cols, kernel.rows);
	cv::Mat dft_kernel(cv::Size(src.cols, src.rows), CV_32FC1, cv::Scalar(0));
	kernel.copyTo(dft_kernel(rect));

	cv::dft(src, dft_img, cv::DFT_COMPLEX_OUTPUT);
	cv::dft(dft_kernel, dft_kernel, cv::DFT_COMPLEX_OUTPUT);

	cv::mulSpectrums(dft_img, dft_kernel, result, 0);
	cv::dft(result, result, cv::DFT_INVERSE | cv::DFT_REAL_OUTPUT);

    return result;
}


void createKernel(cv::Mat input, cv::Mat& output, int width, int height) {
	double minVal, maxVal;
	cv::minMaxLoc(input, &minVal, &maxVal);
	input = (input - minVal) / (maxVal - minVal);
	input.convertTo(output, CV_8UC1, 255);
	cv::resize(output, output, cv::Size(width, height));
}


void cutLowSpec(cv::Mat input, cv::Mat& output, int radius) {
	krasivSpektr(input);
	cv::Mat temp(cv::Size(input.cols, input.rows), CV_32FC2, cv::Scalar(1));
	int xc = input.cols / 2;
	int yc = input.rows / 2;

	cv::circle(temp, cv::Point(xc, yc), radius, cv::Scalar(0), -1);

	cv::mulSpectrums(input, temp, output, 0);
}

void cutHighSpec(cv::Mat input, cv::Mat& output, int radius) {

	cv::Mat temp(cv::Size(input.cols, input.rows), CV_32FC2, cv::Scalar(0));
	int xc = input.cols / 2;
	int yc = input.rows / 2;

	cv::circle(temp, cv::Point(xc, yc), radius, cv::Scalar(1), -1);

	cv::mulSpectrums(input, temp, output, 0);

}

}


#endif // LAB4_FOURIER_HPP