#include <iostream>

#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/imgproc.hpp>
#include <opencv4/opencv2/highgui.hpp>


cv::Mat getDftOptimalSize(cv::Mat& image, int channels) {
	cv::Size dftSize;
	dftSize.width = cv::getOptimalDFTSize(image.cols);
	dftSize.height = cv::getOptimalDFTSize(image.rows);
	cv::Rect rectangle(0, 0, image.cols, image.rows);

	if (channels == 1) {
		cv::Mat dftImg(dftSize, CV_32FC1, cv::Scalar(0));
		image.copyTo(dftImg(rectangle));
		return dftImg;
	}

	if (channels == 2) {
		cv::Mat dftImg(dftSize, CV_32FC2, cv::Scalar(0));
		image.copyTo(dftImg(rectangle));
		return dftImg;
	}

    return cv::Mat();
}


void findNose(cv::Mat& cat, cv::Mat& nose) {
	int num1 = 0;

	for (int i = 0; i < cat.rows; i++) {
		for (int j = 0; j < cat.cols; j++) {
			num1 += cat.at<float>(i, j);
		}
	}
	num1 /= (cat.rows * cat.cols);
	cv::Mat img1 = cat - num1;

	int num2 = 0;
	for (int i = 0; i < nose.rows; i++) {
		for (int j = 0; j < nose.cols; j++) {
			num2 += nose.at<float>(i, j);
		}
	}
	num2 /= (nose.rows * nose.cols);
	cv::Mat img2 = nose - num2;

	cat = img1.clone();
	nose = img2.clone();

	cv::Size originalSize(cat.cols, cat.rows);
	cat = getDftOptimalSize(cat, 1);

	cv::Mat cat_to_show = cat.clone();
	cv::normalize(cat_to_show, cat_to_show, 0, 1, cv::NormTypes::NORM_MINMAX);
	cat_to_show.convertTo(cat_to_show, CV_8UC1, 255);
	cv::imwrite("Cat_OptimalSize.jpg", cat_to_show);

	cv::Mat expandSymbol(cv::Size(cat.cols, cat.rows), CV_32FC1, cv::Scalar());
	cv::Rect rect(0, 0, nose.cols, nose.rows);
	cv::Mat symbolOnExpandImage(cv::Size(cat.cols, cat.rows), CV_32FC1, cv::Scalar(0));
	nose.copyTo(symbolOnExpandImage(rect));

	cv::Mat nose_to_show = symbolOnExpandImage.clone();
	cv::normalize(nose_to_show, nose_to_show, 0, 1, cv::NormTypes::NORM_MINMAX);
	nose_to_show.convertTo(nose_to_show, CV_8UC1, 255);
	cv::imwrite("Nose_OptimalSize.jpg", nose_to_show);

	cv::Mat cat_dft(cv::Size(cat.cols, cat.rows), CV_32FC2, cv::Scalar());
	cv::dft(cat, cat_dft, cv::DFT_COMPLEX_OUTPUT);

	cv::Mat nose_dft(cv::Size(cat.cols, cat.rows), CV_32FC2, cv::Scalar());
	cv::dft(symbolOnExpandImage, nose_dft, cv::DFT_COMPLEX_OUTPUT);

	cv::Mat Spectrum(cv::Size(cat.cols, cat.rows), CV_32FC2, cv::Scalar());
	cv::mulSpectrums(cat_dft, nose_dft, Spectrum, 0, 1);

	cv::Mat resultInverse(cv::Size(cat.cols, cat.rows), CV_32FC1, cv::Scalar());
	cv::dft(Spectrum, resultInverse, cv::DFT_INVERSE | cv::DFT_REAL_OUTPUT);

	cv::Rect rectangle1(0, 0, originalSize.width, originalSize.height);
	cv::Mat backSize(originalSize, CV_32FC1, cv::Scalar(0));
	resultInverse(rectangle1).copyTo(backSize);

	cv::normalize(resultInverse, resultInverse, 0, 1, cv::NormTypes::NORM_MINMAX);
	resultInverse.convertTo(resultInverse, CV_8UC1, 255);
	cv::imwrite("Result_after_mul.jpg", resultInverse);

	cv::normalize(backSize, backSize, 0, 1, cv::NormTypes::NORM_MINMAX);
	double maxValue = 0;
	cv::minMaxLoc(backSize, NULL, &maxValue);
	double thresh = maxValue - 0.014;
	cv::threshold(backSize, backSize, thresh, 0, cv::THRESH_TOZERO);

	cv::Point2i pt;
	for (int i = 0; i < cat.rows; i++) {
		for (int j = 0; j < cat.cols; j++) {
			if (backSize.at<float>(i, j) != 0) {
				pt.x = j;
				pt.y = i;
			}
		}
	}
	
	cv::Mat rect_on_nose_to_show = cat.clone();
	cv::Rect res(pt.x, pt.y, nose.cols, nose.rows);
	rectangle(rect_on_nose_to_show, res, cv::Scalar(255), 1, 8, 0);
	normalize(rect_on_nose_to_show, rect_on_nose_to_show, 0, 1, cv::NormTypes::NORM_MINMAX);
	rect_on_nose_to_show.convertTo(rect_on_nose_to_show, CV_8UC1, 255);
	imwrite("Rect.jpg", rect_on_nose_to_show);

	cv::Mat res_to_show = backSize.clone();
	cv::normalize(res_to_show, res_to_show, 0, 1, cv::NormTypes::NORM_MINMAX);
	res_to_show.convertTo(res_to_show, CV_8UC1, 255);
	imwrite("Result.jpg", res_to_show);
}


int main() {
    cv::Mat cat = cv::imread("/home/user/Documents/cat.jpg", cv::IMREAD_GRAYSCALE);
    cat.convertTo(cat, CV_32FC1);
    cv::Mat cat_copy = cat.clone();

    cv::Mat nose = cv::imread("/home/user/Documents/fragment.jpg", cv::IMREAD_GRAYSCALE);
    cv::imwrite("nose.jpg", nose);
    nose.convertTo(nose, CV_32FC1);
    cv::Mat nose_copy = nose_copy.clone();
    
    findNose(cat, nose);
    
    cv::waitKey(0);
    return 0;
}
