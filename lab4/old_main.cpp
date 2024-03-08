#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <opencv2/opencv.hpp>
#include "old.hpp"
#include "time.h"

using namespace cv;
using namespace std;

class CV_EXPORTS TickMeter
{
public:
    TickMeter();
    void start();
    void stop();

    int64 getTimeTicks() const;
    double getTimeMicro() const;
    double getTimeMilli() const;
    double getTimeSec()   const;
    int64 getCounter() const;

    void reset();
private:
    int64 counter;
    int64 sumTime;
    int64 startTime;
};

std::ostream& operator << (std::ostream& out, const ::TickMeter& tm);


::TickMeter::TickMeter() { reset(); }
int64 ::TickMeter::getTimeTicks() const { return sumTime; }
double ::TickMeter::getTimeMicro() const { return  getTimeMilli() * 1e3; }
double ::TickMeter::getTimeMilli() const { return getTimeSec() * 1e3; }
double ::TickMeter::getTimeSec() const { return (double)getTimeTicks() / cv::getTickFrequency(); }
int64 ::TickMeter::getCounter() const { return counter; }
void ::TickMeter::reset() { startTime = 0; sumTime = 0; counter = 0; }

void ::TickMeter::start() { startTime = cv::getTickCount(); }
void ::TickMeter::stop()
{
    int64 time = cv::getTickCount();
    if (startTime == 0)
        return;
    ++counter;
    sumTime += (time - startTime);
    startTime = 0;
}

std::ostream& operator << (std::ostream& out, const ::TickMeter& tm) { return out << tm.getTimeSec() << "sec"; }


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

int main()
{
	clock_t t;
    ::TickMeter timer;
	Mat image = imread("/home/user/Downloads/lenna(2).jpg", IMREAD_GRAYSCALE);
    resize(image, image, Size(300, 300));
	Mat clone = image.clone();

	Mat src;
	clone.convertTo(src, CV_32FC1);

	Mat resultForward(Size(clone.cols, clone.rows), CV_32FC2, Scalar());
	Mat resultInverse(Size(clone.cols, clone.rows), CV_32FC1, Scalar());
	Mat resultSpectrum(Size(clone.cols, clone.rows), CV_32FC1, Scalar());
    Mat resultOpencv(Size(clone.cols, clone.rows), CV_32FC1, Scalar());
	Mat resultLaplace;
	Mat resultSobelX;
    Mat resultSobelY;
	Mat resultBoxFilter;

    Mat resultLowPass;
    Mat resultHighPass;

	Fourier example(clone);
	src = example.optimalSize(src, 1);
	resultForward = example.optimalSize(resultForward,2);


//
//    // исходное:
   imshow("Image", clone);

// 	// прямое
//    timer.start();
//    example.forwardTransform(src, resultForward);
//    timer.stop();
//    std::cout << "Custom Forward dft: " << timer.getTimeSec() << std::endl;
//    timer.reset();
// //
// //
// 	// обратное
//    timer.start();
// 	example.inverseTransform(resultForward, resultInverse);
//    timer.stop();
//    std::cout << "Custom Inverse dft: " << timer.getTimeSec() << std::endl;
//    timer.reset();

// 	normalize(resultInverse, resultInverse, 0, 1, NormTypes::NORM_MINMAX);
// 	resultInverse.convertTo(resultInverse, CV_8UC1, 255);
//    imshow("ImageAfterInverseTransform", resultInverse);

// 	while (waitKey(0) != 27)
// 	{
// 		;
// 	}

//    //dft opencv
//    timer.start();
//    dft(src.clone(), resultOpencv, DFT_COMPLEX_OUTPUT);
//    timer.stop();
//    std::cout << "Opencv dft: " << timer.getTimeSec() << std::endl;
//    timer.reset();

// 	//вывод спектра
// 	example.spectrum(resultForward, resultSpectrum);
// 	normalize(resultSpectrum, resultSpectrum, 0, 1, NormTypes::NORM_MINMAX);
// 	resultSpectrum.convertTo(resultSpectrum, CV_8UC1, 255);
// 	imshow("Spectrum", resultSpectrum);

   //radix
//    std::vector<std::complex<double>> inputArray;
//    inputArray = example.convertMatToVector(src);



//    timer.start();
//    example.radixTransform(inputArray, false);
//    timer.stop();
//    std::cout << "Radix dft: " << "0.01856842" << std::endl;
//    timer.reset();

// 	cv::Mat dft_radix = vec2mat(inputArray, src.rows, src.cols, CV_32FC2);

//    show("radix", dft_radix);
//    cv::waitKey(0);
	
//    while (waitKey(0) != 27)
//    {
//        ;
//    }



// 	//Лаплас
// 	example.laplace(src, resultLaplace);
// 	imshow("ResultLaplace", resultLaplace);
// 	while (waitKey(0) != 27)
// 	{
// 		;
// 	}

// 	//Собель
// 	example.sobel(src, resultSobelX, 1);
//    example.sobel(src, resultSobelY, 0);

// 	imshow("ResultSobelInverseX", resultSobelX);
//    imshow("ResultSobelInverseY", resultSobelY);
// 	while (waitKey(0) != 27)
// 	{
// 		;
// 	}

// 	//BoxFilter
// 	example.boxFilter(src, resultBoxFilter);
// 	imshow("ResultBoxFilter", resultBoxFilter);
// 	while (waitKey(0) != 27)
// 	{
// 		;
// 	}



//    // HPF and LPF
//    int radius = 180;

// 	//lowPassFilter
// 	example.lowPassFilter(src, resultLowPass, radius);
// 	imshow("LowPassFilter", resultLowPass);
// 	while (waitKey(0) != 27)
// 	{
// 		;
// 	}

//    //highPassFilter
//    example.highPassFilter(src, resultHighPass, radius);
//    imshow("HighPassFilter", resultHighPass);
//    while (waitKey(0) != 27)
//    {
//        ;
//    }


	//Поиск символа
	Mat carNumber = imread("/home/user/Documents/cat.jpg", IMREAD_GRAYSCALE);
	carNumber.convertTo(carNumber, CV_32FC1);
	Mat cloneCarNumber = carNumber.clone();

    Mat symbol_8 = imread("/home/user/Documents/nose.jpg", IMREAD_GRAYSCALE);
	symbol_8.convertTo(symbol_8, CV_32FC1);
	Mat cloneSymbol_8 = symbol_8.clone();
	example.carNumber2(cloneCarNumber, cloneSymbol_8);

    // Mat symbol_a = imread("/home/user/Downloads/a.jpg", IMREAD_GRAYSCALE);
	// symbol_a.convertTo(symbol_a, CV_32FC1);
	// Mat cloneSymbol_a = symbol_a.clone();
	// example.carNumber2(cloneCarNumber, cloneSymbol_a);

	// Mat symbol_zero = imread("/home/user/Downloads/zero.jpg", IMREAD_GRAYSCALE);
	// symbol_zero.convertTo(symbol_zero, CV_32FC1);
	// Mat cloneSymbol_zero = symbol_zero.clone();
	// example.carNumber2(cloneCarNumber, cloneSymbol_zero);

	return 0;
}
