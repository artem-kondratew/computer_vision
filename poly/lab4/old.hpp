#pragma once
#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp";

using namespace cv;
using namespace std;

class Fourier
{
public:
    Fourier() = default;
    Fourier(Mat m_inputImage);
    ~Fourier() = default;

    void forwardTransform(Mat& inputImage, Mat& outputImage);
    void inverseTransform(Mat& inputImage, Mat& outputImage);
    std::vector<std::complex<double>> convertMatToVector(cv::Mat image);
    void radixTransform(std::vector<std::complex<double>> &inputArray, bool invert);
    Mat optimalSize(Mat& image, int channels);
    void swapSpektr(Mat& magI);
    void spectrum(Mat& imageAfterDFT, Mat& result);
    void laplace(Mat& inputImage, Mat& outputImage);
    void sobel(Mat& inputImage, Mat& outputImage, int flag); //horizontal  flag = 0, vertical flag = 1
    void boxFilter(Mat& inputImage, Mat& outputImage);
    void lowPassFilter(Mat& inputImage, Mat& outputImage, int filterRadius);
    void highPassFilter(Mat& inputImage, Mat& outputImage, int filterRadius);
    void carNumber2(Mat& number, Mat& symbol);

private:
    bool isPowerOfTwo(size_t n);
    Mat m_inputImage;
    Mat m_spectrum;
    Mat m_phase;
    Mat m_imageAfterDFT;

};