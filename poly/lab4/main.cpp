#include "fourier.hpp"
#include "timer.hpp"


int main() {
    _TickMeter timer;

    cv::Mat src = cv::imread("/home/user/Downloads/lenna(2).jpg", cv::IMREAD_GRAYSCALE);

    // cv::Mat car_number = cv::imread("/home/user/Downloads/number.jpg", cv::IMREAD_GRAYSCALE);
    // cv::Mat symbol_8 = cv::imread("/home/user/Downloads/eight.jpg", cv::IMREAD_GRAYSCALE);
    // cv::Mat symbol_a = cv::imread("/home/user/Downloads/a.jpg", cv::IMREAD_GRAYSCALE);
    // cv::Mat symbol_zero = cv::imread("/home/user/Downloads/zero.jpg", cv::IMREAD_GRAYSCALE);

    cv::Mat opt = Fourier::getOptimalDftSize(src, CV_32FC1);

    timer.start();
    cv::Mat dft = Fourier::dft(opt);
    timer.stop();
    std::cout << "custom dft time: " << timer.getTimeSec() << std::endl;
    timer.reset();

    timer.start();
    cv::Mat idft = Fourier::idft(dft);
    timer.stop();
    std::cout << "custom idft time: " << timer.getTimeSec() << std::endl;
    timer.reset();

    cv::Mat cv_dft(cv::Size(src.cols, src.rows), CV_32FC2);
    timer.start();
    cv::dft(opt, cv_dft, cv::DFT_COMPLEX_OUTPUT);
    timer.stop();
    std::cout << "cv idft time: " << timer.getTimeSec() << std::endl;
    timer.reset();

    std::vector<std::complex<double>> dft_vector = Fourier::mat2vec(opt);
	timer.start();
	Fourier::radixTransform(dft_vector, 0);
	cv::Mat dft_radix = Fourier::vec2mat(dft_vector, opt.rows, opt.cols, CV_32FC2);
	timer.stop();
	std::cout << "radix dft time: " << timer.getTimeSec() << std::endl;
	timer.reset();

    timer.start();
	Fourier::radixTransform(dft_vector, 1);
	cv::Mat idft_radix = Fourier::vec2mat(dft_vector, opt.rows, opt.cols, CV_32FC1);
	timer.stop();
	std::cout << "radix idft time: " << timer.getTimeSec() << std::endl;
	timer.reset();

    cv::imshow("src", src);
    Fourier::show("dft", dft);
    Fourier::show("cv", cv_dft);
    cv::imshow("idft", idft);
    Fourier::show("radix", dft_radix);
    cv::imshow("i_radix", idft_radix);

    // cv::Mat sobel_x = cv::Mat_<float>(3, 3) << (-1, 0, 1, -2, 0, 2, -1, 0, 1);
    // cv::Mat sobel_y = cv::Mat_<float>(3, 3) << (-1, -2, -1, 0, 0, 0, 1, 2, 1);
    // float v = 1.0 / 9.0;
    // cv::Mat box = cv::Mat_<float>(3, 3) << (v, v, v, v, v, v, v, v, v);
    // cv::Mat laplace = cv::Mat_<float>(3, 3) << (0, 1, 0, 1, -4, 1, 0, 1, 0);

    // cv::Mat dft_sobel_x = Fourier::getOptimalDftSize(sobel_x, CV_32FC1);
    // cv::Mat dft_sobel_y = Fourier::getOptimalDftSize(sobel_y, CV_32FC1);
    // cv::Mat dft_box = Fourier::getOptimalDftSize(box, CV_32FC1);
    // cv::Mat dft_laplace = Fourier::getOptimalDftSize(laplace, CV_32FC1);

    // Fourier::createKernel(sobel_x, sobel_x, 128, 128);
    // cv::imshow("sobel_x", sobel_x);
    // Fourier::createKernel(sobel_y, sobel_y, 128, 128);
    // cv::imshow("sobel_y", sobel_y);
    // Fourier::createKernel(box, box, 128, 128);
    // cv::imshow("box", box);
    // Fourier::createKernel(laplace, laplace, 128, 128);
    // cv::imshow("laplace", laplace);

    // cv::Mat s_x = Fourier::convolution(opt, dft_sobel_x);
    // cv::imshow("s_x", s_x);
    // cv::Mat s_y = Fourier::convolution(opt, dft_sobel_x);
    // cv::imshow("s_y", s_y);
    // cv::Mat b = Fourier::convolution(opt, dft_sobel_x);
    // cv::imshow("b", b);
    // cv::Mat l = Fourier::convolution(opt, dft_sobel_x);
    // cv::imshow("l", l);

    // cv::Mat low_spec;
    // Fourier::cutLowSpec(opt, low_spec, 180);
    // Fourier::spectrum(low_spec, low_spec);
    // Fourier::show("low_spec", low_spec);

    // cv::Mat high_spec;
    // Fourier::cutHighSpec(opt, high_spec, 180);
    // Fourier::spectrum(high_spec, high_spec);
    // Fourier::show("low_spec", high_spec);

    cv::waitKey(0);
    return 0;
}