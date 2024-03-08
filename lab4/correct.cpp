#include <iostream>
#include <opencv2/opencv.hpp>
#include "stdio.h"
#include <opencv2/imgproc.hpp>
#include "cmath"
#include "array"
#include "ctime"
#include "chrono"

cv::Mat my_own_dft(cv::Mat input_img, bool isForward)
{
    //Если преобразование прямое, дело заходит в эту часть и производит его
    //Если нет, часть пропускается, и преобразование происходит обратное
    if(isForward){
        cv::Mat_<cv::Complex<float>> complex_img(input_img.rows, input_img.cols);

        for(int row = 0; row < input_img.rows; row++){
            for(int col = 0; col < input_img.cols; col++){
                complex_img.at<cv::Complex<float>>(row, col).re = input_img.at<cv::Vec2f>(row, col)[0];
                complex_img.at<cv::Complex<float>>(row, col).im = input_img.at<cv::Vec2f>(row, col)[1];
            }
        }

        cv::Mat W_matrix_row = cv::Mat(input_img.rows, input_img.rows, CV_32FC2);
        cv::Mat W_matrix_col = cv::Mat(input_img.cols, input_img.cols, CV_32FC2);

        for(int row = 0; row < W_matrix_row.rows; row++){
            for(int col = 0; col < W_matrix_row.cols; col++){
                float var_r = -1 * (2 * M_PI  * row * col / (input_img.rows));
                cv::Complex<float> complex_num_row;
                complex_num_row.re = cos(var_r);
                complex_num_row.im = sin(var_r);
                W_matrix_row.at<cv::Complex<float>>(row, col) = complex_num_row;
            }
        }

        for(int row = 0; row < W_matrix_col.rows; row++){
            for(int col = 0; col < W_matrix_col.cols; col++){
                float var_c = -1 * (2 * M_PI  * row * col / (input_img.cols));
                cv::Complex<float> complex_num_col;
                complex_num_col.re = cos(var_c);
                complex_num_col.im = sin(var_c);
                W_matrix_col.at<cv::Complex<float>>(row, col) = complex_num_col;
            }
        }

        cv::Mat result_img = cv::Mat(input_img.rows, input_img.cols, CV_32FC1);
        result_img = W_matrix_col * complex_img * W_matrix_row;

        /*
        for(int row = 0; row < rows_num; row++){
            complex_img.row(row) = complex_img.row(row) * W_matrix_row;
        }

        for(int col = 0; col < cols_num; col++){
            complex_img.col(col) = W_matrix_col * complex_img.col(col);
        }*/

        return result_img;
    }

    cv::Mat_<cv::Complex<float>> complex_img(input_img.rows, input_img.cols);

    for(int row = 0; row < input_img.rows; row++){
        for(int col = 0; col < input_img.cols; col++){
            complex_img.at<cv::Complex<float>>(row, col).re = input_img.at<cv::Vec2f>(row, col)[0];
            complex_img.at<cv::Complex<float>>(row, col).im = input_img.at<cv::Vec2f>(row, col)[1];
        }
    }

    cv::Mat W_matrix_row = cv::Mat(input_img.rows, input_img.rows, CV_32FC2);
    cv::Mat W_matrix_col = cv::Mat(input_img.cols, input_img.cols, CV_32FC2);

    for(int row = 0; row < W_matrix_row.rows; row++){
        for(int col = 0; col < W_matrix_row.cols; col++){
            float var_r = (2 * M_PI  * row * col / (input_img.rows));
            cv::Complex<float> complex_num_row;
            complex_num_row.re = cos(var_r)  / 1;
            complex_num_row.im = sin(var_r)  / 1;
            W_matrix_row.at<cv::Complex<float>>(row, col) = complex_num_row;
        }
    }

    for(int row = 0; row < W_matrix_col.rows; row++){
        for(int col = 0; col < W_matrix_col.cols; col++){
            float var_c = (2 * M_PI  * row * col / (input_img.cols));
            cv::Complex<float> complex_num_col;
            complex_num_col.re = cos(var_c)  / 1;
            complex_num_col.im = sin(var_c)  / 1;
            W_matrix_col.at<cv::Complex<float>>(row, col) = complex_num_col;
        }
    }

    cv::Mat result_img = cv::Mat(input_img.rows, input_img.cols, CV_32FC1);
    result_img = W_matrix_col * complex_img * W_matrix_row;

    /*
    for(int row = 0; row < rows_num; row++){
        complex_img.row(row) = complex_img.row(row) * W_matrix_row;
    }

    for(int col = 0; col < cols_num; col++){
        complex_img.col(col) = W_matrix_col * complex_img.col(col);
    }*/

    return result_img;
}

void krasivSpektr(cv::Mat &magI){
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

cv::Mat good_picture(cv::Mat input_img, cv::Mat *planes_4_func, bool isForward){

    cv::split(input_img, planes_4_func);                   // planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))
    cv::Mat magI;
    cv::magnitude(planes_4_func[0], planes_4_func[1], magI);// planes[0] = magnitude

    magI += cv::Scalar::all(1);                    // switch to logarithmic scale
    if(isForward){
        log(magI, magI);
    }
    // crop the spectrum, if it has an odd number of rows or columns
    //magI = magI(cv::Rect(0, 0, magI.cols & -2, magI.rows & -2));
    // rearrange the quadrants of Fourier image  so that the origin is at the image center
    krasivSpektr(magI);
    cv::normalize(magI, magI, 0, 1, cv::NORM_MINMAX); // Transform the matrix with float values into a
    // viewable image form (float between values 0 and 1).

    return magI;
}

std::array<cv::Mat, 2> forward_fourier_tf_comparison(cv::Mat starting_image, std::string win_name){

    cv::Mat padded;
    int m = cv::getOptimalDFTSize( starting_image.rows );
    int n = cv::getOptimalDFTSize( starting_image.cols );
    cv::copyMakeBorder(starting_image, padded, 0, m - starting_image.rows, 0,
                       n - starting_image.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));
    cv::Mat planes_my[] = {cv::Mat_<float>(padded), cv::Mat::zeros(padded.size(), CV_32F)};
    cv::Mat complexImg_my;
    cv::merge(planes_my, 2, complexImg_my);

    cv::Mat planes_builtIn[] = {cv::Mat_<float>(padded), cv::Mat::zeros(padded.size(), CV_32F)};
    cv::Mat complexImg_builtIn;
    cv::merge(planes_builtIn, 2, complexImg_builtIn);


    cv::Mat tf_img_my;
    cv::Mat tf_img_builtIn;
    auto start_my = std::chrono::high_resolution_clock::now();
    tf_img_my = my_own_dft(complexImg_my, true);
    auto end_my = std::chrono::high_resolution_clock::now();
    auto duration_my = std::chrono::duration_cast<std::chrono::microseconds>(end_my - start_my);

    auto start_builtIn = std::chrono::high_resolution_clock::now();
    cv::dft(complexImg_builtIn, tf_img_builtIn, cv::DFT_COMPLEX_OUTPUT);
    auto end_builtIn = std::chrono::high_resolution_clock::now();
    auto duration_builtIn = std::chrono::duration_cast<std::chrono::microseconds>(end_builtIn - start_builtIn);

    std::cout << "My own DFT time " << duration_my.count() << " mks" << "\n";
    std::cout << "Built in DFT time " << duration_builtIn.count() << " mks" << "\n";

    cv::Mat res_my  = good_picture(tf_img_my, planes_my, true);
    // cv::Mat res_my;
    // spectrum(tf_img_my, res_my, false);
    cv::Mat res_builtIn = good_picture(tf_img_builtIn, planes_builtIn, true);
    // cv::Mat res_builtIn;
    // spectrum(tf_img_builtIn, res_builtIn, false);

    std::string input_name = win_name + " input image";
    imshow(input_name, starting_image);

    imshow(win_name, res_my);
    std::string builtIn = win_name + " built In";
    imshow(builtIn, res_builtIn);

    while (1)
    {
        char c = (char)cv::waitKey(500);
        if (c == 27)
        {
            break;
        }
    }

    std::array<cv::Mat, 2> transformed_img;
    transformed_img[0] = tf_img_my;
    transformed_img[1] = tf_img_builtIn;
    return transformed_img;
}

void backward_fourier_tf_comparison(cv::Mat starting_image, std::array<cv::Mat, 2> transformed_img){

    cv::Mat padded;
    int m = cv::getOptimalDFTSize( starting_image.rows );
    int n = cv::getOptimalDFTSize( starting_image.cols );
    cv::copyMakeBorder(starting_image, padded, 0, m - starting_image.rows, 0,
                       n - starting_image.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));

    cv::Mat planes_my_back[] = {cv::Mat_<float>(padded), cv::Mat::zeros(padded.size(), CV_32F)};
    cv::Mat complexImg_my_back;
    cv::merge(planes_my_back, 2, complexImg_my_back);

    cv::Mat planes_builtIn_back[] = {cv::Mat_<float>(padded), cv::Mat::zeros(padded.size(), CV_32F)};
    cv::Mat complexImg_builtIn_back;
    cv::merge(planes_builtIn_back, 2, complexImg_builtIn_back);


    cv::Mat tf_img_my_back;
    cv::Mat tf_img_builtIn_back;
    tf_img_my_back = my_own_dft(transformed_img[0], false);
    cv::dft(transformed_img[1], tf_img_builtIn_back, cv::DFT_INVERSE|cv::DFT_REAL_OUTPUT);

    //cv::Mat res_builtIn_back = cv::Mat(tf_img_builtIn.rows, tf_img_builtIn.cols, CV_32FC1);
    //res_builtIn_back = tf_img_builtIn;

    cv::Mat res_my_back;
    good_picture(tf_img_my_back, planes_my_back, false);
    cv::Mat res_builtIn_back;
    good_picture(tf_img_builtIn_back, planes_builtIn_back, false);

    krasivSpektr(res_my_back);
    krasivSpektr(res_builtIn_back);

    imshow("Input Image", starting_image);

    imshow("My dft back", res_my_back);
    imshow("Built in dft back", res_builtIn_back);

    while (1)
    {
        char c = (char)cv::waitKey(500);
        if (c == 27)
        {
            break;
        }
    }
}

void fft_rec_forward(std::vector<std::complex<float>> &x, int N) {
// Check if it is splitted enough
    if (N <= 1) {
        return;
    }

    // Split even and odd
    std::vector<std::complex<float>> odd(N / 2);
    std::vector<std::complex<float>> even(N / 2);
    for (int i = 0; i < N / 2; i++) {
        even[i] = x[i * 2];
        odd[i] = x[i * 2 + 1];
    }

    // Split on tasks
    fft_rec_forward(even, N / 2);
    fft_rec_forward(odd, N / 2);


    // Calculate DFT
    for (int k = 0; k < N / 2; k++) {
        std::complex<float> W = exp(std::complex<float>(0, -2 * CV_PI * k / N));

        x[k] = even[k] + W * odd[k];
        x[N / 2 + k] = even[k] - W * odd[k];
    }
}

void fft_rec_backward(std::vector<std::complex<float>> &x, int N) {
// Check if it is splitted enough
    if (N <= 1) {
        return;
    }

    // Split even and odd
    std::vector<std::complex<float>> odd(N / 2);
    std::vector<std::complex<float>> even(N / 2);
    for (int i = 0; i < N / 2; i++) {
        even[i] = x[i * 2];
        odd[i] = x[i * 2 + 1];
    }

    // Split on tasks
    fft_rec_backward(even, N / 2);
    fft_rec_backward(odd, N / 2);


    // Calculate DFT
    for (int k = 0; k < N / 2; k++) {
        std::complex<float> W = exp(std::complex<float>(0, 2 * CV_PI * k / N));

        x[k] = even[k] + W * odd[k];
        x[N / 2 + k] = even[k] - W * odd[k];
    }
}

cv::Mat fft(const cv::Mat &x_in, int N, bool isForward) {
    if(isForward){
        std::vector<std::complex<float>> x_out(N);

        for (int i = 0; i < N; i++) {
            x_out[i] = std::complex<float>(x_in.at<cv::Point2f>(0, i).x, x_in.at<cv::Point2f>(0, i).y);
            x_out[i] *= 1; // Window
        }

        // Start recursion
        fft_rec_forward(x_out, N);

        cv::Mat x_out_(x_in.rows, x_in.cols, CV_32FC2);
        for (int i = 0; i < x_out_.cols; i++) {
            x_out_.at<cv::Point2f>(0, i) = cv::Point2f(x_out[i].real(), x_out[i].imag());
        }

        return x_out_;
    }
    std::vector<std::complex<float>> x_out(N);

    for (int i = 0; i < N; i++) {
        x_out[i] = std::complex<float>(x_in.at<cv::Point2f>(0, i).x, x_in.at<cv::Point2f>(0, i).y);
        x_out[i] *= 1; // Window
    }

    // Start recursion
    fft_rec_backward(x_out, N);

    cv::Mat x_out_(x_in.rows, x_in.cols, CV_32FC2);
    for (int i = 0; i < x_out_.cols; i++) {
        x_out_.at<cv::Point2f>(0, i) = cv::Point2f(x_out[i].real(), x_out[i].imag());
    }

    return x_out_;
}

void fast_fourier_tf_forward(cv::Mat starting_image){

    cv::Mat padded;
    int m = cv::getOptimalDFTSize( starting_image.rows );
    int n = cv::getOptimalDFTSize( starting_image.cols );
    cv::copyMakeBorder(starting_image, padded, 0, m - starting_image.rows, 0,
                       n - starting_image.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));
    cv::Mat planes_my_fft[] = {cv::Mat_<float>(padded), cv::Mat::zeros(padded.size(), CV_32F)};
    cv::Mat complexImg_my;
    cv::merge(planes_my_fft, 2, complexImg_my);

    auto start_fft = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < complexImg_my.rows; i++) {
        complexImg_my.row(i) = fft(complexImg_my.row(i), complexImg_my.cols, true).clone() + 0;
    }
    cv::transpose(complexImg_my, complexImg_my);

    for (int i = 0; i < complexImg_my.rows; i++) {
        complexImg_my.row(i) = fft(complexImg_my.row(i), complexImg_my.cols, true).clone() + 0;
    }
    cv::transpose(complexImg_my, complexImg_my);
    auto end_fft = std::chrono::high_resolution_clock::now();
    auto duration_fft = std::chrono::duration_cast<std::chrono::microseconds>(end_fft - start_fft);

    std::cout << "FFT time " << duration_fft.count() << " mks" << "\n";

    cv::Mat res_my  = good_picture(complexImg_my, planes_my_fft, true);

    cv::applyColorMap(starting_image, starting_image, cv::COLORMAP_JET);
    cv::applyColorMap(res_my, res_my, cv::COLORMAP_JET);

    cv::imshow("Input Image", starting_image);

    cv::imshow("My fft forward", res_my);

    while (1)
    {
        char c = (char)cv::waitKey(500);
        if (c == 27)
        {
            break;
        }
    }
}

void fast_fourier_tf_backward(cv::Mat starting_image, std::array<cv::Mat, 2> transformed_img){

    cv::Mat padded;
    int m = cv::getOptimalDFTSize( starting_image.rows );
    int n = cv::getOptimalDFTSize( starting_image.cols );
    cv::copyMakeBorder(starting_image, padded, 0, m - starting_image.rows, 0,
                       n - starting_image.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));
    cv::Mat planes_my_fft[] = {cv::Mat_<float>(padded), cv::Mat::zeros(padded.size(), CV_32F)};
    cv::Mat complexImg_my;
    cv::merge(planes_my_fft, 2, complexImg_my);

    complexImg_my = transformed_img[0].clone();

    for (int i = 0; i < complexImg_my.rows; i++) {
        complexImg_my.row(i) = fft(complexImg_my.row(i), complexImg_my.cols, false).clone() + 0;
    }

    cv::transpose(complexImg_my, complexImg_my);

    for (int i = 0; i < complexImg_my.rows; i++) {
        complexImg_my.row(i) = fft(complexImg_my.row(i), complexImg_my.cols, false).clone() + 0;
    }

    cv::transpose(complexImg_my, complexImg_my);

    cv::Mat res_my  = good_picture(complexImg_my, planes_my_fft, false);

    krasivSpektr(res_my);

    cv::imshow("Input Image", starting_image);

    cv::imshow("My fft backward", res_my);

    while (1)
    {
        char c = (char)cv::waitKey(500);
        if (c == 27)
        {
            break;
        }
    }
}

void convolution_task(cv::Mat starting_image, cv::Mat kernel, std::string win_name){

    cv::Mat padded_img;
    cv::copyMakeBorder(starting_image, padded_img, 0, kernel.rows, 0,
                       kernel.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));
    cv::Mat planes_img[] = {cv::Mat_<float>(padded_img), cv::Mat::zeros(padded_img.size(), CV_32F)};
    cv::Mat complexImg_img;
    cv::merge(planes_img, 2, complexImg_img);
    cv::Mat tf_img_img;
    cv::dft(complexImg_img, tf_img_img, cv::DFT_COMPLEX_OUTPUT);

    cv::Mat padded_kernel;
    cv::copyMakeBorder(kernel, padded_kernel, 0, starting_image.rows, 0,
                       starting_image.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));
    cv::Mat planes_kernel[] = {cv::Mat_<float>(padded_kernel), cv::Mat::zeros(padded_kernel.size(), CV_32F)};
    cv::Mat complexImg_kernel;
    cv::merge(planes_kernel, 2, complexImg_kernel);
    cv::Mat tf_img_kernel;
    cv::dft(complexImg_kernel, tf_img_kernel, cv::DFT_COMPLEX_OUTPUT);

    //std::cout << tf_img_img.size() << "\n" << tf_img_kernel.size();

    cv::Mat magn_img  = good_picture(tf_img_img, planes_img, true);
    cv::Mat magn_kernel = good_picture(tf_img_kernel, planes_kernel, true);

    std::string input_name = win_name + " input image";
    imshow(input_name, starting_image);

    std::string magn_img_name = win_name + " starting img magn";
    imshow(magn_img_name, magn_img);

    std::string magn_kernel_name = win_name + " kernel magn";
    imshow(magn_kernel_name, magn_kernel);

    cv::Mat multed_img;
    cv::mulSpectrums(tf_img_img, tf_img_kernel, multed_img, 0, false);

    cv::Mat padded_backwar;
    cv::copyMakeBorder(starting_image, padded_backwar, 0, kernel.rows, 0,
                       kernel.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));
    cv::Mat planes_backward[] = {cv::Mat_<float>(padded_backwar), cv::Mat::zeros(padded_backwar.size(), CV_32F)};
    cv::Mat complexImg_backward;
    cv::merge(planes_backward, 2, complexImg_backward);
    cv::Mat multed_img_normal;
    cv::dft(multed_img, multed_img_normal, cv::DFT_INVERSE|cv::DFT_REAL_OUTPUT);

    cv::Mat res_multed  = good_picture(multed_img_normal, planes_backward, false);

    krasivSpektr(res_multed);

    cv::Mat ready_picture = res_multed(cv::Range(0,512),cv::Range(0,512));

    imshow(win_name, ready_picture);

    while (1)
    {
        char c = (char)cv::waitKey(500);
        if (c == 27)
        {
            break;
        }
    }
}

void filter_task(cv::Mat starting_image, bool isUpper, std::string win_name){

        cv::Mat padded_forward;
        int m = cv::getOptimalDFTSize( starting_image.rows );
        int n = cv::getOptimalDFTSize( starting_image.cols );
        cv::copyMakeBorder(starting_image, padded_forward, 0, m - starting_image.rows, 0,
                           n - starting_image.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));
        cv::Mat planes_forward[] = {cv::Mat_<float>(padded_forward), cv::Mat::zeros(padded_forward.size(), CV_32F)};
        cv::Mat complexImg_forward;
        cv::merge(planes_forward, 2, complexImg_forward);
        cv::Mat tf_img_forward;
        cv::dft(complexImg_forward, tf_img_forward, cv::DFT_COMPLEX_OUTPUT);

        cv::Mat msk_mat = cv::Mat::zeros(512, 512, CV_8U);
        int radius = 362;
        if(isUpper){
            radius = 100;
        }
        cv::circle(msk_mat, cv::Point(512/2, 512/2), radius, cv::Scalar::all(255), -1);

        cv::Mat crop_mat;
        if(isUpper){
            krasivSpektr(tf_img_forward);
        }
        cv::copyTo(tf_img_forward, crop_mat, msk_mat);
        cv::Mat res_my  = good_picture(crop_mat, planes_forward, true);
        if(isUpper){
            krasivSpektr(res_my);
        }

        cv::Mat padded_backward;
        cv::copyMakeBorder(starting_image, padded_backward, 0, m - starting_image.rows, 0,
                           n - starting_image.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));
        cv::Mat planes_backward[] = {cv::Mat_<float>(padded_backward), cv::Mat::zeros(padded_backward.size(), CV_32F)};
        cv::Mat complexImg_backward;
        cv::merge(planes_backward, 2, complexImg_backward);
        cv::Mat backward_cropped;
        cv::dft(crop_mat, backward_cropped, cv::DFT_INVERSE|cv::DFT_REAL_OUTPUT);
        cv::Mat res_backward  = good_picture(backward_cropped, planes_backward, false);

        krasivSpektr(res_backward);


        std::string input_name = win_name + " input image";
        imshow(input_name, starting_image);

        std::string cropped_magn_name = win_name + "magnitude of cropped img";
        imshow(cropped_magn_name, res_my);

        std::string cropped_img_name = win_name + " cropping result";
        imshow(cropped_img_name, res_backward);

        while (1)
        {
            char c = (char)cv::waitKey(500);
            if (c == 27)
            {
                break;
            }
        }

        return;
}

void correlation_task(cv::Mat first_image, cv::Mat second_image, std::string win_name){

    cv::Mat padded_first;
    cv::copyMakeBorder(first_image, padded_first, 0, second_image.rows, 0,
                       second_image.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));
    cv::Mat planes_first[] = {cv::Mat_<float>(padded_first), cv::Mat::zeros(padded_first.size(), CV_32F)};
    cv::Mat complexImg_first;
    cv::merge(planes_first, 2, complexImg_first);
    cv::Mat tf_img_first;
    cv::dft(complexImg_first, tf_img_first, cv::DFT_COMPLEX_OUTPUT);

    cv::Mat padded_second;
    cv::copyMakeBorder(second_image, padded_second, 0, first_image.rows, 0,
                       first_image.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));
    cv::Mat planes_second[] = {cv::Mat_<float>(padded_second), cv::Mat::zeros(padded_second.size(), CV_32F)};
    cv::Mat complexImg_second;
    cv::merge(planes_second, 2, complexImg_second);
    cv::Mat tf_img_second;
    cv::dft(complexImg_second, tf_img_second, cv::DFT_COMPLEX_OUTPUT);

    // std::cout << tf_img_first.size() << "\n" << tf_img_second.size();

    cv::Mat magn_first  = good_picture(tf_img_first, planes_first, true);
    cv::Mat magn_second = good_picture(tf_img_second, planes_second, true);

    std::string input_number = win_name + " number";
    imshow(input_number, first_image);
    std::string input_A = win_name + " A";
    imshow(input_A, second_image);

    std::string magn_first_name = win_name + " number img magn";
    imshow(magn_first_name, magn_first);
    std::string magn_second_name = win_name + " A img magn";
    imshow(magn_second_name, magn_second);

    cv::Mat multed_img;
    cv::mulSpectrums(tf_img_first, tf_img_second, multed_img, 0, true);

    cv::Mat padded_backwar;
    cv::copyMakeBorder(second_image, padded_backwar, 0, first_image.rows, 0,
                       first_image.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));
    cv::Mat planes_backward[] = {cv::Mat_<float>(padded_backwar), cv::Mat::zeros(padded_backwar.size(), CV_32F)};
    cv::Mat complexImg_backward;
    cv::merge(planes_backward, 2, complexImg_backward);
    cv::Mat multed_img_normal;
    cv::dft(multed_img, multed_img_normal, cv::DFT_INVERSE|cv::DFT_REAL_OUTPUT);

    cv::Mat res_multed  = good_picture(multed_img_normal, planes_backward, false);

    krasivSpektr(res_multed);

    cv::Mat ready_picture = res_multed(cv::Range(0,first_image.rows),cv::Range(0,first_image.cols));

    double max_pixel;
    cv::minMaxLoc(ready_picture, NULL, &max_pixel, NULL, NULL);

    double threshold = max_pixel - 0.01;

    cv::Mat out_picture;
    cv::threshold(ready_picture, out_picture, threshold, max_pixel, cv::THRESH_BINARY);

    imshow(win_name, out_picture);

    while (1)
    {
        char c = (char)cv::waitKey(500);
        if (c == 27)
        {
            break;
        }
    }


}


int main()
{
    cv::Mat starting_image = cv::imread("/home/user/Downloads/lenna(2).jpg", cv::IMREAD_GRAYSCALE);
    cv::resize(starting_image, starting_image, cv::Size(512, 512));

    // cv::Mat number_image = cv::imread("C:/WIP/Tex3penie/Laba_4_Clion/Fourier2.jpg", cv::IMREAD_GRAYSCALE);
    // number_image = 255 - number_image;
    // cv::resize(number_image, number_image, cv::Size(cv::getOptimalDFTSize(number_image.cols),
    //                                                                  cv::getOptimalDFTSize( number_image.rows)));
    // cv::Mat A_image = cv::imread("C:/WIP/Tex3penie/Laba_4_Clion/glaza2.jpg", cv::IMREAD_GRAYSCALE);
    // A_image = 255 - A_image;
    // cv::resize(A_image, A_image, cv::Size(cv::getOptimalDFTSize(A_image.cols),
    //                                                                  cv::getOptimalDFTSize( A_image.rows)));

    std::array<cv::Mat, 2> output_images;

    output_images = forward_fourier_tf_comparison(starting_image, "First test");

    backward_fourier_tf_comparison(starting_image, output_images);

    fast_fourier_tf_forward(starting_image);

    fast_fourier_tf_backward(starting_image, output_images);

    while (1)
    {
       char c = (char)cv::waitKey(0);
       if (c == 27)
       {
           break;
       }
    }
    cv::destroyAllWindows();
//
    //cv::Mat sobelV_core = (cv::Mat_<double>(3, 3) << -1, 0, 1, -2, 0, 2, -1, 0, 1);
    //cv::Mat sobelH_core = (cv::Mat_<double>(3, 3) << 1, 2, 1, 0, 0, 0, -1, -2, -1);
    //cv::Mat box_core = (cv::Mat_<double>(3, 3) << 1, 1, 1, 1, 1, 1, 1, 1, 1);
    //cv::Mat laplacian_core = (cv::Mat_<double>(3, 3) << 0, 1, 0, 1, -4, 1, 0, 1, 0);
//
    //convolution_task(starting_image, sobelV_core, "SobelV dft");
    //convolution_task(starting_image, sobelH_core, "SobelH dft");
    //convolution_task(starting_image, box_core, "Box dft");
    //convolution_task(starting_image, laplacian_core, "Laplacian dft");
//
    //while (1)
    //{
    //    char c = (char)cv::waitKey(500);
    //    if (c == 27)
    //    {
    //        break;
    //    }
    //}
    //cv::destroyAllWindows();
//
    //filter_task(starting_image, true, "Upper ");
    //filter_task(starting_image, false, "Lower ");
//
    //while (1)
    //{
    //    char c = (char)cv::waitKey(500);
    //    if (c == 27)
    //    {
    //        break;
    //    }
    //}
    //cv::destroyAllWindows();

    // correlation_task(number_image, A_image, "correlaion");

    return 0;
}