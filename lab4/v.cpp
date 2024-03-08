#include <iostream>
#include "cmath"
#include "array"
#include "chrono"

#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/imgproc.hpp>
#include <opencv4/opencv2/highgui.hpp>


cv::Mat custom_dft(cv::Mat input_img, bool isForward)
{
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


cv::Mat show(cv::Mat input_img, cv::Mat *planes_4_func, bool isForward){

    cv::split(input_img, planes_4_func);                   // planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))
    cv::Mat magI;
    cv::magnitude(planes_4_func[0], planes_4_func[1], magI);// planes[0] = magnitude

    magI += cv::Scalar::all(1);                    // switch to logarithmic scale
    if(isForward){
        log(magI, magI);
    }
    // rearrange the quadrants of Fourier image  so that the origin is at the image center
    krasivSpektr(magI);

    if (isForward) {
        cv::normalize(magI, magI, 0, 255, cv::NORM_MINMAX);
        magI.convertTo(magI, CV_8UC3);
        cv::applyColorMap(magI, magI, cv::COLORMAP_JET);
    }
    else {
        cv::normalize(magI, magI, 0, 1, cv::NORM_MINMAX);
    }

    return magI;
}


std::array<cv::Mat, 2> forward_fourier_tf_comparison(cv::Mat starting_image){

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
    tf_img_my = custom_dft(complexImg_my, true);
    auto end_my = std::chrono::high_resolution_clock::now();
    auto duration_my = std::chrono::duration_cast<std::chrono::microseconds>(end_my - start_my);

    auto start_builtIn = std::chrono::high_resolution_clock::now();
    cv::dft(complexImg_builtIn, tf_img_builtIn, cv::DFT_COMPLEX_OUTPUT);
    auto end_builtIn = std::chrono::high_resolution_clock::now();
    auto duration_builtIn = std::chrono::duration_cast<std::chrono::microseconds>(end_builtIn - start_builtIn);

    std::cout << "custom DFT time " << duration_my.count() << " mks" << "\n";
    std::cout << "opencv DFT time " << duration_builtIn.count() << " mks" << "\n";

    cv::Mat res_my  = show(tf_img_my, planes_my, true);
    cv::Mat res_builtIn = show(tf_img_builtIn, planes_builtIn, true);

    std::string input_name = "input image";
    cv::imshow(input_name, starting_image);

    cv::imshow("custom dft", res_my);
    std::string builtIn = "opencv dft";
    cv::imshow(builtIn, res_builtIn);

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
    tf_img_my_back = custom_dft(transformed_img[0], false);
    cv::dft(transformed_img[1], tf_img_builtIn_back, cv::DFT_INVERSE|cv::DFT_REAL_OUTPUT);

    cv::Mat res_my_back  = show(tf_img_my_back, planes_my_back, false);
    cv::Mat res_builtIn_back = show(tf_img_builtIn_back, planes_builtIn_back, false);

    krasivSpektr(res_my_back);
    krasivSpektr(res_builtIn_back);

    cv::imshow("custom dft back", res_my_back);
    cv::imshow("opencv dft back", res_builtIn_back);
}


void fft_forward(std::vector<std::complex<float>> &x, int N) {
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
    fft_forward(even, N / 2);
    fft_forward(odd, N / 2);


    // Calculate DFT
    for (int k = 0; k < N / 2; k++) {
        std::complex<float> W = exp(std::complex<float>(0, -2 * CV_PI * k / N));

        x[k] = even[k] + W * odd[k];
        x[N / 2 + k] = even[k] - W * odd[k];
    }
}


void fft_backward(std::vector<std::complex<float>> &x, int N) {
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
    fft_backward(even, N / 2);
    fft_backward(odd, N / 2);


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
        fft_forward(x_out, N);

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
    fft_backward(x_out, N);

    cv::Mat x_out_(x_in.rows, x_in.cols, CV_32FC2);
    for (int i = 0; i < x_out_.cols; i++) {
        x_out_.at<cv::Point2f>(0, i) = cv::Point2f(x_out[i].real(), x_out[i].imag());
    }

    return x_out_;
}


cv::Size getOptSize(cv::Mat src) {
    cv::Size sz;
    sz.height = cv::getOptimalDFTSize( src.rows );
    sz.width = cv::getOptimalDFTSize( src.cols );
    return sz;
}


void fft2df(cv::Mat src) {
    cv::Mat empty;
    cv::Size sz = getOptSize(src);
    
    cv::copyMakeBorder(src, empty, 0, sz.height - src.rows, 0, sz.width - src.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));
    cv::Mat planes[] = {cv::Mat_<float>(empty), cv::Mat::zeros(empty.size(), CV_32F)};
    cv::Mat complexImg_my;
    cv::merge(planes, 2, complexImg_my);

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

    std::cout << "fft duration " << duration_fft.count() << " microseconds" << std::endl;

    cv::imshow("custom fft forward", show(complexImg_my, planes, true));
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

    cv::Mat res_my  = show(complexImg_my, planes_my_fft, false);

    krasivSpektr(res_my);

    cv::imshow("custom fft backward", res_my);
}


int main() {
    cv::Mat starting_image = cv::imread("/home/user/Downloads/lenna(2).jpg", cv::IMREAD_GRAYSCALE);
    cv::resize(starting_image, starting_image, cv::Size(512, 512));

    std::array<cv::Mat, 2> output_images;

    output_images = forward_fourier_tf_comparison(starting_image);

    backward_fourier_tf_comparison(starting_image, output_images);

    fast_fourier_tf_forward(starting_image);

    fast_fourier_tf_backward(starting_image, output_images);
    
    cv::waitKey(0);
    return 0;
}