#include "gfrst.h"

#include <iostream>
#include <string.h>

int main(int argc, char* argv[]) {
    cv::Mat image;
    if (argc > 1) {
        image = cv::imread(argv[1]);
    }
    else {
        image = cv::imread("image.jpeg");
    }
    if (!image.data) {
        std::cout << "Could not open or find the image" << std::endl;
        return -1;
    }

    cv::namedWindow("Gradient Field", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Gradient Norm", cv::WINDOW_AUTOSIZE);

    // lose the Alpha channel
    if (image.channels() == 4) {
        cv::cvtColor(image, image, CV_BGRA2BGR);
    }
    cv::Mat grayImg;
    cv::cvtColor(image, grayImg, CV_BGR2GRAY);
    cv::GaussianBlur(grayImg, grayImg, cv::Size(3,3), 0);

    int width = image.cols;
    int height = image.rows;
    int flowResolution = 2;

    cv::Mat gx, gy;
    gradx(grayImg, gx);
    grady(grayImg, gy);

    for (int i = 0 ; i < height ; i += flowResolution){
        for (int j = 0 ; j < width ; j+= flowResolution){
            cv::Point2f p(j,i);
            double gNorm = std::pow(gx.at<double>(p), 2) + std::pow(gy.at<double>(p), 2);
            if(gNorm > 200.0){
                double angle = std::atan2(gy.at<double>(p), gx.at<double>(p)) * 4;

                cv::Point2f p2(cv::Point2f(30 * std::cos(angle), 30 * std::sin(angle)) + p);
                cv::arrowedLine(image,p,p2,cv::Scalar(0,200,0),1.5,8,0,0.1);
            }

        }
    }

    cv::Mat squareGradientNorm;
    cv::add(gx.mul(gx), gy.mul(gy), squareGradientNorm);
    cv::sqrt(squareGradientNorm, squareGradientNorm);
    cv::normalize(squareGradientNorm, squareGradientNorm, 0.0, 255.0, cv::NORM_MINMAX);
    squareGradientNorm.convertTo(squareGradientNorm, CV_8U, 1.0);

    for (;;) {
        cv::imshow("Gradient Field", image);
        cv::imshow("Gradient Norm", squareGradientNorm);
        char ch = cv::waitKey(0);
        if (char(ch) == 27 || ch == 'q') // esc key
            break;
    }
}
