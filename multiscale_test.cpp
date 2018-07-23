#include "gfrstMultiscale.h"
#include "polygone.h"

#include <iostream>
#include <string.h>
#include <chrono>

int main(int argc, char* argv[]) {
    cv::Mat image;
    int numberOfSide = 4;

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

    cv::namedWindow("Display window", cv::WINDOW_AUTOSIZE);

    // lose the Alpha channel
    if (image.channels() == 4) {
        cv::cvtColor(image, image, CV_BGRA2BGR);
    }

    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();

    // convert to grayscale
    cv::Mat grayImg;
    cv::cvtColor(image, grayImg, CV_BGR2GRAY);
    int blurKernel = std::ceil(grayImg.size[0] / 100.0);
    cv::GaussianBlur(grayImg, grayImg, cv::Size(blurKernel, blurKernel), 0);
    cv::imshow("Gray Image", grayImg);

    // apply FRST
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    std::vector<Polygone> polygones = detectPolygones(grayImg, 40, 110, numberOfSide, FRST_MODE_BOTH, 2.0, 4);
    std::chrono::steady_clock::time_point end= std::chrono::steady_clock::now();


    std::cout << "init run time " << std::chrono::duration_cast<std::chrono::microseconds>(begin - start).count() / 1000.0 << " ms" <<std::endl;
    std::cout << "GFRST run time " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() / 1000.0 << " ms" <<std::endl;

    std::cout << "fund " << polygones.size() << " poylogones" << std::endl;

    // draw the Polygones
    for(Polygone& p : polygones)
    {
        std::cout << "Polygone: radius " << p.radius << " angle " << p.angle << std::endl;
        cv::circle(image, p.center, p.radius, CV_RGB(0,255,0), 1, 8, 0);
        cv::RotatedRect rRect = cv::RotatedRect(p.center, cv::Size2f(2*p.radius,2*p.radius), p.angle * 180.0 / M_PI);
        cv::Point2f vertices[4];
        rRect.points(vertices);
        for (int i = 0; i < 4; i++)
        {
            cv::line(image, vertices[i], vertices[(i+1)%4], cv::Scalar(0,255,0), 1);
        }
    }

    // display the image
    for (;;) {
        cv::imshow("Display window", image);

        char ch = cv::waitKey(0);

        if (char(ch) == 27 || ch == 'q') // esc key
            break;
    }

    return 0;
}
