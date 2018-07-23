#include "gfrst.h"

#include <iostream>
#include <string.h>
#include <chrono>

int main(int argc, char* argv[]) {


	cv::Mat image;
    int radius = 80;
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
    if (argc > 2)
    {
        radius = std::atoi(argv[2]);
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
    cv::GaussianBlur(grayImg, grayImg, cv::Size(3, 3), 0);
    cv::imshow("Gray Image", grayImg);

	// apply FRST
	cv::Mat frstImage;
    cv::Mat gfrstVectorScore[2];
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    gfrst2d(grayImg, frstImage, gfrstVectorScore, radius, FRST_MODE_BOTH, numberOfSide);
    std::chrono::steady_clock::time_point end= std::chrono::steady_clock::now();

//    cv::GaussianBlur(frstImage, frstImage, cv::Size(11, 11), 0);

    double min, max;
    cv::Point minLoc, maxLoc;
    cv::minMaxLoc(frstImage, &min, &max, &minLoc, &maxLoc);
    std::cout << "max value: " << max << std::endl;
    std::cout << "min value: " << min << std::endl;

    // the frst will have irregular values, normalize them!
    cv::normalize(frstImage, frstImage, 0.0, 255.0, cv::NORM_MINMAX);
    frstImage.convertTo(frstImage, CV_8U, 1.0);

	// the frst image is grayscale, let's binarize it
	cv::Mat markers;
//	cv::threshold(frstImage, frstImage, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
    cv::threshold(frstImage, markers, 150, 255, CV_THRESH_BINARY);
    bwMorph(markers, markers, cv::MORPH_CLOSE, cv::MORPH_ELLIPSE, 5);

	// the 'markers' image contains dots of different size. Let's vectorize it
	std::vector< std::vector<cv::Point> > contours;
	std::vector<cv::Vec4i> hierarchy;

	contours.clear();
	cv::findContours(markers, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

	// get the moments
	std::vector<cv::Moments> mu(contours.size());
    for (unsigned i = 0; i < contours.size(); i++)
	{
		mu[i] = moments(contours[i], false);
	}

	//  get the mass centers:
	std::vector<cv::Point2f> mc(contours.size());
    for (unsigned i = 0; i < contours.size(); i++)
	{
		mc[i] = cv::Point2f(mu[i].m10 / mu[i].m00, mu[i].m01 / mu[i].m00);
	}

    std::chrono::steady_clock::time_point stop = std::chrono::steady_clock::now();
    std::cout << "init run time " << std::chrono::duration_cast<std::chrono::microseconds>(begin - start).count() / 1000.0 << " ms" <<std::endl;
    std::cout << "GFRST run time " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() / 1000.0 << " ms" <<std::endl;
    std::cout << "end run time " << std::chrono::duration_cast<std::chrono::microseconds>(stop - end).count() / 1000.0 << " ms" <<std::endl;

	// draw the point centers	
    for(cv::Point2f& p : mc)
	{		
        if(std::isnan(p.x) || std::isnan(p.y))
            continue;

        cv::circle(image, p, radius, CV_RGB(0,255,0), 1, 8, 0);
        double angle = std::atan2(gfrstVectorScore[1].at<double>(p),
                                  gfrstVectorScore[0].at<double>(p));
        angle /= numberOfSide;
        cv::RotatedRect rRect = cv::RotatedRect(p, cv::Size2f(2*radius,2*radius), angle * 180.0 / M_PI);
        cv::Point2f vertices[4];
        rRect.points(vertices);
        for (int i = 0; i < 4; i++)
        {
            cv::line(image, vertices[i], vertices[(i+1)%4], cv::Scalar(0,255,0), 1);
//            cv::line(frstImage, vertices[i], vertices[(i+1)%4], 255, 1);
        }
	}
	
	// display the image
	for (;;) {
        cv::imshow("Display window", image);
        cv::imshow("Votes", frstImage);

        char ch = cv::waitKey(0);

        if (char(ch) == 27 || ch == 'q') // esc key
			break;
	}

	return 0;
}

