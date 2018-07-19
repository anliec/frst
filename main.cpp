#include "gfrst.h"

#include <iostream>
#include <string.h>

int main(int argc, char* argv[]) {

	cv::Mat image;
    int radius = 79;
    int numberOfSide = 4;

	if (argc > 1) {
		image = cv::imread(argv[1]);
	}
	else {
		image = cv::imread("image.jpeg");
    }
    if (argc > 2)
    {
        radius = std::atoi(argv[2]);
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

	// convert to grayscale
	cv::Mat grayImg;
	cv::cvtColor(image, grayImg, CV_BGR2GRAY);

	// apply FRST
	cv::Mat frstImage;
    std::pair<cv::Mat,cv::Mat> gfrstVectorScore;
    gfrst2d(grayImg, frstImage, gfrstVectorScore, radius, FRST_MODE_BOTH, numberOfSide);

    double min, max;
    cv::minMaxLoc(frstImage, &min, &max);
    std::cout << "max value: " << max << std::endl;
    std::cout << "min value: " << min << std::endl;

    // the frst will have irregular values, normalize them!
    cv::normalize(frstImage, frstImage, 0.0, 255.0, cv::NORM_MINMAX);
    frstImage.convertTo(frstImage, CV_8U, 1.0);

    cv::imshow("Votes", frstImage);
    cv::minMaxLoc(frstImage, &min, &max);
    std::cout << "max value: " << max << std::endl;
    std::cout << "min value: " << min << std::endl;

	// the frst image is grayscale, let's binarize it
	cv::Mat markers;
//	cv::threshold(frstImage, frstImage, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
    cv::threshold(frstImage, frstImage, 150, 255, CV_THRESH_BINARY);
	bwMorph(frstImage, markers, cv::MORPH_CLOSE, cv::MORPH_ELLIPSE, 5);

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

	// draw the point centers	
    for(cv::Point2f& p : mc)
	{		
        if(std::isnan(p.x) || std::isnan(p.y))
            continue;

        cv::circle(image, p, radius, CV_RGB(0,255,0), 5, 8, 0);
        double angle = std::atan2(gfrstVectorScore.second.at<double>(p.x, p.y),
                                  gfrstVectorScore.first.at<double>(p.x, p.y));
        angle /= numberOfSide;
        cv::RotatedRect rRect = cv::RotatedRect(p, cv::Size2f(2*radius,2*radius), angle * 180 / M_PI);
        cv::Point2f vertices[4];
        rRect.points(vertices);
        for (int i = 0; i < 4; i++)
            cv::line(image, vertices[i], vertices[(i+1)%4], cv::Scalar(0,255,0), 2);
	}
	
	// display the image
	for (;;) {
        cv::imshow("Display window", image);

		char ch = cv::waitKey(10);

        if (char(ch) == 27 || ch == 'q') // esc key
			break;
	}

	return 0;
}

