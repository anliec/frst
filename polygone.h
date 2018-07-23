#ifndef POLYGONE_H
#define POLYGONE_H

#include <opencv2/opencv.hpp>

class Polygone
{
public:
    Polygone(cv::Point polygoneCenter, int polygoneRadius, double polygoneAngle);

    cv::Point center;
    int radius;
    double angle;
};

#endif // POLYGONE_H
