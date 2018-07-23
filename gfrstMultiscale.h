#ifndef GFRST_MULTISCALE_H
#define GFRST_MULTISCALE_H

#include <opencv2/opencv.hpp>

#include "gfrst.h"
#include "polygone.h"

void gfrstMultiscale(const cv::Mat& image, std::vector<std::vector<cv::Mat>> &outputVoteVector, const unsigned &minRadius, const unsigned &maxRadius,
                     const int& numberOfSides, const int& mode, const unsigned &radiusStep=1);

std::vector<Polygone> detectPolygones(const cv::Mat& image, const int& minRadius, const int& maxRadius, const int& numberOfSides,
                                      const int &mode, const double &threshold, const unsigned &radiusStep=1);

#endif // GFRST_MULTISCALE_H
