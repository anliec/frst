#include "gfrstMultiscale.h"

#include <vector>

void gfrstMultiscale(const cv::Mat &image, std::vector<std::vector<cv::Mat>> &outputVoteVector, const unsigned &minRadius, const unsigned &maxRadius,
                     const int &numberOfSides, const int &mode, const unsigned &radiusStep)
{
    std::vector<std::vector<cv::Mat>> resultsMatrixArray;
    for(unsigned radius=minRadius ; radius<=maxRadius ; radius+=radiusStep)
    {
        // init the double vector structure
        resultsMatrixArray.emplace_back();
        resultsMatrixArray.back().emplace_back();
        resultsMatrixArray.back().emplace_back();
        resultsMatrixArray.back().emplace_back();
        // compute GFRST and stor the results in the vectors
        gfrst2d(image, resultsMatrixArray.back()[0], resultsMatrixArray.back().data() + 1, radius, mode, numberOfSides);
    }
    outputVoteVector = resultsMatrixArray;
}



std::vector<Polygone> detectPolygones(const cv::Mat &image, const int &minRadius, const int &maxRadius, const int &numberOfSides,
                                      const int& mode, const double& threshold, const unsigned& radiusStep)
{
    std::vector<std::vector<cv::Mat>> voteLayers;
    gfrstMultiscale(image, voteLayers, minRadius, maxRadius, numberOfSides, mode, radiusStep);

    std::vector<std::pair<double, Polygone>> detectedPolygone;
    for(unsigned l=0 ; l<voteLayers.size() ; ++l){
        std::vector<cv::Mat> layers = voteLayers[l];
        unsigned radius = (l * radiusStep) + minRadius;

        double maxVote, minVote;
        cv::Point maxPos, minPos;

        while (true){
            cv::minMaxLoc(layers[0], &minVote, &maxVote, &minPos, &maxPos);
            if(maxVote > threshold){
                double angle = std::atan2(layers[2].at<double>(maxPos),
                                          layers[1].at<double>(maxPos));
                angle /= numberOfSides;
                detectedPolygone.emplace_back(maxVote, Polygone(maxPos, radius, angle));

                // remove all the points in the current polygone area
                cv::circle(layers[0], maxPos, radius * 2, 0, -1);
            }
            else{
                std::cout << "break at " << maxVote << std::endl;
                break;
            }
        }
    }

    // filter superposed polygones
    std::vector<Polygone> filteredPolygones;
    for(std::pair<double, Polygone> p1 : detectedPolygone){
        bool alreadyFund = false;
        for(Polygone fp : filteredPolygones){
            if(cv::norm(p1.second.center - fp.center) < fp.radius){
                alreadyFund = true;
                break;
            }
        }
        if(alreadyFund)
            continue;
        bool canBeAdded = true;
        for(std::pair<double, Polygone> p2 : detectedPolygone){
            if(p2.first > p1.first){
                int dist = cv::norm(p1.second.center - p2.second.center);
                if(dist < p1.second.radius || dist < p2.second.radius){
                    canBeAdded = false;
                    break;
                }
            }
        }
        if(canBeAdded){
            filteredPolygones.push_back(p1.second);
        }
    }

    return filteredPolygones;
}
