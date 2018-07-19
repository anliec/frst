#include "gfrst.h"

#include <cmath>
#include <exception>

#define M_PI 3.14159265358979323846  /* pi */



void grady(const cv::Mat& input, cv::Mat &output)
{
    output = cv::Mat::zeros(input.size(), CV_64FC1);
    for (int y = 0; y<input.rows; y++)
    {
        for (int x = 1; x<input.cols - 1; x++)
        {
            *((double*)output.data + y*output.cols + x) = (double)(*(input.data + y*input.cols + x + 1) - *(input.data + y*input.cols + x - 1)) / 2;
        }
    }
}



void gradx(const cv::Mat& input, cv::Mat &output)
{
    output = cv::Mat::zeros(input.size(), CV_64FC1);
    for (int y = 1; y<input.rows - 1; y++)
    {
        for (int x = 0; x<input.cols; x++)
        {
            *((double*)output.data + y*output.cols + x) = (double)(*(input.data + (y + 1)*input.cols + x) - *(input.data + (y - 1)*input.cols + x)) / 2;
        }
    }
}



void frst2d(const cv::Mat& inputImage, cv::Mat& outputImage, const int& radii, const double& alpha, const double& stdFactor, const int& mode, const int sideNumber)
{
    int width = inputImage.cols;
    int height = inputImage.rows;

    cv::Mat gx, gy;
    gradx(inputImage, gx);
    grady(inputImage, gy);

    // set dark/bright mode
    bool dark = false;
    bool bright = false;

    if (mode == FRST_MODE_BRIGHT)
        bright = true;
    else if (mode == FRST_MODE_DARK)
        dark = true;
    else if (mode == FRST_MODE_BOTH) {
        bright = true;
        dark = true;
    }
    else {
        throw std::runtime_error("invalid mode!");
    }

    // define line limit parametter
    int w = (int)std::round(radii * std::tan(M_PI / double(sideNumber)));

    int borderOffset = radii + 2 * w;

    outputImage = cv::Mat::zeros(inputImage.size(), CV_64FC1);

    cv::Mat S = cv::Mat::zeros(inputImage.rows + 2 * borderOffset, inputImage.cols + 2 * borderOffset, outputImage.type());

    cv::Mat O_n = cv::Mat::zeros(S.size(), CV_64FC1);
    cv::Mat Bx_n = cv::Mat::zeros(S.size(), CV_64FC1);
    cv::Mat By_n = cv::Mat::zeros(S.size(), CV_64FC1);

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            cv::Point p(i, j);

            cv::Vec2d g = cv::Vec2d(gx.at<double>(i, j), gy.at<double>(i, j));

            double gNorm = std::sqrt(g.val[0] * g.val[0] + g.val[1] * g.val[1]);
            // compute n times the gradient angle to have a 2*pi constant value for each side of a regular polygon
            double nAngle = std::atan2(g.val[1], g.val[0]) * double(sideNumber);

            if (gNorm > 0) {

                cv::Vec2i gp;
                gp.val[0] = (int)std::round((g.val[0] / gNorm) * radii);
                gp.val[1] = (int)std::round((g.val[1] / gNorm) * radii);

                cv::Point pos(p.x + borderOffset, p.y + borderOffset);
                voteOnLine(O_n, Bx_n, By_n, gp, gNorm, nAngle, pos, bright, dark, w, radii);
            }
        }
    }
    double min, max;

    O_n = cv::abs(O_n);
    cv::minMaxLoc(O_n, &min, &max);
    O_n = O_n / max;

    cv::Mat Bnorm;
    cv::sqrt(By_n.mul(By_n) + Bx_n.mul(Bx_n), Bnorm);
    cv::minMaxLoc(Bnorm, &min, &max);
    Bnorm = Bnorm / max;

    S = O_n.mul(Bnorm) / (4 * (w * radii) * (w * radii));

    int kSize = std::ceil(radii / 2);
    if (kSize % 2 == 0)
        kSize++;

    cv::GaussianBlur(S, S, cv::Size(kSize, kSize), radii * stdFactor);

    outputImage = S(cv::Rect(borderOffset, borderOffset, width, height));
}



inline void voteOnLine(cv::Mat &O_n, cv::Mat & Bx_n, cv::Mat & By_n, const cv::Vec2i &gp, const double &gNorm, const double &nAngle,
                       const cv::Point &gradientPoint, const bool &bright, const bool &dark, const int & w, const double &radii)
{
    cv::Vec2d lineSuport;
    lineSuport.val[0] = -double(gp.val[1]) / radii;
    lineSuport.val[1] = double(gp.val[0]) / radii;

    // positive vote
    for(int m=-w ; m<=w ; ++m)
    {
        voteAtPos(O_n, Bx_n, By_n, bright, dark, gradientPoint, m, lineSuport, gp, gNorm, nAngle, 1);
    }
    // negative vote
    for(int m=w+1 ; m<=2*w ; ++m)
    {
        // vote at one end of the line
        voteAtPos(O_n, Bx_n, By_n, bright, dark, gradientPoint, m, lineSuport, gp, gNorm, nAngle, -1);
        // vote at the other end
        voteAtPos(O_n, Bx_n, By_n, bright, dark, gradientPoint, -m, lineSuport, gp, gNorm, nAngle, -1);
    }
}



inline void voteAtPos(cv::Mat & O_n, cv::Mat & Bx_n, cv::Mat & By_n, const bool & bright, const bool & dark, const cv::Point & gradientPoint,
                      const int & m, const cv::Vec2d &lineSuport, const cv::Vec2i &gp, const double & gnorm, const double &nAngle, const int & voteValue)
{
    double lineX = m * lineSuport.val[0];
    double lineY = m * lineSuport.val[1];

    if (bright)
    {
        // line center is constant on line remove addition ? (or the compiler will take care of that ?)
        cv::Point pos(gradientPoint.x + gp.val[0] + lineX,
                      gradientPoint.y + gp.val[1] + lineY);

        O_n.at<double>(pos.x, pos.y) = O_n.at<double>(pos.x, pos.y) + voteValue;
        Bx_n.at<double>(pos.x, pos.y) = Bx_n.at<double>(pos.x, pos.y) + voteValue * gnorm * std::cos(nAngle); // optimisation: compute cos/sin only once ?
        By_n.at<double>(pos.x, pos.y) = By_n.at<double>(pos.x, pos.y) + voteValue * gnorm * std::sin(nAngle);
    }

    if (dark)
    {
        cv::Point pos(gradientPoint.x - gp.val[0] + lineX,
                      gradientPoint.y - gp.val[1] + lineY);

        O_n.at<double>(pos.x, pos.y) = O_n.at<double>(pos.x, pos.y) + voteValue;
        Bx_n.at<double>(pos.x, pos.y) = Bx_n.at<double>(pos.x, pos.y) + voteValue * gnorm * std::cos(nAngle);
        By_n.at<double>(pos.x, pos.y) = By_n.at<double>(pos.x, pos.y) + voteValue * gnorm * std::sin(nAngle);
    }
}



void bwMorph(cv::Mat& inputImage, const int operation, const int mShape, const int mSize, const int iterations)
{
    int _mSize = (mSize % 2) ? mSize : mSize + 1;

    cv::Mat element = cv::getStructuringElement(mShape, cv::Size(_mSize, _mSize));
    cv::morphologyEx(inputImage, inputImage, operation, element, cv::Point(-1, -1), iterations);
}



void bwMorph(const cv::Mat& inputImage, cv::Mat& outputImage, const int operation, const int mShape, const int mSize, const int iterations)
{
    inputImage.copyTo(outputImage);

    bwMorph(outputImage, operation, mShape, mSize, iterations);
}
