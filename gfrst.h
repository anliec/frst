#ifndef GFRST_H
#define GFRST_H

#include <opencv2/opencv.hpp>

#define FRST_MODE_BRIGHT 1
#define FRST_MODE_DARK 2
#define FRST_MODE_BOTH 3

/**
	Calculate vertical gradient for the input image

	@param input Input 8-bit image
	@param output Output gradient image
*/
void grady(const cv::Mat& input, cv::Mat &output);

/**
	Calculate horizontal gradient for the input image

	@param input Input 8-bit image
	@param output Output gradient image
*/
void gradx(const cv::Mat& input, cv::Mat &output);


/**
 * @brief gfrst2d Applies Generalised Fast radial symmetry transform to image
 * @param inputImage The input grayscale image (8-bit)
 * @param outputImage The output image containing the results of FRST
 * @param radii Gaussian kernel radius
 * @param stdFactor Standard deviation factor
 * @param mode Transform mode ('bright', 'dark' or 'both')
 * @param sideNumber number of sides of the polygon
 */
void gfrst2d(const cv::Mat& inputImage, cv::Mat& outputImage, cv::Mat* outputVector,
             const int& radii, const int& mode, const int sideNumber);

/**
 * @brief voteOnLine Vote for the lines of possible polygone center
 * @param O_n matrix of the vote count for a given center
 * @param Bx_n matrix of the x coordinate of the polygon gradient vector
 * @param By_n matrix of the x coordinate of the polygon gradient vector
 * @param gp gradient vector, normalized to the radius length
 * @param gNorm norm of the original gradient vector
 * @param nAngle angle of the gradient vector
 * @param gradientPoint point where the gradiant was computed
 * @param bright true if we considere the centre to be in the bright direction
 * @param dark true if we considere the centre to be in the dark direction
 * @param w vote line length parameter (positif vote for 2*w+1 plus w negative vote at each end)
 * @param radii internal radius of the polygone we are looking for
 */
inline void voteOnLine(cv::Mat &O_n, cv::Mat & Bx_n, cv::Mat & By_n, const cv::Vec2i &gp, const double &nAngle,
                       const cv::Point &gradientPoint, const bool &bright, const bool &dark, const int & w, const double &radii);
/**
 * @brief voteAtPos Vote for one point of the line, which coordinate are computed with the suport vector and the line parameter
 * @param O_n matrix of the vote count for a given center
 * @param Bx_n matrix of the x coordinate of the polygon gradient vector
 * @param By_n matrix of the x coordinate of the polygon gradient vector
 * @param bright true if we considere the centre to be in the bright direction
 * @param dark true if we considere the centre to be in the dark direction
 * @param gradientPoint point where the gradiant was computed
 * @param m line parameter
 * @param lineSuport unit vector suporting the line
 * @param gp gradient vector, normalized to the radius length
 * @param gNorm norm of the original gradient vector
 * @param nAngle angle of the gradient vector
 * @param voteValue weight of the vote (default: 1)
 */
inline void voteAtPos(cv::Mat & O_n, cv::Mat & Bx_n, cv::Mat & By_n, const bool & bright, const bool & dark, const cv::Point & gradientPoint, const cv::Vec2d &posOnLine, const cv::Vec2i &gp, const cv::Vec2d voteVector, const int & voteValue=1);

/**
Perform the specified morphological operation on input image with structure element of specified type and size
@param inputImage Input image of any type (preferrably 8-bit). The resulting image overwrites the input
@param operation Name of the morphological operation (MORPH_ERODE, MORPH_DILATE, MORPH_OPEN, MORPH_CLOSE)
@param mShape Shape of the structure element (MORPH_RECT, MORPH_CROSS, MORPH_ELLIPSE)
@param mSize Size of the structure element
@param iterations Number of iterations, how many times to perform the morphological operation
*/
void bwMorph(cv::Mat& inputImage, const int operation, const int mShape = cv::MORPH_RECT, const int mSize = 3, const int iterations = 1);
/**
Perform the specified morphological operation on input image with structure element of specified type and size
@param inputImage Input image of any type (preferrably 8-bit)
@param outputImage Output image of the same size and type as the input image
@param operation Name of the morphological operation (MORPH_ERODE, MORPH_DILATE, MORPH_OPEN, MORPH_CLOSE)
@param mShape Shape of the structure element (MORPH_RECT, MORPH_CROSS, MORPH_ELLIPSE)
@param mSize Size of the structure element
@param iterations Number of iterations, how many times to perform the morphological operation
*/
void bwMorph(const cv::Mat& inputImage, cv::Mat& outputImage, const int operation, const int mShape = cv::MORPH_RECT, const int mSize = 3, const int iterations = 1);


#endif // GFRST_H





