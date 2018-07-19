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
	Applies Fast radial symmetry transform to image
	Check paper Loy, G., & Zelinsky, A. (2002). A fast radial symmetry transform for 
	detecting points of interest. Computer Vision, ECCV 2002.
	
	@param inputImage The input grayscale image (8-bit)
	@param outputImage The output image containing the results of FRST
	@param radii Gaussian kernel radius
	@param alpha Strictness of radial symmetry 
	@param stdFactor Standard deviation factor
	@param mode Transform mode ('bright', 'dark' or 'both')
*/
void frst2d(const cv::Mat& inputImage, cv::Mat& outputImage, const int& radii, const double& alpha, const double& stdFactor, const int& mode, const int sideNumber);


inline void voteOnLine(cv::Mat &O_n, cv::Mat & Bx_n, cv::Mat & By_n, const cv::Vec2i &gp, const double &gNorm, const double &nAngle,
                       const cv::Point &gradientPoint, const bool &bright, const bool &dark, const int & w, const double &radii);

inline void voteAtPos(cv::Mat & O_n, cv::Mat & Bx_n, cv::Mat & By_n, const bool & bright, const bool & dark, const cv::Point & gradientPoint,
                      const int & m, const cv::Vec2d &lineSuport, const cv::Vec2i &gp, const double & gnorm, const double &nAngle, const int & voteValue=1);

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








