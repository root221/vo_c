#include <opencv2/core/core.hpp>
using namespace cv;
#include "camera.h"
std::vector<KeyPoint>extracKeypoints(Mat &img);
Mat computeDescriptor(Mat & img,std::vector<KeyPoint> & key_point);
TransformVec motionEstimate(Frame frame1, Frame frame2, Camera camera);
