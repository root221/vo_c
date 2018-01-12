#include <opencv2/highgui/highgui.hpp>
using namespace cv;
struct Frame{
    Mat rgb, depth;
    std::vector<KeyPoint> keypoints;
    Mat descriptors; 
};
