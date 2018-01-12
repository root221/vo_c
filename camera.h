#ifndef __CAMERA_H__
#define __CAMERA_H__
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>

struct Camera{
    double cx,cy,fx,fy;
};

struct TransformVec{
    cv::Mat rvec,tvec;
};

#endif
