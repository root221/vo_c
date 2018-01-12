#include"frame.h"
#include <iostream>
#include <fstream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <algorithm> 
#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include <g2o/types/sba/types_six_dof_expmap.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include "camera.h"
#include "visualOdometry.h"
#include<opencv2/viz.hpp>
using namespace cv;
using namespace std;
double max_norm = 10.0;
double normTransform(cv::Mat rvec, cv::Mat tvec){
    return fabs(min(cv::norm(rvec), 2*M_PI-cv::norm(rvec)))+ fabs(cv::norm(tvec));
}

int main(){
    Camera camera;
    camera.fx = 520.9;
    camera.fy = 521.0;
    camera.cx = 325.1;
    camera.cy = 249.7;
   
   
   
    Frame old_frame; 
    ifstream fin ("associate.txt");
    vector<string> rgb_files, depth_files;
    vector<double> rgb_times, depth_times;

    string rgb_time, depth_time, rgb_file, depth_file;   

    while(!fin.eof()){
        string rgb_time, rgb_file, depth_time, depth_file;
        fin>>rgb_time>>rgb_file>>depth_time>>depth_file;
        rgb_times.push_back(atof(rgb_time.c_str()));
        rgb_files.push_back("data_1/"+rgb_file);
        depth_times.push_back(atof(depth_time.c_str()));
        depth_files.push_back("data_1/"+depth_file);
    }

    //visualization
    viz::Viz3d vis("Visaul Odometry");
    viz::WCoordinateSystem world_coor ( 1.0 ), camera_coor ( 0.5 ); 
    cv::Point3d cam_pos ( 0, -1.0, -1.0 ), cam_focal_point ( 0,0,0 ), cam_y_dir ( 0,1,0 );
    cv::Affine3d cam_pose = cv::viz::makeCameraPose ( cam_pos, cam_focal_point, cam_y_dir );
    vis.setViewerPose ( cam_pose ); 
    vis.showWidget("World", world_coor);
    vis.showWidget("Camera", camera_coor);
    rgb_files.clear();
    depth_files.clear();
    rgb_files.push_back("1.png");    
    rgb_files.push_back("2.png");    
    depth_files.push_back("1_depth.png");    
    depth_files.push_back("2_depth.png");    
    for(int i=0;i<rgb_files.size();i++){
        Mat rgb = imread(rgb_files[i]);
        Mat depth = imread(depth_files[i],-1);
        Frame new_frame;
        new_frame.rgb = rgb;
        new_frame.depth = depth;
        new_frame.keypoints= extracKeypoints(rgb);
        new_frame.descriptors = computeDescriptor(rgb,new_frame.keypoints);
        
        if(i==0){
            old_frame = new_frame; 
            continue;
        }
        TransformVec transform = motionEstimate(old_frame, new_frame, camera);
        // check motion
        double norm = normTransform(transform.rvec,transform.tvec);
        if(norm >= max_norm){
            cout << norm <<endl;
            continue;
        }
        Mat R;
        cv::Rodrigues(transform.rvec,R);        
        cout << R << endl;
        cout << transform.tvec << endl;
        Affine3d M(
            Affine3d::Mat3(
                R.at<double>(0,0),R.at<double>(0,1),R.at<double>(0,2),
                R.at<double>(1,0),R.at<double>(1,1),R.at<double>(1,2),
                R.at<double>(2,0),R.at<double>(2,1),R.at<double>(2,2)
            ),
            Affine3d::Vec3(
                transform.tvec.at<double>(0,0),
                transform.tvec.at<double>(0,1),
                transform.tvec.at<double>(0,2)
            )        
        );
        
        old_frame = new_frame; 
//        cv::imshow("image",rgb);
  //      cv::waitKey(1); 
        vis.setWidgetPose("Camera",M);
        vis.spinOnce(1,false);        
    }    

    

}
