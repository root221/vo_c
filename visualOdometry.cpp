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
using namespace cv;
using namespace std;

Ptr<FeatureDetector> detector = ORB::create();
Ptr<DescriptorExtractor> descriptor = ORB::create();
Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");

Point2f pixelToCam(Point2f p,Mat K){
	return Point2f(
		(p.x - K.at<float>(0,2)) / K.at<float>(0,0),
		(p.y - K.at<float>(1,2)) / K.at<float>(1,1)
	);
}


std::vector<KeyPoint>extracKeypoints(Mat &img){
    vector<KeyPoint> key_points;
    detector->detect(img,key_points);
    return key_points;
}

Mat computeDescriptor(Mat & img,vector<KeyPoint> & key_point){
    Mat descriptors;
    descriptor->compute(img,key_point,descriptors);
    return descriptors;
}

TransformVec motionEstimate(Frame frame1, Frame frame2, Camera camera){
	
    std::vector<DMatch> matches;
	matcher->match(frame1.descriptors,frame2.descriptors,matches);
	
    double min_dist = 10000, max_dist = 0;
	for(int i=0;i<matches.size();i++){
		double dist = matches[i].distance;
		if ( dist < min_dist ) min_dist = dist;
        if ( dist > max_dist ) max_dist = dist;
	}
	cout << "min_dist: " << min_dist << endl; 
	cout << "max_dist: " << max_dist << endl;

	std::vector<DMatch> good_matches;
	for(int i=0;i<matches.size();i++){
		if(matches[i].distance <= std::max ( 2*min_dist, 30.0 )){
			good_matches.push_back(matches[i]);
		}
	}
	/*
    Mat img; 
    drawMatches(frame1.rgb,frame1.keypoints,frame2.rgb,frame2.keypoints,matches,img);
    cv::imshow("image",img);
    cv::waitKey(0); 
    */  
	std::vector<Point3f> pts_3d;
	std::vector<Point2f> pts_2d;

    Mat K = ( Mat_<float> ( 3,3 ) << camera.fx, 0.0, camera.cx, 0.0, camera.fy, camera.fx, 0.0, 0.0, 1.0);
	
    for(int i=0;i<good_matches.size();i++){

		int x = int(frame1.keypoints[good_matches[i].queryIdx].pt.x);
		int y = int(frame1.keypoints[good_matches[i].queryIdx].pt.y);
        
		ushort d = frame1.depth.ptr<ushort>(y)[x];
        if ( d == 0 )   // bad depth
            continue;
		
		Point2f p = pixelToCam (frame1.keypoints[good_matches[i].queryIdx].pt, K );
		pts_3d.push_back(Point3f(p.x * d / 1000.0,p.y * d / 1000.0, d/1000.0));
		
        pts_2d.push_back(frame2.keypoints[good_matches[i].trainIdx].pt);
	}
    
    
    // Solving pnp
    
    Mat t,r;
	solvePnP(pts_3d,pts_2d,K,Mat(),r,t,false);
    
    TransformVec transform;
    transform.rvec = r;
    transform.tvec = t;
    return transform;
}

void bundleAdjustment(vector<Point3f> points_3d, vector<Point2f> points_2d, Mat K, Mat R, Mat t)
{

	typedef g2o::BlockSolver< g2o::BlockSolverTraits<6,3> > Block; 
    	std::unique_ptr<Block::LinearSolverType> linearSolver;
	linearSolver = g2o::make_unique<g2o::LinearSolverCSparse<Block::PoseMatrixType>>();

	g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(
	g2o::make_unique<Block>(std::move(linearSolver))
	);
	
	g2o::SparseOptimizer optimizer;
	optimizer.setAlgorithm ( solver );  
	
	g2o::VertexSE3Expmap * pose = new g2o::VertexSE3Expmap();
	
	Eigen::Matrix3d R_eigen;
	R_eigen << R.at<double>(0,0), R.at<double>(0,1), R.at<double>(0,2),
			   R.at<double>(1,0), R.at<double>(1,1), R.at<double>(1,2),
			   R.at<double>(2,0), R.at<double>(2,1), R.at<double>(2,2);
	pose->setId(0);
	pose->setEstimate(
		g2o::SE3Quat(
			R_eigen,
			Eigen::Vector3d(t.at<double>(0,0),t.at<double>(1,0),t.at<double>(2,0))
		)
	);
	optimizer.addVertex(pose);

	int index = 1;
	for(int i=0;i<points_3d.size();i++){
		g2o::VertexSBAPointXYZ * point = new g2o::VertexSBAPointXYZ();
		point->setId(index);
		point->setEstimate(
			Eigen::Vector3d(points_3d[i].x, points_3d[i].y, points_3d[i].z)
		);
		point->setMarginalized(true);
		optimizer.addVertex(point);
		index++;
	}

	//parameter CameraParameters(focal_length,  principle_point, baseline)
	g2o::CameraParameters* camera = new g2o::CameraParameters(
		K.at<double>(0,0), Eigen::Vector2d(K.at<double>(0,2),K.at<double>(1,2)), 
		0		
	);
	camera->setId(0);
	optimizer.addParameter(camera);

	//edges
	index = 1;

	for(int i=0; i<points_2d.size();i++){
		Point2d p = points_2d[i];
		g2o::EdgeProjectXYZ2UV * edge = new g2o::EdgeProjectXYZ2UV();
		edge->setId(index);
		edge->setVertex(0,optimizer.vertex(index));
		edge->setVertex(1,pose);
		edge->setMeasurement(Eigen::Vector2d(p.x,p.y));
		edge->setParameterId(0,0);
		edge->setInformation(Eigen::Matrix2d::Identity());
		optimizer.addEdge(edge);
		index++;
	}
	
	//optimizer.setVerbose(true);
	optimizer.initializeOptimization();
	optimizer.optimize(100);
}
