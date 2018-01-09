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

using namespace std;
using namespace cv;
ofstream myfile;
Mat old_rgb,old_depth;
bool first_img = true;
Eigen::Matrix<double,4,1> position;
Point2f pixel2cam(Point2f p,Mat K){
	return Point2f(
		(p.x - K.at<float>(0,2)) / K.at<float>(0,0),
		(p.y - K.at<float>(1,2)) / K.at<float>(1,1)
	);
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
    //cout<<"T="<<endl<<Eigen::Isometry3d ( pose->estimate() ).matrix() <<endl;
    position = Eigen::Isometry3d ( pose->estimate() ).matrix() * position;
    //myfile << position.transpose() <<endl;
    cout << position.transpose() << endl;
}

void visual_odometry(Mat img_1, Mat img_2, Mat d1, Mat d2){
	//Mat img_1,img_2;
	Mat descriptors_1, descriptors_2;
	std::vector<KeyPoint> key_point_1,key_point_2;
	Ptr<FeatureDetector> detector = FeatureDetector::create("ORB");
	Ptr<DescriptorExtractor> descriptor = DescriptorExtractor::create ( "ORB" );
	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");

	//img_1 = imread ( "1.png", CV_LOAD_IMAGE_COLOR );
    //img_2 = imread ( "2.png", CV_LOAD_IMAGE_COLOR );


	// find the keypoints with ORB
	detector->detect(img_1,key_point_1);
	detector->detect(img_2,key_point_2);
	
	//compute the descriptors with ORB
	descriptor->compute(img_1,key_point_1,descriptors_1);
	descriptor->compute(img_2,key_point_2,descriptors_2);

	std::vector<DMatch> matches;
	matcher->match(descriptors_1,descriptors_2,matches);

	//cout << descriptors_1.rows << endl;
	//cout << matches.size();
	
	double min_dist = 10000, max_dist = 0;
	for(int i=0;i<matches.size();i++){
		double dist = matches[i].distance;
		if ( dist < min_dist ) min_dist = dist;
        if ( dist > max_dist ) max_dist = dist;
	}
	//cout << "min_dist: " << min_dist << endl; 
	//cout << "max_dist: " << max_dist << endl;

	std::vector<DMatch> good_matches;
	for(int i=0;i<matches.size();i++){
		if(matches[i].distance <= std::max ( 2*min_dist, 30.0 )){
			good_matches.push_back(matches[i]);
		}
	}
	//cout << "good matches: " << good_matches.size() << endl;


	//Mat d1 = imread("1_depth.png", CV_LOAD_IMAGE_UNCHANGED);

	//Mat K = ( Mat_<float> ( 3,3 ) << 613.8923950195312, 0.0, 324.427978515625, 0.0, 619.4680786132812, 220.7501220703125, 0.0, 0.0, 1.0);
	Mat K = ( Mat_<float> ( 3,3 ) << 525.0, 0.0, 319.5, 0.0, 525.0, 239.5, 0.0, 0.0, 1.0);
    //Mat K = ( Mat_<float> ( 3,3 ) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1 );
    
	std::vector<Point3f> pts_3d;
	std::vector<Point2f> pts_2d;

	for(int i=0;i<good_matches.size();i++){

		int x = int(key_point_1[good_matches[i].queryIdx].pt.x);
		int y = int(key_point_1[good_matches[i].queryIdx].pt.y);
	    //float d = d1.at<short int>(cv::Point(x,y)) / 1000.0;
        
		ushort d = d1.at<unsigned short>(y,x);		
		if ( d == 0 )   // bad depth
            continue;
		
		float dd = d/1000.0;
		
		Point2f p = pixel2cam (key_point_1[good_matches[i].queryIdx].pt, K );
		//cout << p << endl;	
		pts_3d.push_back(Point3f(p.x * dd,p.y * dd,dd));
		pts_2d.push_back(key_point_2[good_matches[i].trainIdx].pt);
	}
    cout << pts_3d.size() <<endl;
    if(pts_3d.size() < 4)
        return;
	Mat r, t, R;
	solvePnP(pts_3d,pts_2d,K,Mat(),r,t,false);//,cv::SOLVEPNP_EPNP);
	cv::Rodrigues ( r, R );
    cout << "R=" << endl;
	cout << R << endl;
    cout << "t=" <<endl;
    cout << t << endl;
    bundleAdjustment(pts_3d,pts_2d,K,R,t);	
    
}

void callback(const Mat rgb_image, const Mat depth_image){
    if(first_img)
        first_img = false; 
    else{
        visual_odometry(old_rgb, rgb_image, old_depth, depth_image);
    }
    old_rgb = rgb_image;
    old_depth = depth_image;
}

int main(int argc, char ** argv){

    //position << 1.3563,0.6305,1.6380,1;

    //position << 1.3563,0.6305,1.6380,1;
    position << 0,0,0,1;
    ifstream fin ("associate.txt");
    vector<string> rgb_files, depth_files;
    vector<double> rgb_times, depth_times;

    string rgb_time, depth_time, rgb_file, depth_file;   

    while(!fin.eof()){
        string rgb_time, rgb_file, depth_time, depth_file;
        fin>>rgb_time>>rgb_file>>depth_time>>depth_file;
        rgb_times.push_back(atof(rgb_time.c_str()));
        rgb_files.push_back("data/"+rgb_file);
        depth_times.push_back(atof(depth_time.c_str()));
        depth_files.push_back("data/"+depth_file);
    }
    /*
    for(int i=0;i<rgb_files.size();i++){
        Mat color = cv::imread(rgb_files[i]);
        Mat depth = cv::imread(depth_files[i]);
        //Mat depth = imread(depth_files[i], CV_LOAD_IMAGE_UNCHANGED);
        callback(color,depth);
        
        
        
    }
    */
    Mat color = cv::imread("1.png");
    Mat depth = cv::imread("1_depth.png");
    Mat d1 = imread("1_depth.png", CV_LOAD_IMAGE_UNCHANGED);
    callback(color,d1);
    color = cv::imread("2.png");
    depth = cv::imread("2_depth.png");
    Mat d2 = imread("2_depth.png", CV_LOAD_IMAGE_UNCHANGED);
    callback(color,d2);
	
    return 0;
}

