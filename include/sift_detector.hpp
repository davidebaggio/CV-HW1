#ifndef SIFT_DETECTOR_HPP
#define SIFT_DETECTOR_HPP

#include <iostream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include "opencv2/features2d.hpp"
#include <filesystem>
#include <fstream>
#include <regex>

using namespace cv;
using namespace std;
using namespace filesystem;

class sift_detector
{
	private:

		// Test image
		Mat img_test;

		//SIFT Detector
		Ptr<SIFT> sift;

		// Regex pattern to match the model images
		string pattern = R"(view.*color\.png)";

		// Different directories for the models
		vector<string> models_path = {
			"data/004_sugar_box/models",
			"data/006_mustard_bottle/models",
			"data/035_power_drill/models"};


		// Vector of points for each model
		// 0: sugar, 1: mustard, 2: drill
		vector<vector<Point>> points = vector<vector<cv::Point>>(3);

		// Vector descriptors for each model
		// 0: sugar, 1: mustard, 2: drill
		vector<vector<Mat>> model_descriptors = vector<vector<Mat>>(3);
		
		// Helper functions
		void get_model_descriptors();
		void optimize_image(Mat &src, bool isImg_test);
		void save_points(vector<cv::DMatch> &matches, vector<KeyPoint> &img_kpt, int category);
		vector<DMatch> get_matches(const Mat &model_desc, const Mat &img_desc);

	public:
		sift_detector();
		vector<vector<Point>> get_points();
		vector<vector<Point>> get_points(float perc);
		void compute_detection(Mat img_test);
		void display_points();
		void display_points(float perc);
};

#endif // SIFT_DETECTOR_HPP