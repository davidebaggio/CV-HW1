// Created by: Zoren Martinez mat. 2123873
#ifndef ORB_DETECTOR_HPP
#define ORB_DETECTOR_HPP

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
namespace fs = filesystem;

class orb_detector
{
private:
	
	// Test image
	Mat test;

	// Different directories for the models
	vector<string> models_path = {
		"data/004_sugar_box/models",
		"data/006_mustard_bottle/models",
		"data/035_power_drill/models"};
	
	// Regex pattern to match the model images
	string pattern = R"(view.*color\.png)";

	// Regex pattern to match the model masks
	string pattern_mask = R"(view.*mask\.png)";
	Ptr<ORB> orb = ORB::create();

	// Vector of points for each model
	// 0: sugar, 1: mustard, 2: drill
	vector<vector<Point>> points = vector<vector<Point>>(3);

	// Vector descriptors for each model
	// 0: sugar, 1: mustard, 2: drill
	vector<vector<Mat>> model_descriptors = vector<vector<Mat>>(3);

	
	// Helper functions
	double compute_median(vector<double> values);
	vector<DMatch> get_matches(const Mat &model_descriptors, const Mat &test_descriptors);
	void save_points(vector<DMatch> matches, vector<KeyPoint> test_keypoints, int category);

public:


	orb_detector();
	vector<vector<Point>> get_points();
	vector<vector<Point>> get_points(float perc);
	void compute_detection(Mat img);
	void display_points();
	void display_points(float perc);
};

#endif // ORB_DETECTOR_HPP