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
	Mat test;
	vector<string> models_path = {
		"data/004_sugar_box/models",
		"data/006_mustard_bottle/models",
		"data/035_power_drill/models"};
	string pattern = R"(view.*color\.png)";
	string pattern_mask = R"(view.*mask\.png)";
	Ptr<ORB> orb = ORB::create();

	vector<vector<Point>> points = vector<vector<Point>>(3);

	// vector<string> winning_file_names = vector<string>(3);

	// vector<Point> medians_per_object = vector<Point>(3);

	// vector<vector<Point>> intermediate_medians = vector<vector<Point>>(3);

	vector<vector<Mat>> model_descriptors = vector<vector<Mat>>(3);

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