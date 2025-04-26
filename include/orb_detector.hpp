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
	Ptr<ORB> orb = ORB::create();

	vector<vector<Point>> points = vector<vector<Point>>(3);

	vector<string> winning_file_names = vector<string>(3);

	vector<Point> medians_per_object = vector<Point>(3);

	vector<vector<Point>> intermediate_medians = vector<vector<Point>>(3);

	double compute_median(vector<double> values);
	vector<DMatch> get_matches(const Mat &descriptors_1, const vector<KeyPoint> &keypoints_1, const Mat &descriptors_2, const vector<KeyPoint> &keypoints_2);
	void save_points(vector<DMatch> matches, Mat descriptors_1, vector<KeyPoint> keypoints_1, Mat descriptors_2, vector<KeyPoint> keypoints_2, string fileName, int category);

public:
	orb_detector(Mat img);
	vector<vector<Point>> get_points();
	void compute_detection();
	void display_points();
};

#endif // ORB_DETECTOR_HPP