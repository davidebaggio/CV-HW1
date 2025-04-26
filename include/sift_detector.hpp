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
	Mat img_test;

	Ptr<SIFT> sift;

	string pattern = R"(view.*color\.png)";
	vector<string> models_path = {
		"data/004_sugar_box/models",
		"data/006_mustard_bottle/models",
		"data/035_power_drill/models"};

	vector<vector<Point>> points = vector<vector<cv::Point>>(3);
	vector<string> winning_file_names = vector<string>(3);

	void optimize_image(Mat &src, bool isImg_test);
	double compute_median(vector<double> values);
	void save_points(vector<cv::DMatch> matches, Mat descriptors_1, vector<KeyPoint> keypoints_1, Mat descriptors_2, vector<KeyPoint> keypoints_2, string fileName, int category);
	vector<DMatch> get_matches(const Mat &model_desc, const Mat &img_desc);

public:
	sift_detector(Mat img);
	vector<vector<Point>> get_points();
	void compute_detection();
	void display_points();
};

#endif // SIFT_DETECTOR_HPP