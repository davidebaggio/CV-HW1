#ifndef HAAR_DETECTOR_HPP
#define HAAR_DETECTOR_HPP

#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include "detection.hpp"

using namespace std;
using namespace cv;

class haar_detector
{
private:
	Mat test;

	CascadeClassifier cascade_sugar;
	CascadeClassifier cascade_mustard;
	CascadeClassifier cascade_drill;

	vector<vector<Point>> points = vector<vector<Point>>(3);

public:
	haar_detector();
	void set_points(vector<Point> points, size_t index);
	void compute_detection(Mat img);
	vector<vector<Point>> get_points();
	void display_points();
};

#endif // HAAR_DETECTOR_HPP