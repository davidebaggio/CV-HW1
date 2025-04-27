#ifndef HAAR_DETECTOR_HPP
#define HAAR_DETECTOR_HPP

#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

class haar_detector
{
private:
	CascadeClassifier cascade;

public:
	haar_detector(string cascade_path);
	vector<Point> detect(Mat img);
};

#endif // HAAR_DETECTOR_HPP