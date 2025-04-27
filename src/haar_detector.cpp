#include "haar_detector.hpp"

haar_detector::haar_detector(string cascade_path)
{
	if (!cascade.load(cascade_path))
	{
		cout << "[ERROR]: loading cascade" << endl;
		exit(1);
	}
}

vector<Point> haar_detector::detect(Mat img)
{
	Mat gray;
	cvtColor(img, gray, COLOR_BGR2GRAY);

	vector<Rect> obj;
	cascade.detectMultiScale(gray, obj, 1.1, 2, 0 | cv::CASCADE_SCALE_IMAGE);
	vector<Point> points;
	for (int i = 0; i < obj.size(); i++)
	{
		points.push_back(Point(obj[i].x + obj[i].width / 2, obj[i].y + obj[i].height / 2));
	}

	return points;
}