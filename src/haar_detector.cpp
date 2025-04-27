// created by Davide Baggio 2122547

#include "haar_detector.hpp"

haar_detector::haar_detector()
{
	// cascade file path
	string sugar_cascade_path = base + sugar + cascade;
	string mustard_cascade_path = base + mustard + cascade;
	string drill_cascade_path = base + drill + cascade;

	if (!cascade_sugar.load(sugar_cascade_path) || !cascade_mustard.load(mustard_cascade_path) || !cascade_drill.load(drill_cascade_path))
	{
		cout << "[ERROR]: loading cascade" << endl;
		exit(1);
	}
}

void haar_detector::compute_detection(Mat img)
{
	this->test = img.clone();
	Mat gray;
	cvtColor(img, gray, COLOR_BGR2GRAY);
	equalizeHist(gray, gray);

	vector<Rect> obj_s, obj_m, obj_d;
	cascade_sugar.detectMultiScale(gray, obj_s, 1.1, 2, 0 | cv::CASCADE_SCALE_IMAGE);
	cascade_mustard.detectMultiScale(gray, obj_m, 1.1, 2, 0 | cv::CASCADE_SCALE_IMAGE);
	cascade_drill.detectMultiScale(gray, obj_d, 1.1, 2, 0 | cv::CASCADE_SCALE_IMAGE);
	vector<Point> pp_s, pp_m, pp_d;
	for (int i = 0; i < obj_s.size(); i++)
	{
		pp_s.push_back(Point(obj_s[i].x + obj_s[i].width / 2, obj_s[i].y + obj_s[i].height / 2));
	}

	for (int i = 0; i < obj_m.size(); i++)
	{
		pp_m.push_back(Point(obj_m[i].x + obj_m[i].width / 2, obj_m[i].y + obj_m[i].height / 2));
	}

	for (int i = 0; i < obj_d.size(); i++)
	{
		pp_d.push_back(Point(obj_d[i].x + obj_d[i].width / 2, obj_d[i].y + obj_d[i].height / 2));
	}
	points[0] = pp_s;
	points[1] = pp_m;
	points[2] = pp_d;

	cout << "Best matches found from HAAR detector\n";
}

vector<vector<Point>> haar_detector::get_points()
{
	return points;
}

void haar_detector::display_points()
{

	vector<Mat> mat_matches = vector<Mat>(3);
	mat_matches[0] = test.clone(); // sugar
	mat_matches[1] = test.clone(); // mustard
	mat_matches[2] = test.clone(); // drill

	for (size_t i = 0; i < points.size(); i++)
	{
		if (points[i].size() == 0)
		{
			cout << "HAAR: No points found for type " << i << endl;
			continue;
		}
		for (int j = 0; j < points[i].size(); j++)
		{
			Point pt = points[i][j];
			circle(mat_matches[i], pt, 5, Scalar(0, 255, 0), 2);
		}
	}

	for (int i = 0; i < mat_matches.size(); i++)
	{

		string category;
		if (i == 0)
		{
			category = "sugar_box";
		}
		else if (i == 1)
		{
			category = "mustard_bottle";
		}
		else if (i == 2)
		{
			category = "power_drill";
		}
		if (mat_matches[i].empty())
		{
			cout << "HAAR: No matches found for type " << i << endl;
			continue;
		}
		imshow(category + " matches", mat_matches[i]);
	}
	waitKey(0);
}

void haar_detector::set_points(vector<Point> points, size_t index)
{
	if (index < 0 || index > 3)
	{
		cout << "[ERROR]: invalid index for insertion\n";
		return;
	}
	this->points[index] = points;
}