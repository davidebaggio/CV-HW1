#include "orb_detector.hpp"
//test

orb_detector::orb_detector(Mat img)
{
	this->test = img;
}

double orb_detector::compute_median(vector<double> values)
{
	sort(values.begin(), values.end());
	size_t n = values.size();
	if (n % 2 == 0)
	{
		return (values[n / 2 - 1] + values[n / 2]) / 2.0;
	}
	else
	{
		return values[n / 2];
	}
}

vector<DMatch> orb_detector::get_matches(const Mat &descriptors_1, const vector<KeyPoint> &keypoints_1, const Mat &descriptors_2, const vector<KeyPoint> &keypoints_2)
{
	BFMatcher matcher(NORM_HAMMING, true);
	vector<DMatch> matches;
	matcher.match(descriptors_1, descriptors_2, matches);

	return matches;
}

void orb_detector::save_points(vector<DMatch> matches, Mat descriptors_1, vector<KeyPoint> keypoints_1, Mat descriptors_2, vector<KeyPoint> keypoints_2, string fileName, int category)
{
	vector<Point> matches_points;

	for (size_t i = 0; i < matches.size(); i++)
	{
		int idx1 = matches[i].queryIdx;
		int idx2 = matches[i].trainIdx;

		Point pt1 = keypoints_1[idx1].pt;
		Point pt2 = keypoints_2[idx2].pt;

		matches_points.push_back(pt2);
	}
	winning_file_names[category] = fileName;
	points[category] = matches_points;
}

vector<vector<Point>> orb_detector::get_points()
{
	return points;
}

void orb_detector::compute_detection()
{

	Mat copy = test.clone();

	vector<KeyPoint> keypoints;
	Mat descriptors;

	orb->detect(test, keypoints);
	orb->compute(test, keypoints, descriptors);

	for (int i = 0; i < models_path.size(); i++)
	{
		try
		{
			regex regexPattern(pattern);
			int index = 0;

			Mat src;

			int max_matches = 0;
			string bestMatchFileName;
			Mat best_descriptors;
			vector<KeyPoint> best_keypoints;
			vector<DMatch> winning_matches;
			vector<Point> medians;

			for (const auto &entry : fs::directory_iterator(models_path[i]))
			{
				if (entry.is_regular_file())
				{

					std::string fileName = entry.path().filename().string();

					if (std::regex_match(fileName, regexPattern))
					{
						std::string fullFileName = models_path[i] + "/" + entry.path().filename().string();
						src = imread(fullFileName);

						if (src.rows == 0 && src.cols == 0)
						{
							std::cerr << "Error: Could not open image file: " << fileName << std::endl;
							continue;
						}

						// medianBlur(src,src, 7);

						index++;

						Mat descriptors_1;
						vector<KeyPoint> keypoints_1;

						Mat gray_frame;
						cvtColor(src, gray_frame, COLOR_BGR2GRAY);

						orb->detect(gray_frame, keypoints_1);
						orb->compute(gray_frame, keypoints_1, descriptors_1);

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
						vector<DMatch> matches = get_matches(descriptors_1, keypoints_1, descriptors, keypoints);

						if (matches.size() == 0)
						{
							cout << "Image matches below minimum threshold: " << matches.size() << endl;
							continue;
						}

						vector<double> xs;
						vector<double> ys;

						for (size_t i = 0; i < matches.size(); i++)
						{
							int idx2 = matches[i].trainIdx;
							Point pt2 = keypoints[idx2].pt;

							xs.push_back(pt2.x);
							ys.push_back(pt2.y);
						}

						int x_median = (int)compute_median(xs);
						int y_median = (int)compute_median(ys);

						Point center = Point(x_median, y_median);
						medians.push_back(center);

						if (matches.size() > max_matches)
						{
							winning_matches = matches;
							max_matches = matches.size();
							bestMatchFileName = fullFileName;
							best_descriptors = descriptors_1;
							best_keypoints = keypoints_1;
						}
	
						if (winning_matches.size() == 0)
						{
							cout << "No matches found for type " << i << " PALLE " << entry << endl;
							continue;
						}
					}
				}
				else
				{
					continue;
				}
			}

			vector<double> xs;
			vector<double> ys;
			for (int i = 0; i < medians.size(); i++)
			{
				xs.push_back(medians[i].x);
				ys.push_back(medians[i].y);
			}

			intermediate_medians[i] = medians;

			int final_median_x = (int)compute_median(xs);
			int final_median_y = (int)compute_median(ys);

			Point final_center = Point(final_median_x, final_median_y);
			medians_per_object[i] = final_center;

			if (best_keypoints.empty())
			{
				cout << "No matches found for type " << i << endl;
				continue;
			}
			save_points(winning_matches, best_descriptors, best_keypoints, descriptors, keypoints, bestMatchFileName, i);
			cout << "Best match for type " << i << " is: " << bestMatchFileName << " with " << max_matches << " matches." << endl;
			cout << "-----------------------------" << endl;
		}
		catch (const std::exception &e)
		{
			std::cerr << "Error: " << e.what() << std::endl;
		}
	}
}

void orb_detector::display_points()
{

	vector<Mat> mat_matches = vector<Mat>(3);
	mat_matches[0] = test.clone(); // sugar
	mat_matches[1] = test.clone(); // mustard
	mat_matches[2] = test.clone(); // drill

	for (size_t i = 0; i < points.size(); i++)
	{
		if (points[i].size() == 0)
		{
			cout << "No points found for type " << i << endl;
			continue;
		}
		for (int j = 0; j < points[i].size(); j++)
		{
			Point pt = points[i][j];
			circle(mat_matches[i], pt, 5, Scalar(0, 255, 0), 2);
		}
	}

	for (int i = 0; i < medians_per_object.size(); i++)
	{
		if (medians_per_object[i].x == 0 && medians_per_object[i].y == 0)
		{
			cout << "No final points found for type " << i << endl;
			continue;
		}
		Point pt = medians_per_object[i];
		circle(mat_matches[i], pt, 5, Scalar(255, 0, 0), 2);
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
			cout << "No matches found for type " << i << endl;
			continue;
		}
		imshow(category + " matches", mat_matches[i]);
	}
	waitKey(0);
}

/*int main()
{

	Mat test = imread("../object_detection_dataset/035_power_drill/test_images/35_0010_001462-color.jpg");

	if (test.empty())
	{
		std::cerr << "Error: Could not open image file." << std::endl;
		return -1;
	}

	orb_detector detector(test);
	detector.compute_detection();

	detector.display_points();
	return 0;
} */
