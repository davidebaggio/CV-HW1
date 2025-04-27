#include "orb_detector.hpp"

orb_detector::orb_detector()
{
	for (int i = 0; i < models_path.size(); i++)
	{
		try
		{
			regex regexPattern(pattern);
			int index = 0;

			Mat src;
			Mat mask;

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

					string file_mask = fileName.substr(fileName.find_last_of("/") + 1);
					file_mask = file_mask.substr(0, file_mask.find_last_of("_")) + "_mask.png";

					if (std::regex_match(fileName, regexPattern))
					{
						std::string fullFileName = models_path[i] + "/" + entry.path().filename().string();
						src = imread(fullFileName);
						mask = imread(models_path[i] + "/" + file_mask, IMREAD_GRAYSCALE);

						if (src.empty() || mask.empty())
						{
							std::cerr << "[ERROR]: Could not open image file: " << fileName << std::endl;
							continue;
						}

						// medianBlur(src,src, 7);

						index++;

						Mat descriptors_1;
						vector<KeyPoint> keypoints_1;

						Mat gray_frame;
						cvtColor(src, gray_frame, COLOR_BGR2GRAY);

						orb->detect(gray_frame, keypoints_1, mask);
						orb->compute(gray_frame, keypoints_1, descriptors_1);

						model_descriptors[i].push_back(descriptors_1);
					}
				}
				else
				{
					continue;
				}
			}
		}
		catch (const std::exception &e)
		{
			std::cerr << "Error: " << e.what() << std::endl;
		}
	}
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

vector<DMatch> orb_detector::get_matches(const Mat &model_descriptors, const Mat &test_descriptors)
{
	BFMatcher matcher(NORM_HAMMING, true);
	vector<DMatch> matches;
	matcher.match(model_descriptors, test_descriptors, matches);

	return matches;
}

void orb_detector::save_points(vector<DMatch> matches, vector<KeyPoint> test_keypoints, int category)
{
	vector<Point> matches_points;

	for (size_t i = 0; i < matches.size(); i++)
	{
		int idx1 = matches[i].queryIdx;
		int idx2 = matches[i].trainIdx;
		Point pt2 = test_keypoints[idx2].pt;

		matches_points.push_back(pt2);
	}
	points[category] = matches_points;
}

vector<vector<Point>> orb_detector::get_points()
{
	return points;
}

vector<vector<Point>> orb_detector::get_points(float perc)
{
	vector<vector<Point>> best_points;
	for (size_t j = 0; j < points.size(); j++)
	{
		cout << points[j].size() << endl;
		int max_index = points[j].size() * perc;
		vector<Point> best_single_vector;
		for (int i = 0; i < max_index; i++)
		{
			best_single_vector.push_back(points[j][i]);
		}
		cout << best_single_vector.size() << endl;
		best_points.push_back(best_single_vector);
	}
	return best_points;
}

void orb_detector::compute_detection(Mat img)
{
	this->test = img;

	Mat copy = test.clone();

	vector<KeyPoint> test_keypoints;
	Mat test_descriptors;

	orb->detect(test, test_keypoints);
	orb->compute(test, test_keypoints, test_descriptors);

	if (test_descriptors.empty())
	{
		cout << "ORB: No descriptors found for tes image " << endl;
		return;
	}

	for (int i = 0; i < models_path.size(); i++)
	{
		int max_matches = 0;
		string bestMatchFileName;
		Mat best_descriptors;
		vector<DMatch> winning_matches;
		vector<Point> medians;

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

		for (int j = 0; j < model_descriptors[i].size(); j++)
		{
			vector<DMatch> matches = get_matches(model_descriptors[i][j], test_descriptors);

			if (matches.empty())
			{
				cout << "ORB: No matches found for type " << i << endl;
				continue;
			}

			if (matches.size() == 0)
			{
				cout << "ORB: Image matches below minimum threshold: " << matches.size() << endl;
				continue;
			}

			if (matches.size() > max_matches)
			{
				winning_matches = matches;
				max_matches = matches.size();
				best_descriptors = model_descriptors[i][j];
			}
		}

		// cout << "ORB: Best match for " << category << ": " << max_matches << endl;
		if (max_matches == 0)
		{
			cout << "ORB: No matches found for type " << i << endl;
			continue;
		}
		sort(winning_matches.begin(), winning_matches.end(), [](const DMatch &a, const DMatch &b)
			 { return a.distance < b.distance; });
		save_points(winning_matches, test_keypoints, i);
	}
	cout << "Best matches found from ORB detector\n";
}

void orb_detector::display_points()
{
	display_points(1.0);
}

void orb_detector::display_points(float perc)
{

	vector<Mat> mat_matches = vector<Mat>(3);
	mat_matches[0] = test.clone(); // sugar
	mat_matches[1] = test.clone(); // mustard
	mat_matches[2] = test.clone(); // drill

	vector<vector<Point>> pp = get_points(perc);

	for (size_t i = 0; i < pp.size(); i++)
	{
		if (points[i].size() == 0)
		{
			cout << "ORB: No points found for type " << i << endl;
			continue;
		}
		for (int j = 0; j < pp[i].size(); j++)
		{
			Point pt = pp[i][j];
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
			cout << "ORB: No matches found for type " << i << endl;
			continue;
		}
		imshow(category + " matches", mat_matches[i]);
	}
	waitKey(0);
}
