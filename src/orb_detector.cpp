// Created by: Zoren Martinez mat. 2123873
#include "orb_detector.hpp"

// Constructor
// Initializes the ORB detector and reads the model images from the dataset folders
orb_detector::orb_detector()
{
	
	// Reads every model of the objects from the the dataset folders
	for (int i = 0; i < models_path.size(); i++)
	{
		try
		{
			regex regex_pattern(pattern);

			Mat src;
			Mat mask;

			int max_matches = 0;
			string best_match_fileName;
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

					
					// Check if the file name matches the regex pattern
					if (std::regex_match(fileName, regex_pattern))
					{
						std::string full_file_name = models_path[i] + "/" + entry.path().filename().string();
						src = imread(full_file_name);
						mask = imread(models_path[i] + "/" + file_mask, IMREAD_GRAYSCALE);

						if (src.empty() || mask.empty())
						{
							std::cerr << "[ERROR]: Could not open image file: " << fileName << std::endl;
							continue;
						}


						// Compute all the descriptors for the each model images of the dataset
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
			std::cerr << "[ERROR]: " << e.what() << std::endl;
		}
	}
}

// Helper function to compute the median of a vector of doubles
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

// Helper function to get matches between an image model descriptors and image test descriptors
vector<DMatch> orb_detector::get_matches(const Mat &model_descriptors, const Mat &test_descriptors)
{
	BFMatcher matcher(NORM_HAMMING, true);
	vector<DMatch> matches;
	matcher.match(model_descriptors, test_descriptors, matches);

	return matches;
}

// Helper function to save the points of the best matches
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

// Returns the points
vector<vector<Point>> orb_detector::get_points()
{
	return points;
}

// Returns the points with a percentage of the best matches
vector<vector<Point>> orb_detector::get_points(float perc)
{

	if (perc > 1.0 || perc < 0.0)
	{
		std::cerr << "[ERROR]: Percentage must be between 0 and 1" << endl;
		return points;
	}

	vector<vector<Point>> best_points;
	for (size_t j = 0; j < points.size(); j++)
	{
		int max_index = points[j].size() * perc;
		vector<Point> best_single_vector;
		for (int i = 0; i < max_index; i++)
		{
			best_single_vector.push_back(points[j][i]);
		}
		best_points.push_back(best_single_vector);
	}
	return best_points;
}

// Computes the detection between the model descriptors and the test image descriptors
// The test image is passed as a parameter
void orb_detector::compute_detection(Mat img)
{
	this->test = img;

	Mat copy = test.clone();

	vector<KeyPoint> test_keypoints;
	Mat test_descriptors;

	// Compute the descriptors for the test image
	orb->detect(test, test_keypoints);
	orb->compute(test, test_keypoints, test_descriptors);

	if (test_descriptors.empty())
	{
		cout << "ORB: No descriptors found for tes image " << endl;
		return;
	}

	// For each model image, get the best match, which are the most numerous matches from a single image model
	for (int i = 0; i < models_path.size(); i++)
	{
		int max_matches = 0;
		string best_match_fileName;
		Mat best_descriptors;
		vector<DMatch> winning_matches;
		vector<Point> medians;

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
			
			// The maximum number of matches is considered the best match
			if (matches.size() > max_matches)
			{
				winning_matches = matches;
				max_matches = matches.size();
				best_descriptors = model_descriptors[i][j];
			}
		}

		if (max_matches == 0)
		{
			cout << "ORB: No matches found for type " << i << endl;
			continue;
		}

		// Sort matches by distance to prioritize the most reliable matches (smallest distance first)
		sort(winning_matches.begin(), winning_matches.end(), [](const DMatch &a, const DMatch &b)
			 { return a.distance < b.distance; });
		
		save_points(winning_matches, test_keypoints, i);
	}
	cout << "Best matches found from ORB detector\n";
}

// Displays all the points of the best matches
void orb_detector::display_points()
{
	display_points(1.0);
}

// Displays the points of the best matches, parameter perc is the percentage of the best matches to be displayed
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
