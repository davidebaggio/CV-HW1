// Created by: Pivotto Francesco mat. 2158296
#include "sift_detector.hpp"

//Helper function to load the model images and their mask from the Dataset folder and saves their descriptors
void sift_detector::get_model_descriptors()
{
	// Reads every model of the objects from the the dataset folders
	for (int i = 0; i < models_path.size(); i++)
	{
		try
		{
			regex regex_pattern(pattern);

			Mat model;
			Mat mask;

			for (const auto &entry : directory_iterator(models_path[i]))
			{
				if (entry.is_regular_file())
				{
					string file_name = entry.path().filename().string();
					string file_mask = fileName.substr(file_name.find_last_of("/") + 1);
					file_mask = file_mask.substr(0, file_mask.find_last_of("_")) + "_mask.png";

					// Check if the file name matches the regex pattern
					if (regex_match(file_name, regex_pattern))
					{
						string full_file_name = models_path[i] + "/" + entry.path().filename().string();

						model = imread(full_file_name);
						mask = imread(models_path[i] + "/" + file_mask, IMREAD_GRAYSCALE);

						if (model.empty() || mask.empty())
						{
							cerr << "[ERROR]: Could not open image file: " << file_name << " [SIFT]" << endl;
							continue;
						}

						// Compute all the descriptors for the each model images of the dataset
						Mat model_desc;
						vector<KeyPoint> model_kpt;

						optimize_image(model, false);

						sift->detect(model, model_kpt, mask);
						sift->compute(model, model_kpt, model_desc);

						model_descriptors[i].push_back(model_desc);
					}
				}
			}
		}
		catch (const exception &e)
		{
			cerr << "[ERROR]: " << e.what() << endl;
		}
	}
}

//Helper function to optimize the image
void sift_detector::optimize_image(Mat &src, bool isImg_test)
{
	// It converts the image to grayscale and equalizes the histogram
	Mat img_gray, img_equalized;

	cvtColor(src, img_gray, cv::COLOR_BGR2GRAY);	

	equalizeHist(img_gray, img_equalized);

	src = img_equalized.clone();
}

// Helper function to save the points of the best matches
void sift_detector::save_points(vector<DMatch> &matches, vector<KeyPoint> &img_kpt, int category)
{
	vector<Point> matches_points;

	for (size_t i = 0; i < matches.size(); i++)
	{
		int i_img_point = matches[i].trainIdx;

		Point img_point = img_kpt[i_img_point].pt;

		matches_points.push_back(img_point);
	}

	points[category] = matches_points;
}

// Helper function to get matches between an image model descriptors and image test descriptors
vector<DMatch> sift_detector::get_matches(const Mat &model_desc, const Mat &img_desc)
{
	BFMatcher matcher(NORM_L2 , true);
	vector<DMatch> matches;
	matcher.match(model_desc, img_desc, matches);

	return matches;
}

// Constructor
// Initializes the SIFT detector and call the get_model_descriptors function to save the model images from the dataset folders
sift_detector::sift_detector()
{
	sift = SIFT::create();
	get_model_descriptors();
}

// Computes the detection between the model descriptors and the test image descriptors
// The test image is passed as a parameter
void sift_detector::compute_detection(Mat img)
{
	img_test = img;
	
	// Optimize the test image
	Mat img_opt = img_test.clone();
	optimize_image(img_opt, true);

	// Compute the descriptors for the test image
	vector<KeyPoint> img_kpt;
	Mat img_desc;

	sift->detect(img_opt, img_kpt);
	sift->compute(img_opt, img_kpt, img_desc);

	if (img_desc.empty())
	{
		cout << "[ERROR]: No descriptors found for test image [SIFT]" << endl;
		return;
	}

	vector<DMatch> winning_matches;

	// For each model image, get the best match, which are the most numerous matches from a single image model
	for (int i = 0; i < model_descriptors.size(); i++)
	{
		int max_matches = 0;
		for (int j = 0; j < model_descriptors[i].size(); j++)
		{
			vector<DMatch> matches = get_matches(model_descriptors[i][j], img_desc);

			if (matches.empty())
			{
				cout << "[ERROR]: No matches found for type " << i << " [SIFT]" << endl;
				continue;
			}
	
			if (matches.size() == 0)
			{
				cout << "[ERROR]: Image matches below minimum threshold: " << matches.size() << " [SIFT]" << endl;
				continue;
			}

			// The maximum number of matches is considered the best match
			if (matches.size() > max_matches)
			{
				winning_matches = matches;
				max_matches = matches.size();
			}
		}

		if (max_matches == 0)
		{
			cout << "[ERROR]: No matches found for type " << i << " [SIFT]" << endl;
			continue;
		}

		// Sort matches by distance to prioritize the most reliable matches (smallest distance first)
		sort(winning_matches.begin(), winning_matches.end(), [](const DMatch &a, const DMatch &b)
			 { return a.distance < b.distance; });
		save_points(winning_matches, img_kpt, i);
	}

	cout << "Best matches found from SIFT detector\n";
}

// Returns the points
vector<vector<Point>> sift_detector::get_points()
{
	return points;
}

// Returns the points with a percentage of the best matches
vector<vector<Point>> sift_detector::get_points(float perc)
{
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

// Displays all the points of the best matches
void sift_detector::display_points()
{
	display_points(1.0);
}

// Displays the points of the best matches, parameter perc is the percentage of the best matches to be displayed
void sift_detector::display_points(float perc)
{

	vector<Mat> mat_matches = vector<Mat>(3);
	mat_matches[0] = img_test.clone(); // sugar
	mat_matches[1] = img_test.clone(); // mustard
	mat_matches[2] = img_test.clone(); // drill

	vector<vector<Point>> pp = get_points(perc);

	for (size_t i = 0; i < pp.size(); i++)
	{
		if (points[i].size() == 0)
		{
			cout << "[ERROR]: No points found for type " << i << " [SIFT]" << endl;
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
			cout << "[ERROR]: No matches found for type " << i << " [SIFT]" << endl;
			continue;
		}
		imshow(category + " matches", mat_matches[i]);
	}
	waitKey(0);
}