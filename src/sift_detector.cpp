#include "sift_detector.hpp"

void sift_detector::get_model_descriptors()
{
	for (int i = 0; i < models_path.size(); i++)
	{
		try
		{
			regex regexPattern(pattern);

			Mat model;
			Mat mask;

			for (const auto &entry : directory_iterator(models_path[i]))
			{
				if (entry.is_regular_file())
				{
					string fileName = entry.path().filename().string();
					string file_mask = fileName.substr(fileName.find_last_of("/") + 1);
					file_mask = file_mask.substr(0, file_mask.find_last_of("_")) + "_mask.png";

					if (regex_match(fileName, regexPattern))
					{
						string fullFileName = models_path[i] + "/" + entry.path().filename().string();

						model = imread(fullFileName);
						mask = imread(models_path[i] + "/" + file_mask, IMREAD_GRAYSCALE);

						if (model.empty() || mask.empty())
						{
							cerr << "Error: Could not open image file: " << fileName << endl;
							continue;
						}

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
			cerr << "Error: " << e.what() << endl;
		}
	}
}

void sift_detector::optimize_image(Mat &src, bool isImg_test)
{

	Mat img_gray, img_equalized;

	cvtColor(src, img_gray, cv::COLOR_BGR2GRAY);

	/*
	if(isImg_test)
		bilateralFilter(img_gray, img_filtered, 5, 50, 50);
	else
		bilateralFilter(img_gray, img_filtered, 5, 50, 50);
	*/

	equalizeHist(img_gray, img_equalized);

	src = img_equalized.clone();
}

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

vector<DMatch> sift_detector::get_matches(const Mat &model_desc, const Mat &img_desc)
{
	// Matcher con knnMatch
	FlannBasedMatcher matcher;
	vector<vector<DMatch>> knn_matches;						// Vettore di vettori per knnMatch
	matcher.knnMatch(model_desc, img_desc, knn_matches, 2); // k=2 per il test di Lowe

	// Lowe's Ratio Test
	vector<DMatch> good_matches;
	for (const auto &m : knn_matches)
	{
		if (m.size() == 2 && m[0].distance < 0.88f * m[1].distance)
		{
			good_matches.push_back(m[0]);
		}
	}

	return good_matches;
}

sift_detector::sift_detector()
{
	sift = SIFT::create();
	get_model_descriptors();
}

void sift_detector::compute_detection(Mat &img_test)
{
	Mat img_opt = img_test.clone();
	optimize_image(img_opt, true);

	vector<KeyPoint> img_kpt;
	Mat img_desc;

	sift->detect(img_opt, img_kpt);
	sift->compute(img_opt, img_kpt, img_desc);

	vector<DMatch> winning_matches;

	for (int i = 0; i < model_descriptors.size(); i++)
	{
		int max_matches = 0;
		for (int j = 0; j < model_descriptors[i].size(); j++)
		{
			vector<DMatch> matches = get_matches(model_descriptors[i][j], img_desc);

			if (matches.size() > max_matches)
			{
				winning_matches = matches;
				max_matches = matches.size();
			}
		}
		sort(winning_matches.begin(), winning_matches.end(), [](const DMatch &a, const DMatch &b)
			 { return a.distance < b.distance; });
		save_points(winning_matches, img_kpt, i);

		cout << "Best match for type " << i << " matches: " << max_matches << endl;
		cout << "-----------------------------" << endl;
	}
}

vector<vector<Point>> sift_detector::get_points()
{
	return points;
}

vector<vector<Point>> sift_detector::get_points(float perc)
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

void sift_detector::display_points()
{
	display_points(1.0);
}

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
