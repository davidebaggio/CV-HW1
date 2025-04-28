// Created by: Pivotto Francesco mat. 2158296
#include "sift_detector.hpp"

void sift_detector::get_model_descriptors()
{
	
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
					string file_mask = file_name.substr(file_name.find_last_of("/") + 1);
					file_mask = file_mask.substr(0, file_mask.find_last_of("_")) + "_mask.png";

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

void sift_detector::optimize_image(Mat &src)
{
	Mat img_gray, img_equalized;

	cvtColor(src, img_gray, cv::COLOR_BGR2GRAY);

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
	FlannBasedMatcher matcher;
	vector<vector<DMatch>> knn_matches;						
	matcher.knnMatch(model_desc, img_desc, knn_matches, 2); 

	vector<DMatch> good_matches;
	for (const auto &m : knn_matches)
	{
		if (m.size() == 2 && m[0].distance < 0.82f * m[1].distance)
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

void sift_detector::compute_detection(Mat img)
{
	img_test = img;

	Mat img_opt = img_test.clone();
	optimize_image(img_opt, true);

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

		sort(winning_matches.begin(), winning_matches.end(), [](const DMatch &a, const DMatch &b)
			 { return a.distance < b.distance; });
		save_points(winning_matches, img_kpt, i);
	}

	cout << "Best matches found from SIFT detector\n";
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


void sift_detector::display_points()
{
	display_points(1.0);
}


void sift_detector::display_points(float perc)
{

	vector<Mat> mat_matches = vector<Mat>(3);
	mat_matches[0] = img_test.clone(); 
	mat_matches[1] = img_test.clone(); 
	mat_matches[2] = img_test.clone(); 

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