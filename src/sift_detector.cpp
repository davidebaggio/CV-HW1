#include "sift_detector.hpp"

double sift_detector::compute_median(vector<double> values)
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

void sift_detector::save_points(vector<DMatch> matches, Mat descriptors_1, vector<KeyPoint> keypoints_1, Mat descriptors_2, vector<KeyPoint> keypoints_2, string fileName, int category)
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
		if (m.size() == 2 && m[0].distance < 0.78f * m[1].distance)
		{
			good_matches.push_back(m[0]);
		}
	}

	return good_matches;
}

sift_detector::sift_detector(Mat img)
{
	this->img_test = img;
	sift = SIFT::create();
}

void sift_detector::compute_detection()
{

	Mat img_opt = img_test.clone();

	cvtColor(img_opt, img_opt, COLOR_BGR2GRAY);

	vector<KeyPoint> img_kpt;
	Mat img_desc;

	sift->detect(img_opt, img_kpt);
	sift->compute(img_opt, img_kpt, img_desc);

	for (int i = 0; i < models_path.size(); i++)
	{
		try
		{
			regex regexPattern(pattern);
			int index = 0;

			Mat model;

			int max_matches = 0;
			string bestMatchFileName;
			Mat best_descriptors;
			vector<KeyPoint> best_keypoints;
			vector<DMatch> winning_matches;

			for (const auto &entry : directory_iterator(models_path[i]))
			{
				if (entry.is_regular_file())
				{

					string fileName = entry.path().filename().string();

					if (regex_match(fileName, regexPattern))
					{
						string fullFileName = models_path[i] + "/" + entry.path().filename().string();

						model = imread(fullFileName);

						if (model.rows == 0 && model.cols == 0)
						{
							cerr << "Error: Could not open image file: " << fileName << endl;
							continue;
						}

						index++;

						Mat model_desc;
						vector<KeyPoint> model_kpt;

						cvtColor(model, model, COLOR_BGR2GRAY);

						sift->detect(model, model_kpt);
						sift->compute(model, model_kpt, model_desc);

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
						vector<DMatch> matches = get_matches(model_desc, img_desc);

						if (matches.size() > max_matches)
						{
							winning_matches = matches;
							max_matches = matches.size();
							bestMatchFileName = fullFileName;
							best_descriptors = model_desc;
							best_keypoints = model_kpt;
						}
					}
				}
			}

			save_points(winning_matches, best_descriptors, best_keypoints, img_desc, img_kpt, bestMatchFileName, i);

			cout << "Best match for type " << i << " is: " << bestMatchFileName << " with " << max_matches << " matches." << endl;
			cout << "-----------------------------" << endl;
		}
		catch (const exception &e)
		{
			cerr << "Error: " << e.what() << endl;
		}
	}
}

void sift_detector::display_points()
{
	vector<Mat> mat_matches = vector<Mat>(3);
	mat_matches[0] = img_test.clone(); // sugar
	mat_matches[1] = img_test.clone(); // mustard
	mat_matches[2] = img_test.clone(); // drill

	for (size_t i = 0; i < points.size(); i++)
	{

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
		imshow(category + " matches", mat_matches[i]);
	}
	waitKey(0);
}

vector<vector<Point>> sift_detector::get_points()
{
	return points;
}

/* int main()
{

	// Load the image
	Mat test = imread("object_detection_dataset/035_power_drill/test_images/35_0010_001462-color.jpg");

	if (test.rows == 0 && test.cols == 0)
	{
		cerr << "Error: Could not open image file: " << "../object_detection_dataset/004_sugar_box/test.png" << endl;
		return -1;
	}

	sift_Detector detector(test);

	detector.computeDetection();
	detector.displayPoints();

	waitKey(0);

	return 0;
} */