#include "haar_detector.hpp"
#include "orb_detector.hpp"
#include "sift_detector.hpp"
#include "dbscan.hpp"
#include "detection.hpp"
#include <random>

template <typename... is_color>
vector<Point> sample_vector_by_color(Mat img, vector<Point> points, is_color... col)
{
	vector<Point> sampled_points;
	for (int k = 0; k < points.size(); k++)
	{
		Point pixel = points[k];
		Vec3b pixel_color = img.at<Vec3b>(pixel);
		if ((col(pixel_color) || ...))
			sampled_points.push_back(points[k]);
	}
	return sampled_points;
}

vector<Point> sample_vector_by_weights(vector<vector<Point>> points, vector<double> weights)
{
	srand(time(NULL));
	if (points.size() != weights.size())
	{
		cout << "[ERROR]: points and weights must have the same size" << endl;
		return vector<Point>();
	}
	vector<Point> sampled_points;

	for (size_t i = 0; i < weights.size(); i++)
	{
		for (size_t j = 0; j < points[i].size(); j++)
		{
			float random = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
			if (random < weights[i])
			{
				sampled_points.push_back(points[i][j]);
			}
		}
	}

	return sampled_points;
}

int main(int argc, char **argv)
{
	// cascade file path
	string sugar_cascade_path = base + sugar + cascade;
	string mustard_cascade_path = base + mustard + cascade;
	string drill_cascade_path = base + drill + cascade;

	// load cascade
	cout << "[INFO]: Initializing HAAR detetctor\n";
	haar_detector cascade_sugar(sugar_cascade_path);
	haar_detector cascade_mustard(mustard_cascade_path);
	haar_detector cascade_drill(drill_cascade_path);

	// load ORB detector
	cout << "[INFO]: Initializing ORB detetctor\n";
	orb_detector orb;

	// load SIFT detector
	cout << "[INFO]: Initializing SIFT detetctor\n";
	sift_detector sift;

	cout << "--------------------------------------------------\n";

	// open all images in folder
	vector<String> sugar_filenames;
	vector<String> mustard_filenames;
	vector<String> drill_filenames;
	glob(base + sugar + img_path, sugar_filenames, false);
	glob(base + mustard + img_path, mustard_filenames, false);
	glob(base + drill + img_path, drill_filenames, false);
	vector<String> filenames;
	filenames.insert(filenames.end(), sugar_filenames.begin(), sugar_filenames.end());
	filenames.insert(filenames.end(), mustard_filenames.begin(), mustard_filenames.end());
	filenames.insert(filenames.end(), drill_filenames.begin(), drill_filenames.end());

	namedWindow("img", WINDOW_NORMAL);
	for (size_t i = 0; i < filenames.size(); i++)
	{
		Mat img = imread(filenames[i], IMREAD_COLOR);
		if (img.empty())
		{
			cerr << "[ERROR]: Could not open image file." << endl;
			return 1;
		}

		// detection HAAR
		vector<Point> s_haar, m_haar, d_haar;
		s_haar = cascade_sugar.detect(img);
		m_haar = cascade_mustard.detect(img);
		d_haar = cascade_drill.detect(img);
		cout << "Best matches found from HAAR detector\n";

		// detection ORB
		vector<Point> s_orb, m_orb, d_orb;
		orb.compute_detection(img);
		vector<vector<Point>> s_orb_points = orb.get_points(0.2);
		s_orb = s_orb_points[0];
		m_orb = s_orb_points[1];
		d_orb = s_orb_points[2];
		orb.display_points();

		// detection SIFT
		vector<Point> s_sift, m_sift, d_sift;
		sift.compute_detection(img);
		vector<vector<Point>> s_sift_points = sift.get_points();
		s_sift = s_sift_points[0];
		m_sift = s_sift_points[1];
		d_sift = s_sift_points[2];

		// concatenate all detected points
		vector<Point> s_total;
		s_total.insert(s_total.end(), s_haar.begin(), s_haar.end());
		s_total.insert(s_total.end(), s_orb.begin(), s_orb.end());
		s_total.insert(s_total.end(), s_sift.begin(), s_sift.end());
		// s_total = sample_vector_by_color(img, s_total, is_yellow, is_white, is_dark);
		s_total = sample_vector_by_weights({s_haar, s_orb, s_sift}, {1.0, 1.0, 1.0});
		vector<Point> m_total;
		m_total.insert(m_total.end(), m_haar.begin(), m_haar.end());
		m_total.insert(m_total.end(), m_orb.begin(), m_orb.end());
		m_total.insert(m_total.end(), m_sift.begin(), m_sift.end());
		// m_total = sample_vector_by_color(img, m_total, is_yellow, is_white, is_blue);
		m_total = sample_vector_by_weights({m_haar, m_orb, m_sift}, {1.0, 1.0, 1.0});
		vector<Point> d_total;
		d_total.insert(d_total.end(), d_haar.begin(), d_haar.end());
		d_total.insert(d_total.end(), d_orb.begin(), d_orb.end());
		d_total.insert(d_total.end(), d_sift.begin(), d_sift.end());
		// d_total = sample_vector_by_color(img, d_total, is_red, is_dark);
		d_total = sample_vector_by_weights({d_haar, d_orb, d_sift}, {1.0, 1.0, 1.0});

		for (size_t i = 0; i < s_total.size(); i++)
		{
			circle(img, s_total[i], 4, Scalar(255, 0, 0), 1);
		}
		for (size_t i = 0; i < m_total.size(); i++)
		{
			circle(img, m_total[i], 4, Scalar(0, 255, 0), 1);
		}
		for (size_t i = 0; i < d_total.size(); i++)
		{
			circle(img, d_total[i], 4, Scalar(0, 0, 255), 1);
		}

		float eps = 40.0f;
		int minPts = 3;

		Rect dense_s = get_dense_cluster(s_total, eps, minPts);
		rectangle(img, dense_s, Scalar(255, 0, 0), 2);

		Rect dense_m = get_dense_cluster(m_total, eps, minPts);
		rectangle(img, dense_m, Scalar(0, 255, 0), 2);

		Rect dense_d = get_dense_cluster(d_total, eps, minPts);
		rectangle(img, dense_d, Scalar(0, 0, 255), 2);

		string img_output_path = "./output/" + get_filename(filenames[i]) + "-box.jpg";
		string txt_output_path = "./output/" + get_filename(filenames[i]) + "-box.txt";

		cout << "[INFO]: saving images and annotations to files\n";

		imwrite(img_output_path, img);

		ofstream file;
		file.open(txt_output_path);
		if (!file.is_open())
		{
			cerr << "[ERROR]: Could not open output txt file." << endl;
			return 1;
		}
		file << sugar.substr(0, sugar.size() - 1) << " " << dense_s.x << " " << dense_s.y << " " << dense_s.x + dense_s.width << " " << dense_s.y + dense_s.height << endl;
		file << mustard.substr(0, mustard.size() - 1) << " " << dense_m.x << " " << dense_m.y << " " << dense_m.x + dense_m.width << " " << dense_m.y + dense_m.height << endl;
		file << drill.substr(0, drill.size() - 1) << " " << dense_d.x << " " << dense_d.y << " " << dense_d.x + dense_d.width << " " << dense_d.y + dense_d.height;
		file.close();

		imshow("img", img);
		waitKey(0);

		cout << "--------------------------------------------------\n";
	}

	return 0;
}