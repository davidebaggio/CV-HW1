#include <opencv2/opencv.hpp>
#include <iostream>
#include "haar_detector.hpp"
#include "orb_detector.hpp"
#include "sift_detector.hpp"
#include "dbscan.hpp"
#include "detection.hpp"

int main(int argc, char **argv)
{
	// cascade file path
	std::string sugar_cascade_path = base + sugar + cascade;
	std::string mustard_cascade_path = base + mustard + cascade;
	std::string drill_cascade_path = base + drill + cascade;

	// load cascade
	haar_detector cascade_sugar(sugar_cascade_path);
	haar_detector cascade_mustard(mustard_cascade_path);
	haar_detector cascade_drill(drill_cascade_path);

	// open all images in folder
	std::vector<cv::String> sugar_filenames;
	std::vector<cv::String> mustard_filenames;
	std::vector<cv::String> drill_filenames;
	cv::glob(base + sugar + img_path, sugar_filenames, false);
	cv::glob(base + mustard + img_path, mustard_filenames, false);
	cv::glob(base + drill + img_path, drill_filenames, false);
	std::vector<cv::String> filenames;
	filenames.insert(filenames.end(), sugar_filenames.begin(), sugar_filenames.end());
	filenames.insert(filenames.end(), mustard_filenames.begin(), mustard_filenames.end());
	filenames.insert(filenames.end(), drill_filenames.begin(), drill_filenames.end());

	cv::namedWindow("img", cv::WINDOW_NORMAL);
	for (size_t i = 0; i < filenames.size(); i++)
	{
		cv::Mat img = cv::imread(filenames[i], cv::IMREAD_COLOR);
		if (img.empty())
		{
			std::cerr << "[ERROR]: Could not open image file." << std::endl;
			return 1;
		}

		// detection HAAR
		std::vector<cv::Point> s_haar, m_haar, d_haar;
		s_haar = cascade_sugar.detect(img);
		m_haar = cascade_mustard.detect(img);
		d_haar = cascade_drill.detect(img);

		// detection ORB
		std::vector<cv::Point> s_orb, m_orb, d_orb;
		orb_detector orb(img);
		orb.compute_detection();
		std::vector<std::vector<cv::Point>> s_orb_points = orb.get_points();
		s_orb = s_orb_points[0];
		m_orb = s_orb_points[1];
		d_orb = s_orb_points[2];

		// detection SIFT
		std::vector<cv::Point> s_sift, m_sift, d_sift;
		sift_detector sift(img);
		sift.compute_detection();
		std::vector<std::vector<cv::Point>> s_sift_points = sift.get_points();
		s_sift = s_sift_points[0];
		m_sift = s_sift_points[1];
		d_sift = s_sift_points[2];

		// concatenate all detected points
		std::vector<cv::Point> s_total;
		s_total.insert(s_total.end(), s_haar.begin(), s_haar.end());
		s_total.insert(s_total.end(), s_orb.begin(), s_orb.end());
		s_total.insert(s_total.end(), s_sift.begin(), s_sift.end());
		s_total = sample_vector(img, s_total, is_yellow, is_white, is_dark);
		std::vector<cv::Point> m_total;
		m_total.insert(m_total.end(), m_haar.begin(), m_haar.end());
		m_total.insert(m_total.end(), m_orb.begin(), m_orb.end());
		m_total.insert(m_total.end(), m_sift.begin(), m_sift.end());
		m_total = sample_vector(img, m_total, is_yellow, is_white, is_blue);
		std::vector<cv::Point> d_total;
		d_total.insert(d_total.end(), d_haar.begin(), d_haar.end());
		d_total.insert(d_total.end(), d_orb.begin(), d_orb.end());
		d_total.insert(d_total.end(), d_sift.begin(), d_sift.end());
		d_total = sample_vector(img, d_total, is_red, is_dark);

		for (size_t i = 0; i < s_total.size(); i++)
		{
			cv::circle(img, s_total[i], 4, cv::Scalar(255, 0, 0), 1);
		}
		for (size_t i = 0; i < m_total.size(); i++)
		{
			cv::circle(img, m_total[i], 4, cv::Scalar(0, 255, 0), 1);
		}
		for (size_t i = 0; i < d_total.size(); i++)
		{
			cv::circle(img, d_total[i], 4, cv::Scalar(0, 0, 255), 1);
		}

		float eps = 80.0f;
		int minPts = 3;

		Rect dense_s = get_dense_cluster(s_total, eps, minPts);
		cv::rectangle(img, dense_s, cv::Scalar(255, 0, 0), 2);

		Rect dense_m = get_dense_cluster(m_total, eps, minPts);
		cv::rectangle(img, dense_m, cv::Scalar(0, 255, 0), 2);

		Rect dense_d = get_dense_cluster(d_total, eps, minPts);
		cv::rectangle(img, dense_d, cv::Scalar(0, 0, 255), 2);

		std::string img_output_path = "./output/" + get_filename(filenames[i]) + "-box.jpg";
		std::string txt_output_path = "./output/" + get_filename(filenames[i]) + "-box.txt";

		cv::imwrite(img_output_path, img);

		std::ofstream file;
		file.open(txt_output_path);
		if (!file.is_open())
		{
			std::cerr << "[ERROR]: Could not open output txt file." << std::endl;
			return 1;
		}
		file << sugar.substr(0, sugar.size() - 1) << " " << dense_s.x << " " << dense_s.y << " " << dense_s.x + dense_s.width << " " << dense_s.y + dense_s.height << std::endl;
		file << mustard.substr(0, mustard.size() - 1) << " " << dense_m.x << " " << dense_m.y << " " << dense_m.x + dense_m.width << " " << dense_m.y + dense_m.height << std::endl;
		file << drill.substr(0, drill.size() - 1) << " " << dense_d.x << " " << dense_d.y << " " << dense_d.x + dense_d.width << " " << dense_d.y + dense_d.height;
		file.close();

		cv::imshow("img", img);
		cv::waitKey(0);
	}

	display_performances();

	return 0;
}