#include <opencv2/opencv.hpp>
#include <iostream>
#include "orb_detector.hpp"
#include "sift_detector.hpp"
#include "dbscan.hpp"

const static std::string base = "./data/";
const static std::string sugar = "004_sugar_box/";
const static std::string mustard = "006_mustard_bottle/";
const static std::string drill = "035_power_drill/";
const static std::string img_path = "test_images/*.jpg";
const static std::string label_path = "labels/";
const static std::string models_path = "models/*_mask.png";
const static std::string negative_path = "negative_images/";
const static std::string cascade = "object_cascade/cascade.xml";

bool is_yellow(cv::Vec3b pixel)
{
	return (pixel[0] < 30 && pixel[1] > 100 && pixel[2] > 110);
}

bool is_dark(cv::Vec3b pixel)
{
	return (pixel[0] < 30 && pixel[1] < 30 && pixel[2] < 30);
}

bool is_white(cv::Vec3b pixel)
{
	return (pixel[0] > 200 && pixel[1] > 200 && pixel[2] > 200);
}

bool is_red(cv::Vec3b pixel)
{
	return (pixel[0] < 30 && pixel[1] < 30 && pixel[2] > 200);
}

bool is_blue(cv::Vec3b pixel)
{
	return (pixel[0] > 200 && pixel[1] < 30 && pixel[2] < 30);
}

template <typename... is_color>
std::vector<cv::Point> sample_vector(cv::Mat img, std::vector<cv::Point> points, is_color... col)
{
	std::vector<cv::Point> sampled_points;
	for (int k = 0; k < points.size(); k++)
	{
		cv::Point pixel = points[k];
		/* bool in_vec = true;
		for (int i = -1; i <= 1; i++)
		{
			for (int j = -1; j <= 1; j++)
			{
				cv::Vec3b pixel_color = img.at<cv::Vec3b>(pixel.x + i, pixel.y + j);
				if (!((col(pixel_color) || ...)))
				{
					in_vec = false;
					}
					}
					}
					if (in_vec)
					sampled_points.push_back(points[k]); */
		cv::Vec3b pixel_color = img.at<cv::Vec3b>(pixel);
		if ((col(pixel_color) || ...))
			sampled_points.push_back(points[k]);
	}
	return sampled_points;
}

int main(int argc, char **argv)
{
	// cascade file path
	std::string sugar_cascade_path = base + sugar + cascade;
	std::string mustard_cascade_path = base + mustard + cascade;
	std::string drill_cascade_path = base + drill + cascade;

	// load cascade
	cv::CascadeClassifier cascade_sugar;
	cv::CascadeClassifier cascade_mustard;
	cv::CascadeClassifier cascade_drill;

	if (!cascade_drill.load(drill_cascade_path) || !cascade_mustard.load(mustard_cascade_path) || !cascade_sugar.load(sugar_cascade_path))
	{
		std::cout << "Error loading cascades" << std::endl;
		return 1;
	}

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
			return -1;
		}
		cv::Mat img_thresholded = img.clone();
		// threshold on yellow
		for (size_t i = 0; i < img_thresholded.rows; i++)
		{
			for (size_t j = 0; j < img_thresholded.cols; j++)
			{
				if ((img_thresholded.at<cv::Vec3b>(i, j)[0] > 30 || img_thresholded.at<cv::Vec3b>(i, j)[1] < 100 || img_thresholded.at<cv::Vec3b>(i, j)[2] < 110) && (img_thresholded.at<cv::Vec3b>(i, j)[0] > 30 || img_thresholded.at<cv::Vec3b>(i, j)[1] > 30 || img_thresholded.at<cv::Vec3b>(i, j)[2] < 130))
				{
					img_thresholded.at<cv::Vec3b>(i, j) = cv::Vec3b(0, 0, 0);
				}
			}
		}

		/* cv::imshow("img", img_thresholded);
		cv::waitKey(0); */

		cv::Mat gray;

		cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
		std::vector<cv::Rect> s, m, d;
		cascade_sugar.detectMultiScale(gray, s, 1.1, 2, 0 | cv::CASCADE_SCALE_IMAGE);
		cascade_mustard.detectMultiScale(gray, m, 1.1, 2, 0 | cv::CASCADE_SCALE_IMAGE);
		cascade_drill.detectMultiScale(gray, d, 1.1, 2, 0 | cv::CASCADE_SCALE_IMAGE);

		std::vector<cv::Point> s_haar, m_haar, d_haar;
		for (size_t i = 0; i < s.size(); i++)
		{
			s_haar.push_back(cv::Point(s[i].x + s[i].width / 2, s[i].y + s[i].height / 2));
		}
		for (size_t i = 0; i < m.size(); i++)
		{
			m_haar.push_back(cv::Point(m[i].x + m[i].width / 2, m[i].y + m[i].height / 2));
		}
		for (size_t i = 0; i < d.size(); i++)
		{
			d_haar.push_back(cv::Point(d[i].x + d[i].width / 2, d[i].y + d[i].height / 2));
		}

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

		std::vector<cv::Point> s_total;
		s_total.insert(s_total.end(), s_haar.begin(), s_haar.end());
		s_total.insert(s_total.end(), s_orb.begin(), s_orb.end());
		s_total.insert(s_total.end(), s_sift.begin(), s_sift.end());
		printf("%d\n", s_total.size());
		s_total = sample_vector(img, s_total, is_yellow, is_white, is_dark);
		printf("%d\n", s_total.size());

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

		float eps = 75.0f;
		int minPts = 3;

		Rect dense_s = getDensestClusterRect(s_total, eps, minPts);
		cv::rectangle(img, dense_s, cv::Scalar(255, 0, 0), 2);

		Rect dense_m = getDensestClusterRect(m_total, eps, minPts);
		cv::rectangle(img, dense_m, cv::Scalar(0, 255, 0), 2);

		Rect dense_d = getDensestClusterRect(d_total, eps, minPts);
		cv::rectangle(img, dense_d, cv::Scalar(0, 0, 255), 2);
		// show image
		cv::imshow("img", img);
		cv::waitKey(0);
	}
	return 0;
}