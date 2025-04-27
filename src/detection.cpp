#include "detection.hpp"

std::string get_filename(std::string path)
{
	std::string filename = path.substr(path.find_last_of("/") + 1);
	return filename.substr(0, filename.find_last_of("-"));
}

bool is_yellow(cv::Vec3b pixel)
{
	return (pixel[0] < 30 && pixel[1] > 90 && pixel[2] > 90);
}

bool is_dark(cv::Vec3b pixel)
{
	return (pixel[0] < 40 && pixel[1] < 40 && pixel[2] < 40);
}

bool is_white(cv::Vec3b pixel)
{
	return (pixel[0] > 180 && pixel[1] > 180 && pixel[2] > 180);
}

bool is_red(cv::Vec3b pixel)
{
	return (pixel[0] < 40 && pixel[1] < 40 && pixel[2] > 180);
}

bool is_blue(cv::Vec3b pixel)
{
	return (pixel[0] > 180 && pixel[1] < 40 && pixel[2] < 40);
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

float intersection_over_union(cv::Rect rect1, cv::Rect rect2)
{
	cv::Rect intersection = rect1 & rect2;
	float intersection_area = intersection.area();
	float union_area = rect1.area() + rect2.area() - intersection_area;
	return intersection_area / union_area;
}

void display_performances()
{
	std::vector<cv::String> tested_annotations;
	cv::glob("output/*.txt", tested_annotations, false);

	std::vector<cv::String> true_labels, true_label_s, true_label_m, true_label_d;
	cv::glob(base + sugar + label_path + "*.txt", true_label_s, false);
	cv::glob(base + mustard + label_path + "*.txt", true_label_m, false);
	cv::glob(base + drill + label_path + "*.txt", true_label_d, false);

	true_labels.insert(true_labels.end(), true_label_s.begin(), true_label_s.end());
	true_labels.insert(true_labels.end(), true_label_m.begin(), true_label_m.end());
	true_labels.insert(true_labels.end(), true_label_d.begin(), true_label_d.end());

	int obj_detected_num = 0;
	int obj_total = 0;
	for (size_t i = 0; i < true_labels.size(); i++)
	{
		for (size_t j = 0; j < tested_annotations.size(); j++)
		{
			if (!(get_filename(true_labels[i]) != get_filename(tested_annotations[j])))
				continue;

			std::ifstream label_file(true_labels[i]);
			std::ifstream tested_file(tested_annotations[i]);

			if (!label_file.is_open() || !tested_file.is_open())
			{
				std::cerr << "[ERROR]: Could not open label or tested file." << std::endl;
				return;
			}
			std::string label_line, tested_line;
			while (std::getline(label_file, label_line))
			{
				while (std::getline(tested_file, tested_line))
				{
					std::string label_class = label_line.substr(0, label_line.find(" "));
					std::string tested_class = tested_line.substr(0, tested_line.find(" "));
					if (label_class == tested_class)
					{
						obj_total++;
						std::string label_box = label_line.substr(label_line.find(" ") + 1);
						std::string tested_box = tested_line.substr(tested_line.find(" ") + 1);

						std::vector<std::string> label_box_values, tested_box_values;
						std::string label_box_value, tested_box_value;
						std::stringstream label_box_ss(label_box);
						std::stringstream tested_box_ss(tested_box);

						while (std::getline(label_box_ss, label_box_value, ' '))
						{
							label_box_values.push_back(label_box_value);
						}
						while (std::getline(tested_box_ss, tested_box_value, ' '))
						{
							tested_box_values.push_back(tested_box_value);
						}

						cv::Rect label_rect = cv::Rect(std::stoi(label_box_values[0]), std::stoi(label_box_values[1]), std::stoi(label_box_values[2]) - std::stoi(label_box_values[0]), std::stoi(label_box_values[3]) - std::stoi(label_box_values[1]));
						cv::Rect tested_rect = cv::Rect(std::stoi(tested_box_values[0]), std::stoi(tested_box_values[1]), std::stoi(tested_box_values[2]) - std::stoi(tested_box_values[0]), std::stoi(tested_box_values[3]) - std::stoi(tested_box_values[1]));

						float IoU = intersection_over_union(label_rect, tested_rect);
						printf("%s - %s -> IoU: %f\n", get_filename(true_labels[i]), label_class.c_str(), IoU);
						if (IoU > 0.5)
						{
							obj_detected_num++;
						}
					}
				}
				tested_file.clear();
				tested_file.seekg(0, std::ios::beg);
			}
		}
	}

	printf("Total objects detected: %d/%d\n", obj_detected_num, obj_total);
}