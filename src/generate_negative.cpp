#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <string>

const static std::string base = "./data/";
const static std::string sugar = "004_sugar_box/";
const static std::string mustard = "006_mustard_bottle/";
const static std::string drill = "035_power_drill/";
const static std::string img_path = "test_images/*.jpg";
const static std::string label_path = "labels/";
const static std::string models_path = "models/";
const static std::string negative_path = "negative_images/";

// generate negative images from cutting the original images
std::vector<cv::Mat> cut_image(cv::Mat img, cv::Point2i p1, cv::Point2i p2)
{
	std::vector<cv::Mat> cuts;
	cuts.push_back(img(cv::Rect(0, 0, p1.x, img.rows)));
	cuts.push_back(img(cv::Rect(0, 0, img.cols, p1.y)));
	cuts.push_back(img(cv::Rect(p2.x, 0, img.cols - p2.x, img.rows)));
	cuts.push_back(img(cv::Rect(0, p2.y, img.cols, img.rows - p2.y)));
	return cuts;
}

std::string get_filename(std::string path)
{
	std::string filename = path.substr(path.find_last_of("/") + 1);
	return filename.substr(0, filename.find_last_of("-"));
}

bool generate_negative(std::vector<cv::String> files, std::string type_folder)
{
	cv::namedWindow("image", cv::WINDOW_NORMAL);
	for (size_t i = 0; i < files.size(); i++)
	{
		// printf("[INFO] file: %s\n", files[i].c_str());
		cv::Mat img = cv::imread(files[i]);
		if (img.empty())
		{
			printf("[ERROR]: could not read image\n");
			return false;
		}

		// read label file
		std::string label_file = base + type_folder + label_path + get_filename(files[i]) + "-box.txt";
		// printf("label_file: %s\n", label_file.c_str());
		std::ifstream label_stream(label_file);
		if (!label_stream.is_open())
		{
			printf("[ERROR]: could not open label file\n");
			return false;
		}

		std::string line;
		while (std::getline(label_stream, line))
		{
			// split by space
			// printf("line: %s\n", line.c_str());
			std::vector<std::string> tokens;
			std::string token;
			std::istringstream tokenStream(line);
			while (std::getline(tokenStream, token, ' '))
			{
				tokens.push_back(token);
			}
			if (tokens[0] != type_folder.substr(0, type_folder.size() - 1))
				continue;
			cv::Point2i p1(std::stoi(tokens[1]), std::stoi(tokens[2]));
			cv::Point2i p2(std::stoi(tokens[3]), std::stoi(tokens[4]));
			std::vector<cv::Mat> cuts = cut_image(img, p1, p2);
			for (size_t j = 0; j < cuts.size(); j++)
			{
				if (cuts[j].empty())
				{
					// printf("[ERROR]: could not cut image\n");
					continue;
				}
				std::string neg = base + type_folder + negative_path + get_filename(files[i]) + "-cut" + std::to_string(j) + ".jpg";
				// printf("neg: %s\n", neg.c_str());
				if (!cv::imwrite(neg, cuts[j]))
				{
					printf("[ERROR]: could not write image\n");
					return false;
				}
				printf("%s\n", ("./" + negative_path + get_filename(files[i]) + "-cut" + std::to_string(j) + ".jpg").c_str());
			}
			// printf("[INFO]: negative images saved\n");
			break;
		}
	}
	return true;
}

int main(int argc, char **argv)
{

	std::vector<cv::String> sugar_box;
	cv::glob(base + sugar + img_path, sugar_box, false);

	std::vector<cv::String> mustard_bottle;
	cv::glob(base + mustard + img_path, mustard_bottle, false);

	std::vector<cv::String> power_drill;
	cv::glob(base + drill + img_path, power_drill, false);

	printf("sugar_box: %ld\n", sugar_box.size());
	printf("mustard_bottle: %ld\n", mustard_bottle.size());
	printf("power_drill: %ld\n", power_drill.size());

	if (!generate_negative(sugar_box, sugar))
	{
		printf("[ERROR]: could not generate negative images\n");
		return 1;
	}
	printf("---------\n");
	if (!generate_negative(mustard_bottle, mustard))
	{
		printf("[ERROR]: could not generate negative images\n");
		return 1;
	}
	printf("---------\n");
	if (!generate_negative(power_drill, drill))
	{
		printf("[ERROR]: could not generate negative images\n");
		return 1;
	}

	return 0;
}