#include <opencv2/opencv.hpp>
#include <filesystem>
#include <fstream>

const static std::string base = "./data/";
const static std::string sugar = "004_sugar_box/";
const static std::string mustard = "006_mustard_bottle/";
const static std::string drill = "035_power_drill/";
const static std::string img_path = "test_images/*.jpg";
const static std::string label_path = "labels/";
const static std::string models_path = "models/*_mask.png";
const static std::string negative_path = "negative_images/";

std::string get_filename(std::string path)
{
	std::string filename = path.substr(path.find_last_of("/") + 1);
	return filename.substr(0, filename.find_last_of("_"));
}

bool out_annotations(std::vector<std::string> files, std::string type_folder)
{
	std::string out_path = base + type_folder + "annotations.txt";
	std::ofstream out(out_path);
	if (!out.is_open())
	{
		printf("[ERROR]: could not open file\n");
		return false;
	}

	for (size_t i = 0; i < files.size(); i++)
	{
		cv::Mat mask = cv::imread(files[i], cv::IMREAD_GRAYSCALE);
		std::vector<std::vector<cv::Point>> contours;
		cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

		for (auto &contour : contours)
		{
			cv::Rect box = cv::boundingRect(contour);
			out << files[i] << " 1 " << box.x << " " << box.y << " "
				<< box.width << " " << box.height << std::endl;
		}
	}
	printf("[INFO]: annotations generated\n");
	return true;
}

int main()
{
	std::vector<cv::String> sugar_box;
	cv::glob(base + sugar + models_path, sugar_box, false);

	std::vector<cv::String> mustard_bottle;
	cv::glob(base + mustard + models_path, mustard_bottle, false);

	std::vector<cv::String> power_drill;
	cv::glob(base + drill + models_path, power_drill, false);

	if (!out_annotations(sugar_box, sugar))
	{
		printf("[ERROR]: could not generate annotations\n");
		return 1;
	}

	/* if (!out_annotations(mustard_bottle, mustard))
	{
		printf("[ERROR]: could not generate annotations\n");
		return 1;
	}

	if (!out_annotations(power_drill, drill))
	{
		printf("[ERROR]: could not generate annotations\n");
		return 1;
	} */

	return 0;
}
