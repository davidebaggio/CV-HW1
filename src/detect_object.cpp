#include <opencv2/opencv.hpp>
#include <iostream>

const static std::string base = "./data/";
const static std::string sugar = "004_sugar_box/";
const static std::string mustard = "006_mustard_bottle/";
const static std::string drill = "035_power_drill/";
const static std::string img_path = "test_images/";
const static std::string label_path = "labels/";
const static std::string models_path = "models/*_mask.png";
const static std::string negative_path = "negative_images/";
const static std::string cascade = "object_cascade/cascade.xml";

int main(int argc, char **argv)
{
	// cascade file path
	// std::string cascade_path = base + sugar + cascade;
	// std::string cascade_path = base + mustard + cascade;
	std::string cascade_path = base + drill + cascade;

	// load cascade
	cv::CascadeClassifier cascade_drill;

	if (!cascade_drill.load(cascade_path))
	{
		std::cout << "Error loading drill cascade" << std::endl;
		return 1;
	}
	// load image
	// cv::Mat img = cv::imread(base + mustard + img_path + "6_0030_001027-color.jpg", cv::IMREAD_COLOR);
	cv::Mat img = cv::imread(base + drill + img_path + "35_0010_000491-color.jpg", cv::IMREAD_COLOR);
	cv::Mat gray;

	cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
	std::vector<cv::Rect> objects;

	cascade_drill.detectMultiScale(gray, objects, 1.1, 2, 0 | cv::CASCADE_SCALE_IMAGE); //

	cv::namedWindow("img", cv::WINDOW_NORMAL);
	for (size_t i = 0; i < objects.size(); i++)
	{
		cv::rectangle(img, objects[i], cv::Scalar(0, 255, 0), 2);
	}
	// show image
	cv::imshow("img", img);
	cv::waitKey(0);

	return 0;
}