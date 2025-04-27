#ifndef DETECTION_HPP
#define DETECTION_HPP

#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

// CONSTANTS
const static std::string base = "./data/";
const static std::string sugar = "004_sugar_box/";
const static std::string mustard = "006_mustard_bottle/";
const static std::string drill = "035_power_drill/";
const static std::string img_path = "test_images/*.jpg";
const static std::string label_path = "labels/";
const static std::string models_path = "models/*_mask.png";
const static std::string negative_path = "negative_images/";
const static std::string cascade = "object_cascade/cascade.xml";

std::string get_filename(std::string path);
bool is_yellow(cv::Vec3b pixel);
bool is_dark(cv::Vec3b pixel);
bool is_white(cv::Vec3b pixel);
bool is_red(cv::Vec3b pixel);
bool is_blue(cv::Vec3b pixel);

float intersection_over_union(cv::Rect rect1, cv::Rect rect2);
void display_performances();

#endif // DETECTION_HPP