// Created by: Zoren Martinez mat. 2123873
#ifndef ORB_DETECTOR_HPP
#define ORB_DETECTOR_HPP

#include <iostream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include "opencv2/features2d.hpp"
#include <filesystem>
#include <fstream>
#include <regex>

using namespace cv;
using namespace std;
namespace fs = filesystem;

class orb_detector
{
private:
	
	// Test image
	Mat test;

	// Different directories for the models
	vector<string> models_path = {
		"data/004_sugar_box/models",
		"data/006_mustard_bottle/models",
		"data/035_power_drill/models"};
	
	// Regex pattern to match the model images
	string pattern = R"(view.*color\.png)";

	// Regex pattern to match the model masks
	string pattern_mask = R"(view.*mask\.png)";
	Ptr<ORB> orb = ORB::create();

	// Vector of points for each model
	// 0: sugar, 1: mustard, 2: drill
	vector<vector<Point>> points = vector<vector<Point>>(3);

	// Vector descriptors for each model
	// 0: sugar, 1: mustard, 2: drill
	vector<vector<Mat>> model_descriptors = vector<vector<Mat>>(3);

	
	
	/*
	 * 
	 * Helper function to compute the median of a vector of doubles
	 */
	double compute_median(vector<double> values);
	
	/*
	* 
	* Helper function to get matches between an image model descriptors and image test descriptors
	*/
	vector<DMatch> get_matches(const Mat &model_descriptors, const Mat &test_descriptors);
	
	/*
	* 
	* Helper function to save the points of the best matches
	*
	*/
	void save_points(vector<DMatch> matches, vector<KeyPoint> test_keypoints, int category);

public:

	/*
	 * Constructor
	 *
	 *
	 * Behavior:
	 * - Initializes the ORB detector and reads the model images from the dataset folders	
	 * - Reads every model of the objects from the the dataset folders
	 * - Check if the file name matches the regex pattern
	 * - Compute all the descriptors for the each model images of the dataset
	 */
	orb_detector();
	
	
	/*
	 * Returns the points
	 */ 
	vector<vector<Point>> get_points();


	/*
	 * Returns the points with a percentage of the best matches
	 */ 
	vector<vector<Point>> get_points(float perc);
	
	/*
	 * Parameters:
	 * - img: test image
	 *
	 * Behavior:
	 * - Computes the detection between the model descriptors and the test image descriptors
	 * - Compute the descriptors for the test image
	 * - For each model image, get the best match, which are the most numerous matches from a single image model
	 * - The maximum number of matches is considered the best match
	 * - Sort matches by distance to prioritize the most reliable matches (smallest distance first)
	 * 
	 */ 
	void compute_detection(Mat img);
	
	/*
	 * 
	 * Behavior:
	 * - Displays all the points of the best matches
	 *
	 */
	void display_points();
	
	/*
	 * 
	 * Parameters:
	 * - perc: percentage of the best matches to display
	 *
	 * Behavior:
	 * - Displays the points of the best matches with a percentage of the best matches
	 *
	 */
	void display_points(float perc);
};

#endif // ORB_DETECTOR_HPP