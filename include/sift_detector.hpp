// Created by: Pivotto Francesco mat. 2158296
#ifndef SIFT_DETECTOR_HPP
#define SIFT_DETECTOR_HPP

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
using namespace filesystem;

class sift_detector
{
	private:

		// Test image
		Mat img_test;

		//SIFT Detector
		Ptr<SIFT> sift;

		// Regex pattern to match the model images
		string pattern = R"(view.*color\.png)";

		// Different directories for the models
		vector<string> models_path = {
			"data/004_sugar_box/models",
			"data/006_mustard_bottle/models",
			"data/035_power_drill/models"};


		// Vector of points for each model
		// 0: sugar, 1: mustard, 2: drill
		vector<vector<Point>> points = vector<vector<cv::Point>>(3);

		// Vector descriptors for each model
		// 0: sugar, 1: mustard, 2: drill
		vector<vector<Mat>> model_descriptors = vector<vector<Mat>>(3);
		
		/*
		* Extracts and stores the SIFT descriptors for each object model using the provided masks.
		*
		* Parameters:
		* - None (the function uses the internal `models_path` and `pattern` attributes of the class).
		*
		* Behavior:
		* - Iterates through the directories listed in `models_path`, each corresponding to an object category.
		* - For each file matching the specified `pattern`, it loads the model image and the associated segmentation mask.
		* - If either the model image or the mask cannot be opened, an error is printed and the file is skipped.
		* - The model image is optimized (e.g., preprocessing) before feature extraction.
		* - SIFT keypoints are detected and descriptors are computed using the mask to focus on relevant regions.
		* - The descriptors are stored in the `model_descriptors` vector, organized by object category.
		* - If an exception occurs during directory traversal or file processing, it is caught and logged.
		*
		* Returns:
		* - None (the function modifies the internal `model_descriptors` vector).
		*
		* Notes:
		* - Requires that `models_path` is correctly initialized and that `pattern` matches the desired model images.
		* - Assumes that for each model image, a corresponding mask file exists with the expected naming convention.
		*/
		void get_model_descriptors();
		
		/*
		* Optimizes an input image by converting it to grayscale and applying histogram equalization.
		*
		* Parameters:
		* - src: A reference to the input image (cv::Mat) to be optimized. It will be modified in place.
		*
		* Returns:
		* - None (the function modifies the input `src` directly).
		*
		* Behavior:
		* - Converts the input color image from BGR to grayscale.
		* - Applies histogram equalization to enhance the contrast of the grayscale image.
		* - Replaces the original image with the processed (equalized) grayscale image.
		*
		*/
		void optimize_image(Mat &src);
		
		/*
		* Saves the points corresponding to the matched keypoints between the test image and the model.
		*
		* Parameters:
		* - matches: A vector of DMatch objects representing the matches between keypoints in the test image and the model.
		* - img_kpt: A vector of KeyPoint objects representing the detected keypoints in the test image.
		* - category: An integer representing the object category for which the keypoints are being saved.
		*
		* Returns:
		* - None (the function modifies the internal `points` map).
		*
		* Behavior:
		* - Iterates over the matches and retrieves the corresponding keypoints' positions from the test image (`img_kpt`).
		* - For each match, the corresponding point (from the `trainIdx` of the `DMatch`) is added to the `matches_points` vector.
		* - The `matches_points` vector is then stored in the `points` map, using the specified `category` as the key.
		* - This ensures that the points are categorized by the object model they belong to.
		*/
		void save_points(vector<cv::DMatch> &matches, vector<KeyPoint> &img_kpt, int category);
		
		/*
		* Computes the good matches between the model descriptors and the image descriptors using the FLANN-based matcher.
		*
		* Parameters:
		* - model_desc: A cv::Mat containing the descriptors of the object model.
		* - img_desc: A cv::Mat containing the descriptors of the test image.
		*
		* Returns:
		* - A vector of DMatch objects representing the good matches between the model and image descriptors.
		*
		* Behavior:
		* - The function uses the FLANN-based matcher to find the k-nearest neighbors (k=2) for each model descriptor.
		* - For each pair of matches, it applies the Lowe's ratio test to filter out bad matches:
		*     - If the distance of the closest match (m[0]) is significantly smaller than that of the second closest match (m[1]), it is considered a good match.
		*     - The ratio threshold is set to 1.0, meaning that the first match should be at least twice as good as the second one.
		* - The function returns a vector of the filtered "good matches".
		*
		* Notes:
		* - The FLANN-based matcher is used for fast approximate nearest neighbor search.
		* - The distance ratio used for filtering matches is hardcoded to 0.82.
		*/
		vector<DMatch> get_matches(const Mat &model_desc, const Mat &img_desc);

	public:
		
		/*
		* Constructor for the sift_detector class.
		*
		* Initializes the SIFT detector and computes the descriptors for the object models.
		*
		* Parameters:
		* - None
		*
		* Returns:
		* - None
		*
		* Behavior:
		* - The constructor initializes the SIFT detector by creating an instance of the SIFT algorithm using `SIFT::create()`.
		* - It then calls the `get_model_descriptors()` function to compute and store the descriptors for the object models.
		*
		* Notes:
		* - This constructor is used to set up the detector and prepare the model descriptors for further matching in the object recognition process.
		* - The `get_model_descriptors()` function is called automatically as part of the construction process to ensure that the descriptors are ready.
		*/
		sift_detector();

		/*
		* Returns the points associated with each object category based on the detected matches.
		*
		* Parameters:
		* - None
		*
		* Returns:
		* - A vector of vectors of `cv::Point` objects, where each inner vector contains points corresponding to a particular object category.
		*
		* Behavior:
		* - This function simply returns the `points` member variable, which stores the matched points for each object category in the form of a vector of vectors.
		* - The `points` vector was populated during the object detection process, where each category corresponds to the points found in the test image that match a model.
		*/
		vector<vector<Point>> get_points();

		/*
		* Returns a subset of the points associated with each object category, based on a specified percentage.
		*
		* Parameters:
		* - perc: A floating-point value between 0 and 1 representing the percentage of points to be returned for each category.
		* 
		* Returns:
		* - A vector of vectors of `cv::Point` objects, where each inner vector contains a subset of points corresponding to a particular object category.
		* 
		* Behavior:
		* - For each category, the function selects the top `perc` percentage of points (based on the order they were added) and stores them in a new vector.
		* - The selected points are then returned as a vector of vectors, where each inner vector corresponds to a category and contains the filtered points.
		* - The function ensures that only the specified percentage of points are included in the result for each category.
		*/
		vector<vector<Point>> get_points(float perc);

		/*
		* Computes the object detection for a given test image using the SIFT algorithm.
		*
		* Parameters:
		* - img: A cv::Mat containing the test image on which object detection is to be performed.
		*
		* Returns:
		* - None
		*
		* Behavior:
		* - The function starts by cloning the input test image and applies optimization (grayscale conversion and histogram equalization) using the `optimize_image()` method.
		* - The function then uses the SIFT algorithm to detect keypoints in the optimized test image and computes the corresponding descriptors.
		* - If no descriptors are found for the test image, an error message is printed, and the function terminates early.
		* - The function iterates through each model's descriptors (stored in `model_descriptors`), comparing them to the test image descriptors using the `get_matches()` function.
		* - For each model, the function determines the number of matches and selects the model with the highest number of matches to the test image.
		* - If no matches are found for a model or if the matches are below a minimum threshold, an error message is printed, and the function continues with the next model.
		* - The selected "winning" matches (those with the highest number of matches) are stored in `winning_matches`.
		*
		* Notes:
		* - The `get_matches()` function is used to find matches between the model and test image descriptors, and it applies Lowe's ratio test to filter the matches.
		* - This function is used to determine which object model best matches the test image based on the number of matching keypoints.
		*/
		void compute_detection(Mat img_test);

		/*
		* Displays all the points associated with each object category by calling the display_points function with a percentage of 100%.
		*
		* Parameters:
		* - None
		* 
		* Returns:
		* - None
		* 
		* Behavior:
		* - This function is a wrapper that calls the `display_points` method with a default parameter value of `1.0`, which indicates that all the points for each category should be displayed.
		* - It simplifies the function call for cases where the user wants to display all the points without specifying a percentage.
		*/
		void display_points();

		/*
		* Displays the points corresponding to each object category (sugar box, mustard bottle, and power drill) on the test image.
		* The points are displayed with a green circle, and the percentage of points to be displayed is controlled by the parameter `perc`.
		*
		* Parameters:
		* - perc: A float value between 0.0 and 1.0, representing the percentage of points to display for each category.
		*         (1.0 means all points, while lower values display only a subset of points).
		* 
		* Returns:
		* - None
		* 
		* Behavior:
		* - A clone of the test image (`img_test`) is created for each category (sugar box, mustard bottle, and power drill).
		* - For each object category, a subset of points is selected based on the `perc` parameter, and circles are drawn around these points on the cloned image.
		* - If no points are found for a category, an error message is printed, and that category is skipped.
		* - The processed images are displayed in separate windows for each object category, with the points highlighted.
		* - The function waits for a key press before closing the display windows.
		*/
		void display_points(float perc);
};

#endif // SIFT_DETECTOR_HPP