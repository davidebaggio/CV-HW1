// created by Davide Baggio 2122547

#ifndef HAAR_DETECTOR_HPP
#define HAAR_DETECTOR_HPP

#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include "detection.hpp"

using namespace std;
using namespace cv;

class haar_detector
{
private:
	Mat test;

	CascadeClassifier cascade_sugar;
	CascadeClassifier cascade_mustard;
	CascadeClassifier cascade_drill;

	vector<vector<Point>> points = vector<vector<Point>>(3);

public:
	/*
	 * Constructor for the `haar_detector` class.
	 *
	 * Loads Haar cascade classifiers for sugar boxes, mustard bottles, and power drills.
	 *
	 * If any cascade file fails to load, an error is printed and the program exits.
	 */
	haar_detector();

	/*
	 * Sets or updates the detected points for a specific category index.
	 *
	 * Parameters:
	 * - points: Vector of points to set.
	 * - index: Category index (0 for sugar, 1 for mustard, 2 for drill).
	 *
	 * Behavior:
	 * - Validates the index before inserting.
	 * - If the index is invalid, prints an error message.
	 */
	void set_points(vector<Point> points, size_t index);

	/*
	 * Performs object detection on the given image using Haar cascade classifiers.
	 *
	 * Parameters:
	 * - img: Input image on which detection is performed.
	 *
	 * Behavior:
	 * - Converts the input image to grayscale and equalizes the histogram.
	 * - Detects objects for each category (sugar, mustard, drill).
	 * - Saves the center points of detected bounding boxes into the `points` array.
	 */
	void compute_detection(Mat img);

	/*
	 * Retrieves the detected points for each object category.
	 *
	 * Returns:
	 * - A vector of vectors containing points for each detected object category (sugar, mustard, drill).
	 */
	vector<vector<Point>> get_points();

	/*
	 * Displays detected points on separate copies of the input image.
	 *
	 * Behavior:
	 * - Draws a green circle around each detected center point.
	 * - Displays the results separately for each category (sugar, mustard, drill).
	 * - If no points are found for a category, prints a message and skips display.
	 */
	void display_points();
};

#endif // HAAR_DETECTOR_HPP