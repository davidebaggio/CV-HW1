// created by Davide Baggio 2122547

#ifndef DETECTION_HPP
#define DETECTION_HPP

#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

// CONSTANTS
const static string base = "./data/";
const static string sugar = "004_sugar_box/";
const static string mustard = "006_mustard_bottle/";
const static string drill = "035_power_drill/";
const static string img_path = "test_images/*.jpg";
const static string label_path = "labels/";
const static string models_path = "models/*_mask.png";
const static string negative_path = "negative_images/";
const static string cascade = "object_cascade/cascade.xml";

/*
 * Extracts and returns the base filename from a given file path, ignoring file extension or additional suffixes.
 *
 * Parameters:
 * - path: The full path of the file.
 *
 * Returns:
 * - The base filename as a string.
 */
string get_filename(string path);

/*
 * Checks if a pixel color is considered yellow based on BGR values.
 *
 * Parameters:
 * - pixel: BGR color value of the pixel.
 *
 * Returns:
 * - True if the pixel is yellow, false otherwise.
 */
bool is_yellow(Vec3b pixel);

/*
 * Checks if a pixel color is considered dark based on BGR values.
 *
 * Parameters:
 * - pixel: BGR color value of the pixel.
 *
 * Returns:
 * - True if the pixel is dark, false otherwise.
 */
bool is_dark(Vec3b pixel);

/*
 * Checks if a pixel color is considered white based on BGR values.
 *
 * Parameters:
 * - pixel: BGR color value of the pixel.
 *
 * Returns:
 * - True if the pixel is white, false otherwise.
 */
bool is_white(Vec3b pixel);

/*
 * Checks if a pixel color is considered red based on BGR values.
 *
 * Parameters:
 * - pixel: BGR color value of the pixel.
 *
 * Returns:
 * - True if the pixel is red, false otherwise.
 */
bool is_red(Vec3b pixel);

/*
 * Checks if a pixel color is considered blue based on BGR values.
 *
 * Parameters:
 * - pixel: BGR color value of the pixel.
 *
 * Returns:
 * - True if the pixel is blue, false otherwise.
 */
bool is_blue(Vec3b pixel);

/*
 * Calculates the Intersection over Union (IoU) between two rectangles.
 *
 * Parameters:
 * - rect1: First rectangle.
 * - rect2: Second rectangle.
 *
 * Returns:
 * - IoU value as a float between 0 and 1.
 */
float intersection_over_union(Rect rect1, Rect rect2);

/*
 * Displays the performance of an object detection model by comparing predicted annotations with ground truth labels.
 *
 * It computes the number of correctly detected objects and the average Intersection over Union (IoU).
 *
 * Behavior:
 * - Reads tested annotations from the 'output' folder.
 * - Reads ground truth labels for different object classes.
 * - Matches files based on filename.
 * - Compares bounding boxes and calculates IoU for each object.
 * - Prints out per-object IoU and overall detection statistics.
 */
void display_performances();

#endif // DETECTION_HPP