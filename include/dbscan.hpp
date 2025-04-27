// created by Davide Baggio 2122547

#ifndef DBSCAN_HPP
#define DBSCAN_HPP

#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>
#include <random>
#include <unordered_map>

using namespace std;
using namespace cv;

/*
 * Structure that stores clusters and noise
 */
struct cluster_result
{
	vector<vector<Point>> clusters;
	vector<Point> noise;
};

/*
 * Computes the Euclidean distance between two points.
 *
 * Parameters:
 * - a: First point.
 * - b: Second point.
 *
 * Returns:
 * - Euclidean distance as a float.
 */
float euclidean_dist(const Point &a, const Point &b);

/*
 * Finds all points within a specified radius (`eps`) from a given point.
 *
 * Parameters:
 * - points: The dataset of points.
 * - idx: Index of the target point in the dataset.
 * - eps: Radius to consider for neighborhood points.
 *
 * Returns:
 * - Vector of indices representing neighboring points.
 */
vector<int> region_query(const vector<Point> &points, int idx, float eps);

/*
 * Expands a cluster from a seed point by recursively adding neighboring points that meet DBSCAN criteria.
 *
 * Parameters:
 * - points: The dataset of points.
 * - labels: Vector indicating the cluster label for each point.
 * - idx: Index of the seed point.
 * - cluster_id: Identifier for the current cluster.
 * - eps: Radius to consider for neighborhood points.
 * - min_points: Minimum number of points required to form a cluster.
 */
void expand_cluster(const vector<Point> &points, vector<int> &labels, int idx, int cluster_id, float eps, int min_points);

/*
 * Performs DBSCAN clustering on a set of points.
 *
 * Parameters:
 * - points: The dataset of points.
 * - eps: Radius to consider for neighborhood points.
 * - min_points: Minimum number of points required to form a cluster.
 *
 * Returns:
 * - A `cluster_result` containing clustered points and noise points.
 */
cluster_result dbscan(const vector<Point> &points, float eps, int min_points);

/*
 * Identifies and returns the bounding rectangle of the densest cluster found using DBSCAN.
 *
 * Parameters:
 * - points: The dataset of points.
 * - eps: Radius to consider for neighborhood points (default: 30.0).
 * - min_points: Minimum number of points required to form a cluster (default: 5).
 *
 * Returns:
 * - A `Rect` representing the bounding rectangle of the densest cluster.
 * - Returns an empty rectangle if no clusters are found.
 */
Rect get_dense_cluster(const vector<Point> &points, float eps, int min_points);

/*
 * Visualizes clusters, noise points, and the bounding box of the densest cluster using OpenCV.
 *
 * Parameters:
 * - points: The dataset of points.
 * - result: Clustering result obtained from DBSCAN.
 * - densest_box: Bounding rectangle of the densest cluster.
 */
void draw_cluster(const vector<Point> &points, const cluster_result &result, const Rect &densest_box);

/*
 * Generates random synthetic data consisting of multiple clusters and noise points for testing clustering algorithms.
 *
 * Parameters:
 * - num_clusters: Number of clusters to generate (default: 5).
 * - ppoints_per_cluster: Number of points per cluster (default: 100).
 * - noise_points: Number of random noise points to generate (default: 50).
 * - canvas_size: Size of the square canvas area to generate points within (default: 1000).
 *
 * Returns:
 * - Vector of randomly generated points.
 */
vector<Point> generate_random_points(int num_clusters, int ppoints_per_cluster, int noise_points, int canvas_size);

#endif // DBSCAN_HPP