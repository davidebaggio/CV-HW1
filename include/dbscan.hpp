#ifndef DBSCAN_HPP
#define DBSCAN_HPP

#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>
#include <random>
#include <unordered_map>

using namespace std;
using namespace cv;

struct ClusterResult
{
	vector<vector<Point>> clusters;
	vector<Point> noise;
};

float euclidean_dist(const Point &a, const Point &b);
vector<int> region_query(const vector<Point> &points, int idx, float eps);
void expand_cluster(const vector<Point> &points, vector<int> &labels, int idx, int clusterId, float eps, int minPts);
ClusterResult dbscan(const vector<Point> &points, float eps, int minPts);
Rect get_dense_cluster(const vector<Point> &points, float eps, int minPts);
void draw_cluster(const vector<Point> &points, const ClusterResult &result, const Rect &densestBox);
vector<Point> generate_random_points(int numClusters, int pointsPerCluster, int noisePoints, int canvasSize);

#endif // DBSCAN_HPP