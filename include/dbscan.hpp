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

float euclideanDist(const Point &a, const Point &b);
vector<int> regionQuery(const vector<Point> &points, int idx, float eps);
void expandCluster(const vector<Point> &points, vector<int> &labels, int idx, int clusterId, float eps, int minPts);
ClusterResult dbscan(const vector<Point> &points, float eps, int minPts);
Rect getDensestClusterRect(const vector<Point> &points, float eps, int minPts);
void drawClusters(const vector<Point> &points, const ClusterResult &result, const Rect &densestBox);
vector<Point> generateRandomPoints(int numClusters, int pointsPerCluster, int noisePoints, int canvasSize);

#endif // DBSCAN_HPP