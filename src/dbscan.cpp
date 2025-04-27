#include "dbscan.hpp"

float euclidean_dist(const Point &a, const Point &b)
{
	return sqrt((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y));
}

vector<int> region_query(const vector<Point> &points, int idx, float eps)
{
	vector<int> neighbors;
	for (int i = 0; i < points.size(); ++i)
	{
		if (euclidean_dist(points[idx], points[i]) <= eps)
		{
			neighbors.push_back(i);
		}
	}
	return neighbors;
}

void expand_cluster(const vector<Point> &points, vector<int> &labels,
					int idx, int clusterId, float eps, int minPts)
{
	vector<int> seeds = region_query(points, idx, eps);
	if (seeds.size() < minPts)
	{
		labels[idx] = -1; // noise
		return;
	}

	labels[idx] = clusterId;

	size_t i = 0;
	while (i < seeds.size())
	{
		int current = seeds[i];
		if (labels[current] == -1)
		{
			labels[current] = clusterId;
		}
		else if (labels[current] == 0)
		{
			labels[current] = clusterId;
			vector<int> result = region_query(points, current, eps);
			if (result.size() >= minPts)
			{
				seeds.insert(seeds.end(), result.begin(), result.end());
			}
		}
		++i;
	}
}

ClusterResult dbscan(const vector<Point> &points, float eps, int minPts)
{
	vector<int> labels(points.size(), 0); // 0 = unvisited
	int clusterId = 1;

	for (int i = 0; i < points.size(); ++i)
	{
		if (labels[i] != 0)
			continue;
		expand_cluster(points, labels, i, clusterId, eps, minPts);
		if (labels[i] == clusterId)
			clusterId++;
	}

	unordered_map<int, vector<Point>> clustersMap;
	vector<Point> noise;

	for (int i = 0; i < labels.size(); ++i)
	{
		if (labels[i] == -1)
		{
			noise.push_back(points[i]);
		}
		else
		{
			clustersMap[labels[i]].push_back(points[i]);
		}
	}

	ClusterResult result;
	for (const auto &entry : clustersMap)
	{
		result.clusters.push_back(entry.second);
	}
	result.noise = noise;

	return result;
}

Rect get_dense_cluster(const vector<Point> &points, float eps = 30.0, int minPts = 5)
{
	ClusterResult result = dbscan(points, eps, minPts);

	if (result.clusters.empty())
		return Rect();

	size_t maxSize = 0;
	vector<Point> *densest = nullptr;

	for (auto &cluster : result.clusters)
	{
		if (cluster.size() > maxSize)
		{
			maxSize = cluster.size();
			densest = &cluster;
		}
	}

	return boundingRect(*densest);
}

void drawClusters(const vector<Point> &points, const ClusterResult &result, const Rect &densestBox)
{
	Mat canvas(1200, 1200, CV_8UC3, Scalar(255, 255, 255));

	RNG rng(12345);

	// Draw clusters
	for (const auto &cluster : result.clusters)
	{
		Scalar color(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
		for (const auto &pt : cluster)
		{
			circle(canvas, pt, 3, color, -1);
		}
	}

	// Draw noise
	for (const auto &pt : result.noise)
	{
		circle(canvas, pt, 3, Scalar(0, 0, 0), -1);
	}

	// Draw densest cluster rectangle
	rectangle(canvas, densestBox, Scalar(0, 0, 255), 2);

	imshow("Clusters and Densest Area", canvas);
	waitKey(0);
}

vector<Point> generateRandomPoints(int numClusters = 5, int pointsPerCluster = 100, int noisePoints = 50, int canvasSize = 1000)
{
	vector<Point> points;
	mt19937 rng(time(NULL)); // fixed seed for reproducibility
	uniform_int_distribution<int> centerDist(100, canvasSize - 100);
	normal_distribution<float> clusterSpread(0, 30); // spread of points in each cluster
	uniform_int_distribution<int> noiseDist(0, canvasSize);

	// Generate clustered points
	for (int c = 0; c < numClusters; ++c)
	{
		int centerX = centerDist(rng);
		int centerY = centerDist(rng);
		for (int i = 0; i < pointsPerCluster; ++i)
		{
			int x = static_cast<int>(centerX + clusterSpread(rng));
			int y = static_cast<int>(centerY + clusterSpread(rng));
			points.emplace_back(x, y);
		}
	}

	// Generate noise points
	for (int i = 0; i < noisePoints; ++i)
	{
		points.emplace_back(noiseDist(rng), noiseDist(rng));
	}

	return points;
}

/* int main()
{
	// random points
	vector<Point> points = generateRandomPoints();

	float eps = 100.0f;
	int minPts = 3;

	ClusterResult result = dbscan(points, eps, minPts);
	Rect denseRect = get_dense_cluster(points, eps, minPts);

	drawClusters(points, result, denseRect);

	cout << "Densest bounding box: " << denseRect << endl;
	return 0;
} */
