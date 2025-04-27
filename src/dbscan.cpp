// created by Davide Baggio 2122547

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
					int idx, int cluster_id, float eps, int min_points)
{
	vector<int> seeds = region_query(points, idx, eps);
	if (seeds.size() < min_points)
	{
		labels[idx] = -1;
		return;
	}

	labels[idx] = cluster_id;

	size_t i = 0;
	while (i < seeds.size())
	{
		int current = seeds[i];
		if (labels[current] == -1)
		{
			labels[current] = cluster_id;
		}
		else if (labels[current] == 0)
		{
			labels[current] = cluster_id;
			vector<int> result = region_query(points, current, eps);
			if (result.size() >= min_points)
			{
				seeds.insert(seeds.end(), result.begin(), result.end());
			}
		}
		++i;
	}
}

cluster_result dbscan(const vector<Point> &points, float eps, int min_points)
{
	vector<int> labels(points.size(), 0);
	int cluster_id = 1;

	for (int i = 0; i < points.size(); ++i)
	{
		if (labels[i] != 0)
			continue;
		expand_cluster(points, labels, i, cluster_id, eps, min_points);
		if (labels[i] == cluster_id)
			cluster_id++;
	}

	unordered_map<int, vector<Point>> cluster_map;
	vector<Point> noise;

	for (int i = 0; i < labels.size(); ++i)
	{
		if (labels[i] == -1)
		{
			noise.push_back(points[i]);
		}
		else
		{
			cluster_map[labels[i]].push_back(points[i]);
		}
	}

	cluster_result result;
	for (const auto &entry : cluster_map)
	{
		result.clusters.push_back(entry.second);
	}
	result.noise = noise;

	return result;
}

Rect get_dense_cluster(const vector<Point> &points, float eps = 30.0, int min_points = 5)
{
	cluster_result result = dbscan(points, eps, min_points);

	if (result.clusters.empty())
		return Rect();

	size_t max_size = 0;
	vector<Point> *densest = nullptr;

	for (auto &cluster : result.clusters)
	{
		if (cluster.size() > max_size)
		{
			max_size = cluster.size();
			densest = &cluster;
		}
	}

	return boundingRect(*densest);
}

void draw_cluster(const vector<Point> &points, const cluster_result &result, const Rect &densest_box)
{
	Mat canvas(1200, 1200, CV_8UC3, Scalar(255, 255, 255));

	RNG rng(12345);

	for (const auto &cluster : result.clusters)
	{
		Scalar color(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
		for (const auto &pt : cluster)
		{
			circle(canvas, pt, 3, color, -1);
		}
	}

	for (const auto &pt : result.noise)
	{
		circle(canvas, pt, 3, Scalar(0, 0, 0), -1);
	}

	rectangle(canvas, densest_box, Scalar(0, 0, 255), 2);

	imshow("Clusters and Densest Area", canvas);
	waitKey(0);
}

vector<Point> generate_random_points(int num_clusters = 5, int points_per_cluster = 100, int noise_points = 50, int canvas_size = 1000)
{
	vector<Point> points;
	mt19937 rng(time(NULL));
	uniform_int_distribution<int> center_dist(100, canvas_size - 100);
	normal_distribution<float> cluster_spread(0, 30);
	uniform_int_distribution<int> noise_dist(0, canvas_size);

	for (int c = 0; c < num_clusters; ++c)
	{
		int centerX = center_dist(rng);
		int centerY = center_dist(rng);
		for (int i = 0; i < points_per_cluster; ++i)
		{
			int x = static_cast<int>(centerX + cluster_spread(rng));
			int y = static_cast<int>(centerY + cluster_spread(rng));
			points.emplace_back(x, y);
		}
	}

	for (int i = 0; i < noise_points; ++i)
	{
		points.emplace_back(noise_dist(rng), noise_dist(rng));
	}

	return points;
}