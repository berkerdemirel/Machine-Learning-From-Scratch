#include "kmeans.h"
#include <cmath>

double euclidean_distance(pair<double, double> & p1, pair<double, double> & p2) {
	return sqrt(pow((p1.first - p2.first), 2) + pow((p1.second - p2.second), 2));
}

KMeans::KMeans(int k, int iterations) :num_clusters(k), n_iter(iterations) {
	centers = vector<pair<double, double>>(num_clusters);
	for (auto & x : centers) {
		x = make_pair(rand() % 5 / 6.0 + rand() % 3, rand() % 5 / 6.0 + rand() % 3);
	}
}

void KMeans::fit(vector<pair<double, double>> & X) {
	vector<vector<int>> indices(num_clusters);
	for (int i = 0; i < n_iter; i++) {
		for (int j = 0; j < X.size(); j++) { // for each point
			auto & point = X[j];
			int class_id = -1;
			double min_dist = 1e6;
			for (int k = 0; k < num_clusters; k++) {
				double cur_dist = euclidean_distance(point, centers[k]);
				if (cur_dist < min_dist) {
					min_dist = cur_dist;
					class_id = k;
				}
			}
			indices[class_id].push_back(j);
		}
		for (int j = 0; j < indices.size(); j++) {
			auto & class_point_indices = indices[j];
			pair<double, double> avg = make_pair(0, 0);
			for (int k = 0; k<class_point_indices.size(); k++) {
				auto point = X[class_point_indices[k]];
				avg.first += point.first;
				avg.second+= point.second;
			}
			if (class_point_indices.size() > 0) {
				avg.first /= class_point_indices.size();
				avg.second /= class_point_indices.size();
			}
			centers[j] = avg;
		}
	}
}

void KMeans::printCenters() {
	for (auto point : centers) {
		cout << point.first << " " << point.second << endl;
	}
}
