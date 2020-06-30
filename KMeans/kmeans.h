#ifndef KMEANS_H
#define KMEANS_H

#include <vector>
#include <iostream>
using namespace std;

class KMeans {
public:
	KMeans(int k, int iterations = 1e2);
	void fit(vector<pair<double,double>> & X);
	void printCenters();
private:
	int num_clusters;
	int n_iter;
	vector<pair<double, double>> centers;
};

#endif // !1
