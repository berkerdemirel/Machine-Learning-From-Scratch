#pragma once
#include <vector>

using namespace std;

class KnnClassifier {
public:
	KnnClassifier(int k_ = 3, bool normalize_ = false);
	void fit(vector<vector<double>> & trainset, vector<int> & label);
	vector<int> test(vector<vector<double>> & testset);
private:
	int k;
	bool normalize;
	vector<vector<double>> data;
	vector<int> labels;
};