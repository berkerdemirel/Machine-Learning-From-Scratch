#include "DecisionTree.h"
#include <unordered_set>
#include <map>
#include <cmath>

// constructor
DecisionTree::DecisionTree(int m_s_s, double m_i, int m_d) : min_samples_split(m_s_s), min_impurity(m_i), max_depth(m_d) {
	root = NULL;
}

// fits to the train data
void DecisionTree::fit(const Matrix & X, const Matrix & y) {
	root = build_tree(X, y);
}

// returns the unique values of given column vector together with the frequency map(optional)
Matrix unique(const Matrix & m, map<double, int> & freqs=map<double,int>(), bool is_freq=false) {
	unordered_set<double> set;
	freqs.clear();
	for (int i = 0; i < m.getRows(); i++) {
		if (set.find(m[i][0]) == set.end()) {
			set.insert(m[i][0]);
		}
		if (is_freq) {
			freqs[m[i][0]]++;
		}
	}
	Matrix res(set.size(), 1);
	int index = 0;
	for (auto x : set) {
		res[index++][0] = x;
	}
	return res;
}

// separetes the given data(Xy) with respect to the jth column's value(whether equal to threshold)
// to Xy1(left branch) and Xy2(right branch)
void divide_on_feature(const Matrix & Xy, const int j, Matrix & Xy1, Matrix & Xy2, double threshold, int row_xy1) {
	Xy1 = Matrix(row_xy1, Xy.getCols());
	Xy2 = Matrix(Xy.getRows()-row_xy1, Xy.getCols());
	int index_1 = 0, index_2 = 0;
	for (int i = 0; i < Xy.getRows(); i++) {
		if (Xy[i][j] == threshold) {
			Xy1.assignRow(Matrix(Xy.return_row(i), Xy.getCols(), "row"), index_1++);
		}
		else {
			Xy2.assignRow(Matrix(Xy.return_row(i), Xy.getCols(), "row"), index_2++);
		}
	}
}

// builds the decision tree
DecisionNode* DecisionTree::build_tree(const Matrix & X, const Matrix & y, int current_depth) {
	double largest_imp = 0; // largest impurity (information gain)
	DecisionNode* res; // resulting node
	Matrix Xy = concatCols(X, y); // concatenate labels to dataset
	int n_samples = X.getRows(); // number of samples
	int n_features = X.getCols(); // number of attributes
	pair<int, double> best_criteria; // best column, with specific value to separate
	vector<Matrix> best_sets(4); // separated sets
	map<double,int> freqs; // frequency map
	if (n_samples >= min_samples_split && current_depth <= max_depth){ // pre conditions to split 
		for (int i = 0; i < n_features; i++) { // for each feature
			
			Matrix feature_values = Matrix(X.return_col(i), n_samples);
			Matrix unique_vals = unique(feature_values, freqs, true);
			if (unique_vals.getRows() == 1) { // if there will be no information gain
				continue;
			}
			Matrix Xy1, Xy2;
			for (int j = 0; j < unique_vals.getCols(); j++) { // for each unique val try splitting w.r.t it
				double & threshold = unique_vals[0][j];
				int row_xy1 = freqs[threshold];
				divide_on_feature(Xy, i, Xy1, Xy2, threshold, row_xy1);
				if (Xy1.getCols() && Xy2.getCols()) {
					Matrix y1 = Xy1.clipCols(n_features, n_features + y.getCols()-1);
					Matrix y2 = Xy2.clipCols(n_features, n_features + y.getCols()-1);
					double impurity = information_gain(y, y1, y2); // calculate information gain
					if (impurity > largest_imp) { // if it is a better division
						largest_imp = impurity;
						best_criteria = make_pair(i, threshold);
						best_sets[0] = Xy1.clipCols(0, n_features-1); // left X
						best_sets[1] = Xy1.clipCols(n_features, n_features+y.getCols()-1); // left y
						best_sets[2] = Xy2.clipCols(0, n_features-1); // right X
						best_sets[3] = Xy2.clipCols(n_features, n_features + y.getCols()-1); // right y

					}
				}

			}
		}
	}
	// check if we need to go deeper
	if (largest_imp > min_impurity) {
		DecisionNode* left_tree = build_tree(best_sets[0], best_sets[1], current_depth + 1);
		DecisionNode* right_tree = build_tree(best_sets[2], best_sets[3], current_depth + 1);
		res = new DecisionNode(best_criteria.first, best_criteria.second, false, left_tree, right_tree);
		return res;
	}

	// if we are at leaf
	double leaf_value = majority_vote(y); // value is decided by majority voting
	/*
	Matrix temp = y;
	temp.T().printMatrix();
	cout << leaf_value << endl << endl;
	*/
	res = new DecisionNode(-1, leaf_value, true, NULL, NULL, leaf_value);
	return res;
}


Matrix DecisionTree::predict_value(const Matrix & x, DecisionNode* ptr) {
	if (!ptr) { // if ptr is null start from root
		ptr = root;
	}
	if (ptr->is_leaf) { // if it is a leaf return its value
		return Matrix(1,1, ptr->val);
	}
	double feature_val = x[0][ptr->feature_index];
	if (feature_val == ptr->threshold) { // search in left branch
		return predict_value(x, ptr->left_child);
	}
	return predict_value(x, ptr->right_child); // search in right branch
}


Matrix DecisionTree::predict(const Matrix & X) {
	Matrix res(X.getRows(), 1);
	for (int i = 0; i < res.getRows(); i++) {
		Matrix sample = Matrix(X.return_row(i), X.getCols(), "row"); // for each sample
		res[i][0] = predict_value(sample)[0][0]; // predict value
	}
	return res;
}

// count the number of l's in column vector y
int count(const Matrix & y, double l) {
	int res = 0;
	for (int i = 0; i < y.getRows(); i++) {
		if (y[i][0] == l) {
			res++;
		}
	}
	return res;
}

// calculates the entropy of column vector y
double entropy(const Matrix & y) {
	Matrix unique_labels = unique(y);
	double res = 0;
	for (int i = 0; i < unique_labels.getRows(); i++) {
		double label = unique_labels[i][0];
		int cnt = count(y, label);
		double p = (double)cnt / y.getRows();
		res += -p*log2(p);
	}
	return res;
}

// calculates the information gain w.r.t parent, left child and right child
double DecisionTree::information_gain(const Matrix & y, const Matrix & y1, const Matrix & y2) {
	double p = (double)y1.getRows() / y.getRows();
	double ent = entropy(y);
	double info_gain = ent - p*entropy(y1) - (1 - p)*entropy(y2);
	return info_gain;
}


// majority voting mechanism
double DecisionTree::majority_vote(const Matrix & y) {
	Matrix y_unique = unique(y);
	double most_common;
	int max_count = 0;
	for (int i = 0; i < y_unique.getCols(); i++) {
		int cnt = count(y, y_unique[i][0]);
		if (cnt > max_count) {
			max_count = cnt;
			most_common = y_unique[i][0];
		}
	}
	return most_common;
}


// deallocates the dynamic memory
void DecisionTree::clearTree(DecisionNode* p) {
	if (p) {
		clearTree(p->left_child);
		clearTree(p->right_child);
		delete p;
	}
}

void DecisionTree::clearTree() {
	clearTree(root);
}


// destructor
DecisionTree::~DecisionTree() {
	clearTree();
}

