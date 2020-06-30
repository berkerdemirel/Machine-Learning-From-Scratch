#ifndef DECISION_TREE_H
#define DECISION_TREE_H

#include <iostream>
#include <vector>
#include <climits>
#include <string>
#include "matrix.h"

using namespace std;

class DecisionNode {
public:
	DecisionNode(int f_i = -1, double t = 0, bool i_l = false, DecisionNode* l_c = NULL, DecisionNode* r_c = NULL, double v=0) :
		feature_index(f_i), threshold(t), is_leaf(i_l), left_child(l_c), right_child(r_c), val(v){};
	int feature_index;
	double threshold;
	bool is_leaf;
	DecisionNode* left_child;
	DecisionNode* right_child;
	double val;
};


class DecisionTree {
public:
	DecisionTree(int m_s_s = 2, double m_i = 1e-5, int m_d = INT_MAX);
	~DecisionTree();
	void fit(const Matrix & X, const Matrix & y);
	Matrix predict_value(const Matrix & x, DecisionNode* ptr = NULL);
	Matrix predict(const Matrix & X);
	void clearTree();

private:
	int min_samples_split;
	double min_impurity;
	int max_depth;
	DecisionNode * root;

	DecisionNode* build_tree(const Matrix & X, const Matrix & y, int current_depth = 0);
	double information_gain(const Matrix & y, const Matrix & y1, const Matrix & y2);
	double majority_vote(const Matrix & y);
	void clearTree(DecisionNode* p);
};
#endif