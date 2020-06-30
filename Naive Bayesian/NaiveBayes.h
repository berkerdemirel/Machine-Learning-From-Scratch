#ifndef NAIVE_BAYES_H
#define NAIVE_BAYES_H

#include "matrix.h"

class NaiveBayes {
public:
	NaiveBayes() {};
	void fit(Matrix & train_X, Matrix & train_y);
	Matrix predict(Matrix & test_X);

private:
	double calculate_likelihood(double mean, double var, double x);
	double calculate_prior(double c);
	double classify(Matrix & x);
	vector<vector<pair<double, double>>> parameters; // first is mean, second is variance
	Matrix X;
	Matrix y;
	Matrix classes;
};

#endif 
