#ifndef SVM_H
#define SVM_H

#include "matrix.h"
using namespace std;

class SVM {

public:
	SVM(int e=1000, double alpha=0.0003):n_epochs(e),lr(alpha){};
	void fit(Matrix & X, Matrix & y);
	Matrix predict(Matrix & X);

private:
	double hinge_loss(Matrix & x, Matrix & y);
	Matrix hinge_gradient(Matrix & X_batch, Matrix & y_batch);
	void SGD(Matrix & X, Matrix & y);
	Matrix W;
	Matrix B;
	int n_epochs;
	double lr;
};

#endif // !SVM_H
