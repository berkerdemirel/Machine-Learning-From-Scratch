#include "SVM.h"

Matrix multiply(Matrix m1, Matrix m2) {
	assert(m1.getRows() == m2.getRows() && m1.getCols() == m2.getCols());
	Matrix res(m1.getRows(), m1.getCols());
	for (int i = 0; i < m1.getRows(); i++) {
		for (int j = 0; j < m1.getCols(); j++) {
			res[i][j] = m1[i][j] * m2[i][j];
		}
	}
	return res;
}

void SVM::fit(Matrix & X, Matrix & y) {
	Matrix X_b = concatCols(X, Matrix(X.getRows(), 1, 1));
	W = Matrix(X.getCols()+1, 1); //initialize
	for (int i = 1; i <= n_epochs; i++) {
		Matrix y_hat = X_b*W;
		Matrix prod = multiply(y, y_hat);
		for (int j = 0; j < prod.getRows(); j++) {
			if (prod[j][0] >= 1) {
				double cost = 0;
				W -= (W*(2 / i))*lr;
			}
			else {
				double cost = 1 - prod[j][0];
				Matrix error_train = Matrix(X_b.return_row(j), X_b.getCols(), "column");
				//error_train.printMatrix();
				W += (error_train*y[j][0] - W * 2 / i) * lr;
			}
			//W.printMatrix();
			//cout << endl;
		}
	}
}

double sign(double d) {
	return d > 0 ? 1 : -1;
}

Matrix sign(Matrix & x) {
	Matrix res(x.getRows(), x.getCols());
	for (int i = 0; i < x.getRows(); i++) {
		res[i][0] = sign(x[i][0]);
	}
	return res;
}

Matrix SVM::predict(Matrix & X) {
	Matrix temp = concatCols(X, Matrix(X.getRows(), 1, 1));
	auto res = sign(temp*W);
	return res;
}
