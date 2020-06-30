#include "LinearRegression.h"

void LinearRegression::fit(Matrix & X, Matrix & y) {
	Matrix col_ones = Matrix(X.getRows(), 1, 1);
	Matrix tmp = concatCols(X, col_ones);
	weights = (tmp.T()*tmp).inverse()*tmp.T()*y; // (X_T*X)^(-1) * X_T * y
}


Matrix LinearRegression::predict(Matrix & X) {
	Matrix res(X.getRows(), 1);
	Matrix col_ones = Matrix(X.getRows(), 1, 1);
	Matrix tmp = concatCols(X, col_ones);
	for (int i = 0; i < tmp.getRows(); i++) {
		Matrix row = Matrix(tmp.return_row(i), tmp.getCols(), "row");
		res[i][0] = (weights.T()*row.T())[0][0];
	}
	return res;
}
