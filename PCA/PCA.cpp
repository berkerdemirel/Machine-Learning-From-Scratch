#include "PCA.h"

Matrix PCA::principle_component_analysis(Matrix & X) {
	auto arr = X.singular_value_decomposition();
	Matrix & V_T = arr[2].clipCols(0,dimension-1);
	cout << "v size: " << V_T.getRows() << " " << V_T.getCols() << endl;
	cout << "X size: " << X.getRows() << " " << X.getCols() << endl;
	Matrix pca = X*V_T;
	return pca;
}
