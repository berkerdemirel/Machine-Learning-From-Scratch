#ifndef PCA_H
#define PCA_H
#include "matrix.h"

class PCA {
public:
	PCA(int dim) { dimension = dim; };
	Matrix principle_component_analysis(Matrix & X);
private:
	int dimension;
};

#endif