#ifndef MATRIX_H
#define MATRIX_H

#include <iostream>
#include <vector>
#include <fstream>
#include <cassert>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <array>
#include <climits>

using namespace std;

class Matrix {
private:
	double ** m;
	int rows;
	int cols;
	void clearMatrix();
public:
	// constructors
	Matrix();
	Matrix(int rows, int cols, double num=0);
	Matrix(double ** lhs, int r, int c);
	Matrix(const Matrix & lhs);
	Matrix(vector<vector<double>> & lhs);
	Matrix(double * vec, int n, string type = "column", bool remove_vec = true);
	Matrix(string fname);
	
	// getters
	int getRows()const { return rows; };
	int getCols()const { return cols; };
	double* return_col(int i)const;
	double* return_row(int i)const;
	
	// operators
	Matrix operator+(const Matrix & rhs)const;
	Matrix operator-(const Matrix & rhs)const;
	Matrix operator*(const Matrix & rhs)const;
	
	Matrix operator*(double rhs)const;
	Matrix operator+(double rhs)const;
	Matrix operator-(double rhs)const;
	Matrix operator/(double rhs)const;
	
	void operator*=(double rhs);
	void operator+=(double rhs);
	void operator-=(double rhs);
	void operator/=(double rhs);

	void operator*=(const Matrix & rhs);
	void operator+=(const Matrix & rhs);
	void operator-=(const Matrix & rhs);

	bool operator==(const Matrix & rhs)const;
	bool operator!=(const Matrix & rhs)const;
	
	const Matrix & operator=(const Matrix & rhs);
	double* operator[](int i)const { return m[i]; }

	// member utils
	void swap_rows(int index_1, int index_2);
	void swap_cols(int index_1, int index_2);
	Matrix remove_row_col(int r, int c, bool self = false);
	Matrix cofactor_matrix();
	Matrix elewiseSquare(bool self = false);
	Matrix sum(int axis = 0, bool self = false);
	Matrix clipCols(int start, int end, bool self = false);
	Matrix clipRows(int start, int end, bool self = false);
	double frobeniusNorm();
	void assignCol(Matrix m, int col);
	void assignRow(Matrix mat, int row);
	void trimMatrix();
	void printMatrix()const;
	Matrix apply(double(*function)(double));

	// matrix functions
	Matrix T(bool self = false);
	Matrix gaussianElimination(bool self = false);
	Matrix inverse();
	double determinant();
	pair<Matrix, Matrix> qr_decomposition(); // analytical gram-schmidt solution (fails when matrix is singular)
	array<Matrix, 3> singular_value_decomposition(); // numerical approximation with jacobi eigenvalue algorithm
	vector<double> eigs();
	
	~Matrix(); // destructor
};

// free utils
Matrix eye(int n);
Matrix zeros(int n);
Matrix ones(int n);
Matrix concatCols(Matrix a, Matrix b);
Matrix concatRows(Matrix a, Matrix b);
Matrix diag(Matrix & s);
Matrix diag(vector<double> & s);
vector<vector<double>> matrixtoVector(Matrix & m);
double max(Matrix m);

# endif
