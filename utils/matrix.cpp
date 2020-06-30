#include "matrix.h"


Matrix::Matrix() {
	rows = 0;
	cols = 0;
	m = NULL;
}


Matrix::Matrix(int r, int c, double num) { // default->zeros
	rows = r;
	cols = c;
	m = new double*[r];
	for(int i=0; i<r; i++){
		m[i] = new double[c];
	}
	for (int i = 0; i < r; i++) {
		for (int j = 0; j < c; j++) {
			m[i][j] = num;
		}
	}
}

Matrix::Matrix(double ** rhs, int r, int c) { // shallow copy
	rows = r;
	cols = c;
	m = rhs;
}

Matrix::Matrix(vector<vector<double>> & rhs) { // 2d vec -> matrix
	rows = rhs.size();
	cols = rhs[0].size();
	m = new double*[rows];
	for (int i = 0; i<rows; i++) {
		m[i] = new double[cols];
	}
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			m[i][j] = rhs[i][j];
		}
	}
}

Matrix::Matrix(const Matrix & rhs) { // copy constructor
	rows = rhs.getRows();
	cols = rhs.getCols();
	m = new double*[rows];
	for (int i = 0; i<rows; i++) {
		m[i] = new double[cols];
	}
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			m[i][j] = rhs[i][j];
		}
	}
}

// make vector (n,1) or (1,n) matrix
Matrix::Matrix(double * vec, int n, string type, bool remove_vec) { 
	if (type == "column") {
		m = new double*[n];
		for (int i = 0; i < n; i++) {
			m[i] = new double[1];
		}
		for (int i = 0; i < n; i++) {
			m[i][0] = vec[i];
		}
		rows = n;
		cols = 1;
	}
	else {
		m = new double*[1];
		m[0] = new double[n];
		for (int i = 0; i < n; i++) {
			m[0][i] = vec[i];
		}
		rows = 1;
		cols = n;
	}
	if (remove_vec) {
		delete[] vec;
		vec = NULL;
	}
}


Matrix::Matrix(string fname) { // reads matrix from input file
	ifstream input(fname.c_str());
	input >> rows >> cols;
	m = new double*[rows];
	for (int i = 0; i < rows; i++) {
		m[i] = new double[cols];
	}
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			input >> m[i][j];
		}
	}
}


Matrix::~Matrix() {
	clearMatrix();
}

double* Matrix::return_col(int c)const { // returns cth col
	assert(c < cols);
	double * res = new double[rows];
	for (int i = 0; i < rows; i++) {
		res[i] = m[i][c];
	}
	return res;
}

double* Matrix::return_row(int c)const { // returns cth row
	assert(c < rows);
	double * res = new double[cols];
	for (int i = 0; i < cols; i++) {
		res[i] = m[c][i];
	}
	return res;
}


Matrix Matrix::operator+(const Matrix & rhs) const { // sum two same sized matrices
	assert(rows == rhs.getRows() && cols == rhs.getCols());
	Matrix res(rows, cols);
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			res[i][j] = m[i][j] + rhs[i][j];
		}
	}
	return res;
}

Matrix Matrix::operator-(const Matrix & rhs) const { // subtract two same sized matrices
	assert(rows == rhs.getRows() && cols == rhs.getCols());
	Matrix res(rows, cols);
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			res[i][j] = m[i][j] - rhs[i][j] ;
		}
	}
	return res;
}

Matrix Matrix::operator*(const Matrix & rhs)const { // matrix multiplication mxn * nxk
	assert(cols == rhs.getRows());
	Matrix res(rows, rhs.getCols());
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < rhs.getCols(); j++) {
			for (int k = 0; k < cols; k++) {
				res[i][j] += m[i][k] * rhs[k][j];
			}
		}
	}
	return res;
}

void Matrix::printMatrix()const { // print matrix with 2 floating point precision
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			cout << setprecision(2) << m[i][j] << "\t";
		}
		cout << endl;
	}
}


const Matrix & Matrix::operator=(const Matrix & rhs) { // assignment operator
	if (!(rows == rhs.getRows() && cols == rhs.getCols())) {
		clearMatrix();
		rows = rhs.getRows();
		cols = rhs.getCols();
		m = new double*[rows];
		for (int i = 0; i<rows; i++) {
			m[i] = new double[cols];
		}
	}
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			m[i][j] = rhs[i][j];
		}
	}
	return *this;
}

Matrix Matrix::operator*(double rhs)const { // scale vector with a constant
	Matrix temp(*this);
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			temp[i][j] = m[i][j] * rhs;
		}
	}
	return temp;
}


Matrix Matrix::operator+(double rhs)const { // subtract a constant from each element of matrix
	Matrix temp(*this);
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			temp[i][j] = m[i][j] + rhs;
		}
	}
	return temp;
}

Matrix Matrix::operator-(double rhs)const { // subtract a constant from each element of matrix
	Matrix temp(*this);
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			temp[i][j] = m[i][j] - rhs;
		}
	}
	return temp;
}

Matrix Matrix::operator/(double rhs)const { // divide each element of matrix by a constant
	Matrix temp(*this);
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			temp[i][j] = m[i][j] / rhs;
		}
	}
	return temp;
}

bool Matrix::operator==(const Matrix & rhs)const { // checks if two matrices are same
	assert(rows == rhs.getRows() && cols == rhs.getCols());
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			if (m[i][j] != rhs[i][j]) {
				return false;
			}
		}
	}
	return true;
}

bool Matrix::operator!=(const Matrix & rhs)const { // checks if two matrices are different
	return !(*this == rhs);
}

// syntactic sugars
void Matrix::operator/=(double rhs){
	*this = *this / rhs;
}

void Matrix::operator+=(double rhs) {
	*this = *this + rhs;
}

void Matrix::operator-=(double rhs) {
	*this = *this - rhs;
}

void Matrix::operator*=(double rhs) {
	*this = *this * rhs;
}

void Matrix::operator*=(const Matrix & rhs) {
	*this = *this * rhs;
}

void Matrix::operator+=(const Matrix & rhs) {
	*this = *this + rhs;
}

void Matrix::operator-=(const Matrix & rhs) {
	*this = *this - rhs;
}


Matrix Matrix::T(bool self) { // returns transpose of matrix, if self is true it updates the object
	Matrix temp(cols, rows);
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			temp[j][i] = m[i][j];
		}
	}
	if (self) {
		*this = temp;
	}
	return temp;
}

void Matrix::clearMatrix() { // deallocates the dynamic memory
	if (!m) {
		return;
	}
	for (int i = 0; i < rows; i++) {
		delete[] m[i];
	}
	delete[] m;
	m = NULL;
	rows = 0;
	cols = 0;
}


int argmax(double* col, double & max, int n) { // returns max_val and argmax of double dynamic array
	int index=-1;
	for (int i = 0; i < n; i++) {
		if (max < abs(col[i])) {
			max = abs(col[i]);
			index = i;
		}
	}
	return index;
}


void Matrix::swap_rows(int index_1, int index_2) { // swaps index_1th row with index_2th
	if (index_1 == index_2) {
		return;
	}
	double *temp = return_row(index_1); // temp = index_1th row
	for (int i = 0; i < cols; i++) { // index_1th row = index_2th row
		m[index_1][i] = m[index_2][i];
	}
	for (int i = 0; i < cols; i++) { // index_2th row = temp
		m[index_2][i] = temp[i];
	}
	delete[] temp;
}


void Matrix::swap_cols(int index_1, int index_2) { // swaps index_1th row with index_2th
	if (index_1 == index_2) {
		return;
	}
	double *temp = return_col(index_1); // temp = index_1th col
	for (int i = 0; i < rows; i++) { // index_1th col = index_2th col
		m[i][index_1] = m[i][index_2];
	}
	for (int i = 0; i < rows; i++) { // index_2th col = temp
		m[i][index_2] = temp[i];
	}
	delete[] temp;
}


Matrix Matrix::apply(double(*function)(double)) {
	Matrix res(rows, cols);
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			res[i][j] = (*function)(m[i][j]);
		}
	}
	return res;
}

Matrix Matrix::gaussianElimination(bool self) { // performs gaussian elimination, if self, updates the object
	Matrix temp(*this);
	int h = 0, k = 0; // pivot row and column
	double * col;
	while (h < rows && k < cols) {
		col = temp.return_col(k); // kth column
		double max=0;
		int index = argmax(col, max, rows);
		if (max == 0) { // column is all zero
			k++;
		}
		else {
			temp.swap_rows(h, index);
			if (index < h)
				h = index;
			// for all rows under pivot
			for (int i = h + 1; i < rows; i++) {
				double coeff = temp[i][k] / temp[h][k];
				temp[i][k] = 0;
				for (int j = k + 1; j < cols; j++) {
					temp[i][j] -= temp[h][j] * coeff;
				}
			}
			h++;
			k++;
		}
		delete[] col;
		col = NULL;
	}
	if (self) {
		*this = temp;
	}
	return temp;
}


Matrix Matrix::remove_row_col(int r, int c, bool self) { // remove rth row and cth col
	assert(r < rows && c < cols);
	Matrix temp(rows-1, cols-1);
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			if (i != r && j != c) {
				int cur_i = i, cur_j = j;
				if (i > r) {
					cur_i--;
				}
				if (j > c) {
					cur_j--;
				}
				temp[cur_i][cur_j] = m[i][j];
			}
		}
	}
	if (self) {
		*this = temp;
	}
	return temp;
}


double Matrix::determinant() { // computes the determinant of the matrix
	assert(rows == cols);
	if (rows == 1) { // trivial case
		return m[0][0];
	}
	else if (rows == 2) { // leibniz rule
		return m[0][0] * m[1][1] - m[0][1] * m[1][0];
	}
	double det = 0;
	int sign = 1;
	for (int i = 0; i < cols; i++) { // iterate over first row
		Matrix reduced_mat = remove_row_col(0, i);
		det += sign* m[0][i] * reduced_mat.determinant(); // recursively find reduced matrix's determinant
		sign *= -1; // alternate the sign
	}
	return det;
}

Matrix Matrix::cofactor_matrix() { // computes the cofactor matrix of the object
	Matrix res(rows, cols);
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			Matrix temp = remove_row_col(i, j);
			res[i][j] = temp.determinant() * pow(-1, i+j);
		}
	}
	return res;
}


void Matrix::trimMatrix() { // removes false elements due to precision
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			double s = abs(m[i][j]);
			if (s < 1e-11) {
				m[i][j] = 0;
			}
		}
	}
}

Matrix Matrix::inverse() { // returns the inverse of the object
	double det = determinant();
	if (det == 0) {
		cout << "Matrix is singular" << endl;
		return Matrix();
	}
	Matrix inv = cofactor_matrix().T()/det; // 1/det(A) * cofactor(A)^T
	return inv;
}


// using gramd schmidt equations applies QR decomposition
pair<Matrix, Matrix> Matrix::qr_decomposition() {
	assert(determinant() != 0); // it fails when matrix is singular
	Matrix Q(rows, cols);
	Matrix R(rows, cols);

	vector<Matrix> us(cols);
	vector<Matrix> es(cols);

	for (int i = 0; i < cols; i++) {
		us[i] = Matrix(return_col(i), rows, string("row")); // 1xn
		for (int j = i; j > 0; j--) { // u_i = a_i - (a_i*e_i-1)e_i-1 - (a_i*e_i-2)e_i-2 - ...
			Matrix temp = Matrix(return_col(i), rows, string("row")) * es[j - 1].T();
			us[i] -= es[j-1] * temp[0][0];
		}
		// normalize u_i
		es[i] = us[i];
		es[i] /= sqrt((us[i] * us[i].T())[0][0]);
	}

	// construct Q
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			Q[i][j] = es[j][0][i];
		}
	}

	// construct R
	for (int i = 0; i < rows; i++) {
		for (int j = i; j < cols; j++) {
			Matrix temp = Matrix(return_col(j), rows, string("row")) * es[i].T();
			R[i][j] = temp[0][0];
		}
	}
	return make_pair(Q, R);
}

Matrix Matrix::elewiseSquare(bool self) { // computes elementwise square of the matrix
	Matrix res(*this);
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			res[i][j] = m[i][j] * m[i][j];
		}
	}
	if (self) {
		*this = res;
	}
	return res;
}

Matrix Matrix::sum(int axis, bool self) { // computes the sum of elements w.r.t. given axis, if self updates the object
	Matrix temp;
	if (axis == 0) {
		temp = Matrix(rows,1);
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				temp[i][0] += m[i][j];
			}
		}
	}
	else if (axis == 1) {
		temp = Matrix(1, cols);
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				temp[0][j] += m[i][j];
			}
		}
	}
	else {
		temp = Matrix(1, 1);
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				temp[0][0] += m[i][j];
			}
		}
	}
	if (self) {
		*this = temp;
	}
	return temp;
}

double Matrix::frobeniusNorm() { // computes the Frobenius norm of the matrix
	return sqrt(elewiseSquare().sum(-1)[0][0]);
}


Matrix Matrix::clipRows(int start, int end, bool self) { // returns the rows between start, end M[start:end]
	assert(start < rows && end < rows);
	Matrix temp(end-start+1, cols);
	for (int i = start; i <= end; i++) { // end included
		for (int j = 0; j < cols; j++) {
			temp[i-start][j] = m[i][j];
		}
	}
	if (self) {
		*this = temp;
	}
	return temp;
}


Matrix Matrix::clipCols(int start, int end, bool self) { // returns the columns between start, end M[start:end]
	assert(start < cols && end < cols);
	Matrix temp(rows, end - start + 1);
	for (int i = 0; i < rows; i++) {
		for (int j = start; j <= end; j++) { // end included
			temp[i][j - start] = m[i][j];
		}
	}
	if (self) {
		*this = temp;
	}
	return temp;
}


void jacobi(double alpha, double beta, double gamma, Matrix & G, double & t) { // returns jacobi transformation matrix
	double c = 1, s=0;
	t = 0;
	G = Matrix(2, 2);
	if (beta != 0) {
		double tau = (gamma - alpha) / (2 * beta);
		if (tau >= 0) {
			t = 1 / (tau + sqrt(1 + tau * tau));
		}
		else {
			t = -1 / (-tau + sqrt(1 + tau * tau));
		}
		c = 1 / sqrt(1 + t * t);
		s = t*c;
	}
	G[0][0] = c;
	G[0][1] = s;
	G[1][0] = -s;
	G[1][1] = c;
	s = t;
}

void Matrix::assignCol(Matrix mat, int col) { // assigns mat to colth column of the object
	if (mat.getCols() != 1) {
		cout << "Warning, mat is not a column vector" << endl;
	}
	for (int i = 0; i < rows; i++) {
		m[i][col] = mat[i][0];
	}
}

void Matrix::assignRow(Matrix mat, int row) { // assigns mat to rowth row of the object
	if (mat.getRows() != 1) {
		cout << "Warning, mat is not a row vector" << endl;
	}
	for (int i = 0; i < cols; i++) {
		m[row][i] = mat[0][i];
	}
}


bool comp(pair<int, double> & p1, pair<int, double> & p2) { // utility function to compare pairs
	return p1.second > p2.second;
}


array<Matrix, 3> Matrix::singular_value_decomposition() { // returns U,E,VT where U*E*VT = Object
	Matrix copy_A = *this;
	Matrix U(rows, rows), V = eye(cols), sigma = elewiseSquare().sum(1);
	double eps = 1e-15, rots = 1;
	double tolsigma = eps*frobeniusNorm();
	while (rots >= 1) { // iterate until all (p,q) pairs give less error than tolerance
		rots = 0;
		for (int p = 0; p < cols - 1; p++) {
			for (int q = p + 1; q < cols; q++) {
				double beta = (Matrix(copy_A.return_col(p), rows, "row") * Matrix(copy_A.return_col(q), rows, "column"))[0][0];
				if (sigma[0][p] * sigma[0][q] > tolsigma && abs(beta) >= eps*sqrt(sigma[0][p] * sigma[0][q])) {
					rots++;
					Matrix G;
					double t;
					jacobi(sigma[0][p], beta, sigma[0][q], G, t); // jacobi rotation matrix
					// update eigenvalues
					sigma[0][p] = sigma[0][p] - beta*t;
					sigma[0][q] = sigma[0][q] + beta*t;
					Matrix temp = concatCols(Matrix(copy_A.return_col(p), rows), Matrix(copy_A.return_col(q), rows))*G;
					// update pth and qth cols of A and V
					copy_A.assignCol(Matrix(temp.return_col(0), copy_A.getRows()), p);
					copy_A.assignCol(Matrix(temp.return_col(1), copy_A.getRows()), q);
					Matrix temp2 = concatCols(Matrix(V.return_col(p), rows), Matrix(V.return_col(q), rows))*G;
					V.assignCol(Matrix(temp2.return_col(0), V.getRows()), p);
					V.assignCol(Matrix(temp2.return_col(1), V.getRows()), q);
				}
			}
		}
	}
	// post processing w.r.t. indices (eigenvalues should appear in descending order)
	Matrix copy_V = V;
	vector<pair<int,double>> ind(sigma.getCols());

	for (int i = 0; i < sigma.getCols(); i++) {
		ind[i] = make_pair(i,sigma[0][i]);
	}
	sort(ind.begin(), ind.end(), comp);

	// Construct U and V
	for (int i = 0; i < U.getRows(); i++) {
		for (int j = 0; j < ind.size(); j++) {
			sigma[0][j] = ind[j].second;
			U[i][j] = copy_A[i][ind[j].first];
		}
	}
	for (int i = 0; i < copy_V.getRows(); i++) {
		for (int j = 0; j < ind.size(); j++) {
			V[i][j] = copy_V[i][ind[j].first];
		}
	}
	// find eigenvalues
	for (int k = 0; k < cols; k++) {
		if (sigma[0][k] == 0) {
			for (int j = k + 1; j < cols; j++) {
				sigma[0][j] = 0;
			}
		}
		sigma[0][k] = sqrt(abs(sigma[0][k]));
		for (int i = 0; i < rows; i++) {
			U[i][k] /= sigma[0][k];
		}
	}

	array<Matrix, 3> arr;
	arr[0] = U;
	arr[1] = diag(sigma);
	arr[2] = V.T(); // V_T
	return arr;
}

vector<double> Matrix::eigs() {
	Matrix copy_A = *this;
	Matrix sigma = elewiseSquare().sum(1);
	double eps = 1e-15, rots = 1;
	double tolsigma = eps*frobeniusNorm();
	while (rots >= 1) { // iterate until all (p,q) pairs give less error than tolerance
		rots = 0;
		for (int p = 0; p < cols - 1; p++) {
			for (int q = p + 1; q < cols; q++) {
				double beta = (Matrix(copy_A.return_col(p), rows, "row") * Matrix(copy_A.return_col(q), rows, "column"))[0][0];
				if (sigma[0][p] * sigma[0][q] > tolsigma && abs(beta) >= eps*sqrt(sigma[0][p] * sigma[0][q])) {
					rots++;
					Matrix G;
					double t;
					jacobi(sigma[0][p], beta, sigma[0][q], G, t); // jacobi rotation matrix
					// update eigenvalues
					sigma[0][p] = sigma[0][p] - beta*t;
					sigma[0][q] = sigma[0][q] + beta*t;
					Matrix temp = concatCols(Matrix(copy_A.return_col(p), rows), Matrix(copy_A.return_col(q), rows))*G;
					// update pth and qth cols of A and V
					copy_A.assignCol(Matrix(temp.return_col(0), copy_A.getRows()), p);
					copy_A.assignCol(Matrix(temp.return_col(1), copy_A.getRows()), q);
				}
			}
		}
	}
	vector<double> res(sigma.getCols());
	for (int i = 0; i < res.size(); i++) {
		res[i] = sigma[0][i];
	}
	return res;
}



Matrix diag(Matrix & s) { // returns a diagonal matrix whose components are s
	if (s.getCols() != 1 && s.getRows() != 1) {
		cout << "Warning, matrix is not one dimensional" << endl;
	}
	int n = max(s.getCols(), s.getRows());
	Matrix temp(n,n);
	int k=0;
	for (int i = 0; i < s.getRows(); i++) {
		for (int j = 0; j < s.getCols(); j++) {
			temp[k][k] = s[i][j];
			k++;
		}
	}
	return temp;
}

Matrix diag(vector<double> & s) { // diagonalize 1d vector
	int n = s.size();
	Matrix temp(n, n);
	for (int i = 0; i < n; i++) {
			temp[i][i] = s[i];
	}
	return temp;
}


Matrix eye(int n) { // returns an identity matrix with nxn
	Matrix res(n, n);
	for (int i = 0; i < n; i++) {
		res[i][i] = 1;
	}
	return res;
}


Matrix concatCols(Matrix a, Matrix b) { // concatenates columns
	assert(a.getRows() == b.getRows());
	Matrix temp(a.getRows(), a.getCols() + b.getCols());
	for (int i = 0; i < a.getRows(); i++) {
		for (int j = 0; j < a.getCols(); j++) {
			temp[i][j] = a[i][j];
		}
		for (int j = a.getCols(); j < a.getCols() + b.getCols(); j++) {
			temp[i][j] = b[i][j - a.getCols()];
		}
	}
	return temp;
}

Matrix concatRows(Matrix a, Matrix b) { // concatenates rows (appends)
	assert(a.getCols() == b.getCols());
	Matrix temp(a.getRows() + b.getRows(), a.getCols());
	for (int i = 0; i < a.getRows(); i++) {
		for (int j = 0; j < a.getCols(); j++) {
			temp[i][j] = a[i][j];
		}
	}
	for (int i = a.getRows(); i < a.getRows()+ b.getRows(); i++) {
		for (int j = 0; j < b.getCols(); j++) {
			temp[i][j] = b[i- a.getRows()][j];
		}
	}
	return temp;
}

Matrix zeros(int n) { // returns a nxn matrix with 0s
	return Matrix(n, n);
}

Matrix ones(int n) { // returns a nxn matrix with 1s
	return Matrix(n, n, 1);
}

double max(Matrix m) {
	double d = INT8_MIN;
	for (int i = 0; i < m.getRows(); i++) {
		for (int j = 0; j < m.getCols(); j++) {
			if (d < m[i][j]) {
				d = m[i][j];
			}
		}
	}
	return d;
}


vector<vector<double>> matrixtoVector(Matrix & m) {
	vector<vector<double>> res(m.getRows(), vector<double>(m.getCols()));
	for (int i = 0; i < m.getRows(); i++) {
		for (int j = 0; j < m.getCols(); j++) {
			res[i][j] = m[i][j];
		}
	}
	return res;
}
