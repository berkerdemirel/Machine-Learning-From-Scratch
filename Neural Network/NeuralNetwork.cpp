#include "NeuralNetwork.h"
#include <algorithm>
#include <omp.h>

double leakyReLu(double d) {
	return max(0.1*d, d);
}

double leakyReLuPrime(double d) {
	return d > 0 ? 1 : 0.1;
}

double sigmoid(double d) {
	return 1/(1+exp(-d));
}

double sigmoidprime(double d) {
	/*double a = (1 / (1 + exp(-d)));
	return a*(1 - a);
	*/
	double nom = exp(-d);
	double denom((1 + exp(-d))*(1 + exp(-d)));
	if (nom > INT_MAX) {
		return 0;
	}
	double res = nom / denom;
	return res>1000?1000:res;
}

double random(double x) {
	return (double)(rand() % 1000 - 1) / 1000 - 0.5;
}


double stepFunction(double x){
	if (x>0.9) {
		return 1.0;
	}
	if (x<0.1) {
		return 0.0;
	}
	return x;
}


NeuralNetwork::NeuralNetwork(int i, int h, int o, double l, int num_epochs, int b_s) 
	:inputlayers(i+1), hiddenlayers(h+1), outputlayers(o), lr(l), n_epochs(num_epochs), batch_size(b_s){
	
	W1 = Matrix(i+1, h+1);
	W2 = Matrix(h+1, o); 

	W1 = W1.apply(random);
	W2 = W2.apply(random);
}

Matrix softmax(Matrix raw_preds) {
	double max = -1000000;
	for (int i = 0; i < raw_preds.getRows(); i++) {
		for (int j = 0; j < raw_preds.getCols(); j++) {
			if (max < raw_preds[i][j]) {
				max = raw_preds[i][j];
			}
		}
	}
	if (max == 0) {
		max = 1;
	}
	raw_preds /= max;// raw_preds.multiply(1 / max);
	Matrix out = raw_preds.apply(exp);
	double sum = out.sum(-1)[0][0];
	out /= sum; // = out.multiply(1 / sum);
	return out;
}

int argmax(Matrix & Y) {
	double max_val = -1e6;
	int index = -1;
	for (int i = 0; i < Y.getCols(); i++) {
		if (Y[0][i] > max_val) {
			max_val = Y[0][i];
			index = i;
		}
	}
	return index;
}

Matrix multiplyy(Matrix m1, Matrix m2) {
	assert(m1.getRows() == m2.getRows() && m1.getCols() == m2.getCols());
	Matrix res(m1.getRows(), m1.getCols());
	for (int i = 0; i < m1.getRows(); i++) {
		for (int j = 0; j < m1.getCols(); j++) {
			res[i][j] = m1[i][j] * m2[i][j];
		}
	}
	return res;
}


// H = f(X*W1+B1) hidden output
// Y = f(H*W2+B2) output
// J = 1/2(Y*-Y)^2 loss function

// dJ/dB2 = dJ/dY * dY/dB2
// dJ/dB2 = (Y-Y*) * f'(X*W1+B1)

// dJ/dW2 = dJ/dY * dY/dW2
// dJ/dW2 = (Y-Y*) * f'(H*W2+B2)*H

// dJ/dB1 = dJ/dY * dY/dH * dH/dB1
// dJ/dB1 = dJ/dB2 * W2 * f'(X*W1+B1)

// dJ/dW1 = dJ/dY * dY/dH * dH/dW1
// dJ/dW1 = dJ/dB2 * X * W2 * f'(X*W1+B1) = X * dJ/dB1
void NeuralNetwork::train(Matrix & train, Matrix & labels) {
	cout << batch_size << endl;
	Matrix Xs = concatCols(train, Matrix(train.getRows(), 1, 1)); // for adding bias
	for (int e = 0; e < n_epochs; e++) {
		int true_count = 0;
		double start = omp_get_wtime(), end;
		Matrix dJ_dW2 = Matrix(W2.getRows(), W2.getCols()), dJ_dW1 = Matrix(W1.getRows(), W1.getCols());
		for (int i = 0; i < Xs.getRows(); i++) {
			int label = labels[i][0];
			// feed forward
			Matrix X = Matrix(Xs.return_row(i), Xs.getCols(), "row");
			Matrix H = (X * W1).apply(sigmoid);
			Matrix out = (H*W2).apply(sigmoid);
			Matrix g_t = Matrix(1, outputlayers); // one hot encoding (ground truth)
			g_t[0][label] = 1; 
			// cout << argmax(out) << " " << label << endl;
			if (argmax(out) == label) {
				true_count++;
			}
			// backprop
			Matrix dH = multiplyy((g_t - out) * 2, out.apply(sigmoidprime));
			dJ_dW2 += H.T() * dH;
			dJ_dW1 += X.T() * (multiplyy(dH*W2.T(), H.apply(sigmoidprime)));
			// Matrix dJ_dW2 = H.T() * dH;
			// Matrix dJ_dW1 = X.T() * (multiply(dH*W2.T(), H.apply(sigmoidprime)));
			// update weights
			if ((i + 1) % batch_size == 0) {
				dJ_dW2 /= batch_size;
				dJ_dW1 /= batch_size;

				W1 += dJ_dW1*lr;
				W2 += dJ_dW2*lr;

				dJ_dW2 = Matrix(dJ_dW2.getRows(), dJ_dW2.getCols(), 0);
				dJ_dW1 = Matrix(dJ_dW1.getRows(), dJ_dW1.getCols(), 0);
			}
		}
		if (Xs.getRows() % batch_size != 0) { // if the last batch is left half finished
			int residual = Xs.getRows() % batch_size;
			dJ_dW2 /= residual;
			dJ_dW1 /= residual;

			W1 += dJ_dW1*lr;
			W2 += dJ_dW2*lr;

			dJ_dW2 = Matrix(dJ_dW2.getRows(), dJ_dW2.getCols(), 0);
			dJ_dW1 = Matrix(dJ_dW1.getRows(), dJ_dW1.getCols(), 0);
		}
		cout << "Epoch accuracy: " << (double)true_count / Xs.getRows() << " in " << omp_get_wtime()-start << " seconds" << endl;
	}
}


Matrix NeuralNetwork::predict(Matrix & X) {
	Matrix res(X.getRows(), 1);
	Matrix X_b = concatCols(X, Matrix(X.getRows(), 1, 1));
	for (int i = 0; i < X_b.getRows(); i++) {
		Matrix X = Matrix(X_b.return_row(i), X_b.getCols(), "row");
		Matrix H = (X * W1).apply(sigmoid);
		Matrix out = (H*W2).apply(sigmoid);
		int label = argmax(out);
		res[i][0] = label;
	}
	return res;
}


void NeuralNetwork::readFromTxt(string fileName) {
	ifstream input(fileName.c_str());
	for (int i = 0; i < W1.getRows(); i++) {
		for (int j = 0; j < W1.getCols(); j++) {
			input >> W1[i][j];
		}
	}
	for (int i = 0; i < W2.getRows(); i++) {
		for (int j = 0; j < W2.getCols(); j++) {
			input >> W2[i][j];
		}
	}
}


void NeuralNetwork::writeToTxt(string fileName) {
	ofstream out;
	out.open(fileName, std::ofstream::out | std::ofstream::trunc);
	for (int i = 0; i < W1.getRows(); i++) {
		for (int j = 0; j < W1.getCols(); j++) {
			out << W1[i][j] << " ";
		}
		out << endl;
	}
	out << endl;
	for (int i = 0; i < W2.getRows(); i++) {
		for (int j = 0; j < W2.getCols(); j++) {
			out << W2[i][j] << " ";
		}
		out << endl;
	}
	out.close();
}
