#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H
#include <cmath>
#include <fstream>
#include "matrix.h"

class NeuralNetwork {
public:
	NeuralNetwork(int i, int h, int o, double l = 0.005, int num_epochs=10, int b_s=128);
	void train(Matrix & train, Matrix & label);
	Matrix predict(Matrix & X);
	void setLearningRate(double l) { lr = l; }
	void readFromTxt(string fileName);
	void writeToTxt(string fileName);
private:
	int inputlayers;
	int hiddenlayers;
	int outputlayers;
	int n_epochs;
	int batch_size;
	double lr;
	Matrix W1;
	Matrix W2;
};
#endif