#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include "Matrix.h"
#include <vector>
#include <functional>

class NeuralNetwork
{
public:
    NeuralNetwork(int inputNodes, int hiddenNodes, int outputNodes);

    std::vector<double> feedForward(const std::vector<double> &inputArray);
    Matrix feedForward(const Matrix &inputMatrix); // Batch processing
    void train(const std::vector<double> &inputArray, const std::vector<double> &targetArray);

    void setLearningRate(double rate) { learningRate = rate; }

private:
    int inputNodes;
    int hiddenNodes;
    int outputNodes;
    double learningRate;

    Matrix weightsIH; // Input -> Hidden
    Matrix weightsHO; // Hidden -> Output
    Matrix biasH;     // Hidden bias
    Matrix biasO;     // Output bias

    // Activation function
    static double sigmoid(double x);
    static double dSigmoid(double y); // Derivative of sigmoid (expects y = sigmoid(x))
};

#endif
