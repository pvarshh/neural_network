#include "NeuralNetwork.h"
#include <cmath>
#include <iostream>

NeuralNetwork::NeuralNetwork(int inputNodes, int hiddenNodes, int outputNodes)
    : inputNodes(inputNodes), hiddenNodes(hiddenNodes), outputNodes(outputNodes),
      weightsIH(hiddenNodes, inputNodes),
      weightsHO(outputNodes, hiddenNodes),
      biasH(hiddenNodes, 1),
      biasO(outputNodes, 1),
      learningRate(0.1)
{
    weightsIH.randomize();
    weightsHO.randomize();
    biasH.randomize();
    biasO.randomize();
}

double NeuralNetwork::sigmoid(double x)
{
    return 1.0 / (1.0 + std::exp(-x));
}

double NeuralNetwork::dSigmoid(double y)
{
    // derivative of sigmoid is sigmoid(x) * (1 - sigmoid(x))
    // here y is already sigmoid(x)
    return y * (1.0 - y);
}

std::vector<double> NeuralNetwork::feedForward(const std::vector<double> &inputArray)
{
    if (inputArray.size() != inputNodes)
    {
        throw std::invalid_argument("Input size does not match neural network input nodes");
    }

    // Convert input to matrix
    Matrix inputs = Matrix::fromVector(inputArray);

    // Hidden layer
    Matrix hidden = weightsIH.multiply(inputs);
    hidden = hidden.add(biasH);
    hidden = hidden.map(sigmoid);

    // Output layer
    Matrix outputs = weightsHO.multiply(hidden);
    outputs = outputs.add(biasO);
    outputs = outputs.map(sigmoid);

    // Convert back to vector
    std::vector<double> result;
    for (int i = 0; i < outputs.getRows(); i++)
    {
        result.push_back(outputs.at(i, 0));
    }
    return result;
}

Matrix NeuralNetwork::feedForward(const Matrix &inputMatrix)
{
    // Batch processing: inputMatrix is (BatchSize x InputNodes)
    if (inputMatrix.getCols() != inputNodes)
    {
        throw std::invalid_argument("Input matrix columns must match input nodes");
    }

    // Hidden layer
    // weightsIH is (Hidden x Input), we need (Input x Hidden) for multiplication
    Matrix weightsIH_T = weightsIH.transpose();
    Matrix hidden = inputMatrix.multiply(weightsIH_T);
    hidden = hidden.addBias(biasH);
    hidden = hidden.map(sigmoid);

    // Output layer
    // weightsHO is (Output x Hidden), we need (Hidden x Output)
    Matrix weightsHO_T = weightsHO.transpose();
    Matrix outputs = hidden.multiply(weightsHO_T);
    outputs = outputs.addBias(biasO);
    outputs = outputs.map(sigmoid);

    return outputs;
}

void NeuralNetwork::train(const std::vector<double> &inputArray, const std::vector<double> &targetArray)
{
    if (inputArray.size() != inputNodes || targetArray.size() != outputNodes)
    {
        throw std::invalid_argument("Input or Target size mismatch");
    }

    // --- Feed Forward ---
    Matrix inputs = Matrix::fromVector(inputArray);

    Matrix hidden = weightsIH.multiply(inputs);
    hidden = hidden.add(biasH);
    hidden = hidden.map(sigmoid);

    Matrix outputs = weightsHO.multiply(hidden);
    outputs = outputs.add(biasO);
    outputs = outputs.map(sigmoid);

    // --- Backpropagation ---

    // 1. Calculate Output Errors
    // Error = Target - Output
    Matrix targets = Matrix::fromVector(targetArray);
    Matrix outputErrors = targets.subtract(outputs);

    // 2. Calculate Output Gradients
    // Gradient = lr * error * dSigmoid(output)
    Matrix gradients = outputs.map(dSigmoid);
    gradients = gradients.hadamard(outputErrors);
    gradients = gradients.multiplyScalar(learningRate);

    // 3. Calculate Hidden->Output Deltas
    // Delta = Gradient * Hidden_Transpose
    Matrix hiddenT = hidden.transpose();
    Matrix weightHO_Deltas = gradients.multiply(hiddenT);

    // 4. Adjust Weights and Biases (Hidden -> Output)
    weightsHO = weightsHO.add(weightHO_Deltas);
    biasO = biasO.add(gradients);

    // 5. Calculate Hidden Errors
    // Hidden_Error = Weights_HO_Transpose * Output_Errors
    Matrix whoT = weightsHO.transpose();
    Matrix hiddenErrors = whoT.multiply(outputErrors);

    // 6. Calculate Hidden Gradients
    Matrix hiddenGradient = hidden.map(dSigmoid);
    hiddenGradient = hiddenGradient.hadamard(hiddenErrors);
    hiddenGradient = hiddenGradient.multiplyScalar(learningRate);

    // 7. Calculate Input->Hidden Deltas
    Matrix inputsT = inputs.transpose();
    Matrix weightIH_Deltas = hiddenGradient.multiply(inputsT);

    // 8. Adjust Weights and Biases (Input -> Hidden)
    weightsIH = weightsIH.add(weightIH_Deltas);
    biasH = biasH.add(hiddenGradient);
}
