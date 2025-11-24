#include "NeuralNetwork.h"
#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>

int main() {
    // Seed random number generator
    std::srand(std::time(nullptr));

    // Create Neural Network: 2 inputs, 4 hidden nodes, 1 output
    NeuralNetwork nn(2, 4, 1);
    nn.setLearningRate(0.1);

    // Training Data (XOR)
    struct TrainingData {
        std::vector<double> inputs;
        std::vector<double> targets;
    };

    std::vector<TrainingData> trainingSet = {
        {{0, 0}, {0}},
        {{0, 1}, {1}},
        {{1, 0}, {1}},
        {{1, 1}, {0}}
    };

    std::cout << "Training..." << std::endl;

    // Train for 50,000 epochs
    for (int i = 0; i < 50000; i++) {
        // Pick a random training example
        int index = std::rand() % trainingSet.size();
        nn.train(trainingSet[index].inputs, trainingSet[index].targets);
    }

    std::cout << "Training Complete." << std::endl;
    std::cout << "Testing XOR:" << std::endl;

    for (const auto& data : trainingSet) {
        std::vector<double> output = nn.feedForward(data.inputs);
        std::cout << "Input: " << data.inputs[0] << ", " << data.inputs[1] 
                  << " -> Output: " << output[0] << std::endl;
    }

    return 0;
}
