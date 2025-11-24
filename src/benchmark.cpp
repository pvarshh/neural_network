#include "NeuralNetwork.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <random>

int main(int argc, char* argv[]) {
    // Parameters
    int input_size = 784;
    int hidden_size = 128;
    int output_size = 10;
    int num_samples = 10000;

    if (argc > 1) {
        num_samples = std::atoi(argv[1]);
    }

    std::cout << "Benchmarking C++ NeuralNetwork" << std::endl;
    std::cout << "Architecture: " << input_size << " -> " << hidden_size << " -> " << output_size << std::endl;
    std::cout << "Samples: " << num_samples << std::endl;

    // Initialize Neural Network
    NeuralNetwork nn(input_size, hidden_size, output_size);

    // Generate random data
    std::cout << "Generating random data..." << std::endl;
    Matrix inputBatch(num_samples, input_size);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    for (int i = 0; i < num_samples; ++i) {
        for (int j = 0; j < input_size; ++j) {
            inputBatch.at(i, j) = dis(gen);
        }
    }

    // Benchmark Inference
    std::cout << "Starting inference benchmark (Batch Mode)..." << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();

    Matrix outputBatch = nn.feedForward(inputBatch);

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end_time - start_time;

    std::cout << "Total time: " << duration.count() << " seconds" << std::endl;
    std::cout << "Average time per sample: " << (duration.count() / num_samples) * 1000.0 << " ms" << std::endl;
    std::cout << "Samples per second: " << num_samples / duration.count() << std::endl;

    return 0;
}
