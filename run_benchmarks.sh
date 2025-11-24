#!/bin/bash

echo "--- Compiling C++ Benchmark ---"
g++ src/benchmark.cpp src/NeuralNetwork.cpp src/Matrix.cpp -o benchmark_cpp -std=c++11 -O3 -ffast-math

echo "--- Running Benchmark Comparison ---"
python3 benchmark.py
