#ifndef MATRIX_H
#define MATRIX_H

#include <vector>
#include <iostream>
#include <functional>

class Matrix {
public:
    Matrix(int rows, int cols);
    
    // Accessors
    int getRows() const { return rows; }
    int getCols() const { return cols; }
    double& at(int r, int c);
    const double& at(int r, int c) const;

    // Operations
    Matrix multiply(const Matrix& other) const;
    Matrix add(const Matrix& other) const;
    Matrix subtract(const Matrix& other) const;
    Matrix multiplyScalar(double scalar) const;
    Matrix hadamard(const Matrix& other) const; // Element-wise multiplication
    Matrix addBias(const Matrix& bias) const; // Broadcast addition
    Matrix transpose() const;
    
    // Element-wise function application (for activation functions)
    Matrix map(std::function<double(double)> func) const;

    // Static helpers
    static Matrix multiply(const Matrix& a, const Matrix& b);
    static Matrix fromVector(const std::vector<double>& vec);
    
    // Utilities
    void randomize();
    void print() const;

    // Raw data access for performance
    const std::vector<double>& getData() const { return data; }
    std::vector<double>& getData() { return data; }

private:
    int rows;
    int cols;
    std::vector<double> data;
};

#endif
