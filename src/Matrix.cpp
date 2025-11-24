#include "Matrix.h"
#include <random>
#include <cassert>
#include <iomanip>
#include <cmath>
#include <stdexcept>
#include <thread>
#include <algorithm>

Matrix::Matrix(int rows, int cols) : rows(rows), cols(cols)
{
    data.resize(rows * cols, 0.0);
}

double &Matrix::at(int r, int c)
{
    return data[r * cols + c];
}

const double &Matrix::at(int r, int c) const
{
    return data[r * cols + c];
}

Matrix Matrix::multiply(const Matrix &other) const
{
    if (cols != other.rows)
    {
        throw std::invalid_argument("Matrix dimensions mismatch for multiplication");
    }
    Matrix result(rows, other.cols);

    // Cache-friendly matrix multiplication (ikj algorithm)
    // A is (rows x cols), B is (other.rows x other.cols)
    // C is (rows x other.cols)

    const int M = rows;
    const int K = cols;
    const int N = other.cols;

    // Pointers to data for faster access
    const double *A_ptr = data.data();
    const double *B_ptr = other.data.data();
    double *C_ptr = result.data.data();

    // Simple heuristic for threading: if total operations > 1M, use threads
    // Ops approx M * K * N
    long long ops = (long long)M * K * N;
    unsigned int num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0)
        num_threads = 4; // Fallback

    if (ops > 1000000 && num_threads > 1)
    {
        std::vector<std::thread> threads;
        int chunk_size = M / num_threads;

        auto worker = [&](int start_row, int end_row)
        {
            for (int i = start_row; i < end_row; ++i)
            {
                const double *rowA = A_ptr + i * K;
                double *rowC = C_ptr + i * N;

                for (int k = 0; k < K; ++k)
                {
                    double valA = rowA[k];
                    const double *rowB = B_ptr + k * N;
                    for (int j = 0; j < N; ++j)
                    {
                        rowC[j] += valA * rowB[j];
                    }
                }
            }
        };

        for (unsigned int t = 0; t < num_threads - 1; ++t)
        {
            threads.emplace_back(worker, t * chunk_size, (t + 1) * chunk_size);
        }
        // Main thread does the last chunk
        worker((num_threads - 1) * chunk_size, M);

        for (auto &t : threads)
        {
            t.join();
        }
    }
    else
    {
        // Single threaded fallback
        for (int i = 0; i < M; ++i)
        {
            const double *rowA = A_ptr + i * K;
            double *rowC = C_ptr + i * N;

            for (int k = 0; k < K; ++k)
            {
                double valA = rowA[k];
                const double *rowB = B_ptr + k * N;

                // This inner loop can be vectorized by the compiler
                for (int j = 0; j < N; ++j)
                {
                    rowC[j] += valA * rowB[j];
                }
            }
        }
    }
    return result;
}

Matrix Matrix::add(const Matrix &other) const
{
    if (rows != other.rows || cols != other.cols)
    {
        throw std::invalid_argument("Matrix dimensions mismatch for addition");
    }
    Matrix result(rows, cols);
    for (size_t i = 0; i < data.size(); i++)
    {
        result.data[i] = data[i] + other.data[i];
    }
    return result;
}

Matrix Matrix::subtract(const Matrix &other) const
{
    if (rows != other.rows || cols != other.cols)
    {
        throw std::invalid_argument("Matrix dimensions mismatch for subtraction");
    }
    Matrix result(rows, cols);
    for (size_t i = 0; i < data.size(); i++)
    {
        result.data[i] = data[i] - other.data[i];
    }
    return result;
}

Matrix Matrix::multiplyScalar(double scalar) const
{
    Matrix result(rows, cols);
    for (size_t i = 0; i < data.size(); i++)
    {
        result.data[i] = data[i] * scalar;
    }
    return result;
}

Matrix Matrix::hadamard(const Matrix &other) const
{
    if (rows != other.rows || cols != other.cols)
    {
        throw std::invalid_argument("Matrix dimensions mismatch for hadamard product");
    }
    Matrix result(rows, cols);
    for (size_t i = 0; i < data.size(); i++)
    {
        result.data[i] = data[i] * other.data[i];
    }
    return result;
}

Matrix Matrix::addBias(const Matrix &bias) const
{
    Matrix result(rows, cols);
    if (bias.getRows() == cols && bias.getCols() == 1)
    {
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                result.at(i, j) = at(i, j) + bias.at(j, 0);
            }
        }
    }
    else if (bias.getRows() == 1 && bias.getCols() == cols)
    {
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                result.at(i, j) = at(i, j) + bias.at(0, j);
            }
        }
    }
    else
    {
        throw std::invalid_argument("Invalid bias dimensions for broadcasting");
    }
    return result;
}

Matrix Matrix::transpose() const
{
    Matrix result(cols, rows);
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            result.at(j, i) = data[i * cols + j];
        }
    }
    return result;
}

Matrix Matrix::map(std::function<double(double)> func) const
{
    Matrix result(rows, cols);

    unsigned int num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0)
        num_threads = 4;
    size_t total_elements = data.size();

    if (total_elements > 100000 && num_threads > 1)
    {
        std::vector<std::thread> threads;
        size_t chunk_size = total_elements / num_threads;

        auto worker = [&](size_t start, size_t end)
        {
            for (size_t i = start; i < end; ++i)
            {
                result.data[i] = func(data[i]);
            }
        };

        for (unsigned int t = 0; t < num_threads - 1; ++t)
        {
            threads.emplace_back(worker, t * chunk_size, (t + 1) * chunk_size);
        }
        worker((num_threads - 1) * chunk_size, total_elements);

        for (auto &t : threads)
        {
            t.join();
        }
    }
    else
    {
        for (size_t i = 0; i < data.size(); i++)
        {
            result.data[i] = func(data[i]);
        }
    }
    return result;
}

Matrix Matrix::multiply(const Matrix &a, const Matrix &b)
{
    return a.multiply(b);
}

Matrix Matrix::fromVector(const std::vector<double> &vec)
{
    Matrix result(vec.size(), 1);
    for (size_t i = 0; i < vec.size(); i++)
    {
        result.data[i] = vec[i];
    }
    return result;
}

void Matrix::randomize()
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-1.0, 1.0);

    for (size_t i = 0; i < data.size(); i++)
    {
        data[i] = dis(gen);
    }
}

void Matrix::print() const
{
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            std::cout << std::setw(10) << std::fixed << std::setprecision(4) << at(i, j) << " ";
        }
        std::cout << std::endl;
    }
}
