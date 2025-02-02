#include "utils.hpp"
#include <algorithm>
#include <cmath>
#include <vector>
#include <iterator>
#include <iostream>
#include <numeric>
#include <stdexcept>

// Definitions of different activation function and their derivatives
namespace activation {

    float Relu(const float &x) { 
        return std::max(0.0f, x);    };

    float Relu_d(const float &x) {
        if (x < 0.0f) {
            return 0.0f;
        } else {
            return 1.0f; 
        }
    }

    float Sigmoid(const float &x) {
        return 1 / (1 + std::exp(-x));
    }

    float Sigmoid_d(const float &x) {
        return Sigmoid(x) * (1 - Sigmoid(x));
    }

    float Tanh(const float &x) {
        return (std::exp(2*x) - 1) / (std::exp(2*x) + 1);
    }

    float Tanh_d(const float &x) {
        return 1 - std::pow(Tanh(x), 2);
    }

    float None(const float &x) {return x;}
}

// Definitions of loss functions and their derivatives
namespace loss_functions {

    float MSE(const std::vector<float> &t, const std::vector<float> &y) {
        float error = 0;
        auto t_i = t.begin();
        auto y_i = y.begin();
        while (t_i != t.end() && y_i != y.end()) {
            error += std::pow(*t_i - *y_i, 2);
            ++t_i;
            ++y_i;
        }
        return error / t.size();
    }

    std::vector<float> MSE_d(const std::vector<float> &t, const std::vector<float> &y) {
        std::vector<float> gradients;
        int n = t.size();
        auto t_i = t.begin();
        auto y_i = y.begin();
        while (t_i != t.end() && y_i != y.end()) {
            gradients.push_back(2 / n * (*y_i - *t_i));
            ++t_i;
            ++y_i;
        }
        return gradients;
    }

    float BCE(const std::vector<int> &t, const std::vector<float> &p) {
        float error = 0;
        auto p_i = p.begin();
        auto t_i = t.begin();
        while (p_i != p.end() && t_i != t.end()) {
            error += *t_i * std::log(*p_i) + (1-*t_i) * std::log(1 - *p_i);
            ++t_i;
            ++p_i;
        }
        return - error / t.size(); 
    }
    std::vector<float> BCE_D(const std::vector<int> &t, const std::vector<float> &p) {
        std::vector<float> gradients;
        auto p_i = p.begin();
        auto t_i = t.begin();
        while(p_i != p.end() && t_i != t.end()) {
            gradients.push_back((*p_i - *t_i) / (*p_i * (1 - *p_i)));
            ++t_i;
            ++p_i;
        }
        return gradients;
    }
}

// other  useful functions
namespace utils {
    
    std::vector<float> Softmax(std::vector<float> &x) {
        float sum = std::accumulate(x.begin(), x.end(), 0.0f,
        [](float acc, float val) {
            return acc + std::exp(val);
            }
        );
        for (auto elem : x) {
            elem = std::exp(elem) / sum;
        }
        return x;
    }

    // tarkista että on oikein.
    std::vector<float> Softmax_d(std::vector<float> &dL_ds, std::vector<float> &s) {
        std::vector<float> gradient;
        gradient.reserve(s.size());

        float dot_product = 0.0;
        auto dL_ds_i = dL_ds.begin();
        auto s_i = s.begin();
        while(dL_ds_i != dL_ds.end() && s_i != s.end()) {
            dot_product += *dL_ds_i * *s_i;
            ++dL_ds_i;
            ++s_i;
        }

        dL_ds_i = dL_ds.begin();
        s_i = s.begin();
        while(dL_ds_i != dL_ds.end() && s_i != s.end()) {
            gradient.push_back(*s_i * (*dL_ds_i - dot_product));
        }
        return gradient;
    }

    std::vector<float> Matrix_vector(const std::vector<std::vector<float>>& M, const std::vector<float>& v) {
        // Check dimensions for validity
        if (M.empty() || v.size() != M[0].size()) {
            throw std::invalid_argument("Invalid dimensions for matrix-vector multiplication.");
        }

        int rowsM = M.size();
        int colsM = M[0].size();

        std::vector<float> result(rowsM, 0.0f);

        // Perform multiplication
        for (int i = 0; i < rowsM; ++i) {
            for (int j = 0; j < colsM; ++j) {
                result[i] += M[i][j] * v[j];
            }
        }
        return result;
    }

    Vector Vector_matrix(const Vector& rowVector, const Matrix& matrix) {

        const size_t matrixRows = matrix.size();
        const size_t matrixCols = matrix[0].size();
        const size_t vectorSize = rowVector.size();

        Vector result(matrixCols, 0.0f);

        // Perform multiplication: rowVector * matrix
        for (size_t col = 0; col < matrixCols; ++col) {
            for (size_t row = 0; row < matrixRows; ++row) {
                result[col] += rowVector[row] * matrix[row][col];
            }
        }

        return result;
    }

    Matrix Matrix_mul(const Matrix& A, const Matrix& B) {

        // Validate matrix dimensions
        size_t rowsA = A.size();
        size_t colsA = A[0].size();
        size_t rowsB = B.size();
        size_t colsB = B[0].size();

        // Initialize result matrix with zeros
        Matrix result(rowsA, Vector(colsB, 0.0f));

        // Perform matrix multiplication
        for (size_t i = 0; i < rowsA; ++i) {
            for (size_t k = 0; k < colsA; ++k) {
                for (size_t j = 0; j < colsB; ++j) {
                    result[i][j] += A[i][k] * B[k][j];
                }
            }
        }

        return result;
    }


    void Matrix_transpose(Matrix& M) {
        size_t numRows = M.size();
        size_t numCols = M[0].size();
        // initialization
        Matrix transposed(numCols, Vector(numRows));

        for (size_t i = 0; i < numRows; ++i) {
            for (size_t j = 0; j < numCols; ++j) {
                transposed[j][i] = M[i][j];
        }
    }
    
    M = std::move(transposed);
    }
}

