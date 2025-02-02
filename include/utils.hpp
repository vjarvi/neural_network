#ifndef UTILS_HPP
#define UTILS_HPP

#include <algorithm>
#include <cmath>
#include <vector>
#include <stdexcept>

using Vector = std::vector<float>;
using Matrix = std::vector<std::vector<float>>;

// declarations for activation functions and derivatives
namespace activation {

    float Relu(const float &x);
    float Relu_d(const float &x);

    float Sigmoid(const float &x);
    float Sigmoid_d(const float &x);

    float Tanh(const float &x);
    float Tanh_d(const float &x);

    float None(const float &x);
}

// declarations for loss functions and derivatives
namespace loss_functions {

    float MSE(const std::vector<float> &t, const std::vector<float> &y);
    std::vector<float> MSE_d(const float &x);

    float BCE(const float &x);
    std::vector<float> BCE_d(const float &x);
}

// other  useful functions
namespace utils {
    
    std::vector<float> Softmax(std::vector<float> &x);
    std::vector<float> Softmax_d(std::vector<float> &x);

    // does matrix multiplication for matrix M and vector v
    std::vector<float> Matrix_vector(const std::vector<std::vector<float>>& M, const std::vector<float>& v);
    // matrix mul for vec and matrix
    Vector Vector_matrix(const Vector& rowVector, const Matrix& matrix);
    // does matrix mul for two matricies
    Matrix Matrix_mul(const Matrix& A, const Matrix& B);

    // transpose
    void Matrix_transpose(Matrix& M);
}

#endif // UTILS_HPP