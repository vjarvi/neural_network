#include "layer.hpp"
#include <vector>
#include <functional>
#include <iostream>



Layer::Layer(int input_size, int neurons_amount, int batch_size, const std::string& activation)
    : input_size(input_size), neurons_amount(neurons_amount), batch_size(batch_size) {
    

    a = Matrix(batch_size, Vector(input_size));
    z = Matrix(batch_size, Vector(input_size));

    // + 1 row for bias terms
    weights = Matrix(neurons_amount + 1, Vector(input_size, 0.0f));
    gradients = Matrix(neurons_amount + 1, Vector(input_size, 0.0f));
    
    initialize_activation(activation);
}

void Layer::initialize_activation(const std::string& activation_str) {
    if (activation_str == "relu") {
        activation = activation::Relu;
        activation_derivative = activation::Relu_d;
    } else if (activation_str == "sigmoid") {
        activation = activation::Sigmoid;
        activation_derivative = activation::Sigmoid_d;
    } else if (activation_str == "tanh") {
        activation = activation::Tanh;
        activation_derivative = activation::Tanh_d;
    } else if (activation_str == "none") {
        activation = activation::None;
        activation_derivative = activation::None;
    } else {
        throw std::invalid_argument("bad activation name: " + activation_str);
    }
}

Matrix Layer::forward_pass(Matrix& input) {

    // matrix * vector multiplication
    Matrix output = utils::Matrix_mul(input, weights);

    // move input to a, and save pre-activation to z
    a = std::move(input);
    z = output;

    // apply activation function
    for (auto& row : output) {
        std::for_each(row.begin(), row.end(), [this](float& x) {
            x = activation(x);
        });
    }

    return output;
}

Matrix Layer::backward(Matrix prev_grad) {

    // apply derivation of activation element-wise to previous gradient
    for (size_t i = 0; i < prev_grad.size(); ++i) {
        std::transform(
            prev_grad[i].begin(), prev_grad[i].end(),
            z[i].begin(),
            prev_grad[i].begin(),
            [&](float grad_element, float z_element) {
                return grad_element * activation_derivative(z_element);
            }
        );
    }

    // gradient for other layers. transpose weight matrix and remove biases
    Matrix weights_T = weights;
    weights_T.pop_back();
    utils::Matrix_transpose(weights_T);
    Matrix output = utils::Matrix_mul(prev_grad, weights_T);

    // gradient for weights
    Matrix a_T = a;
    utils::Matrix_transpose(a_T);
    // remove last vector of ones (needed for biases)
    a_T.pop_back();
    Matrix weight_grad = utils::Matrix_mul(a_T, prev_grad);

    // gradient for biases
    Matrix biases_grad(1, Vector(prev_grad[0].size(), 0.0f));
    for (size_t i = 0; i < prev_grad.size(); ++i) {
        for (size_t j = 0; j < prev_grad[0].size(); ++j) {
            biases_grad[0][j] += prev_grad[i][j];
        } 
    }

    // update layers gradients
    // for weights
    for (size_t i = 0; i < gradients.size() - 1; i++) {
        for (size_t j = 0; j < gradients[0].size(); j++) {
            gradients[i][j] += weight_grad[i][j];
        }
    }
    // for biases
    for (size_t i = 0; i < gradients[-1].size(); i++) {
        gradients[-1][i] += biases_grad[0][i];;
    }

    // send output to differentation chain
    return output;
}

