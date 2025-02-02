#ifndef LAYER_HPP
#define LAYER_HPP

#include <utils.hpp>
#include <vector>
#include <iostream>
#include <functional>
#include <stdexcept>


class Layer {
public:
    Layer(int input_size, int neurons_amount, int batch_size, const std::string& activation);


    Matrix forward_pass(Matrix& input);

    // backpropagates through this layer by first applying derivative of activation to prev_grad.
    // and then multiplying it with dz/da.
    Matrix backward(Matrix prev_grad); // prev_grad dims: batch x output_dim
    
    ~Layer();

private:
    int input_size;                    // Size of input vector
    int neurons_amount;                // Number of neurons in the layer
    int batch_size = 1;                // batch_size for gradient descent
    Matrix z;                          // pre-activation vector stored for backpropagation
    Matrix a;                          // Post-activation vector for backpropagation
    Matrix weights;                    // Layer weights
    Matrix gradients;                  // Gradients of the loss with respect to weights (&biases)

    std::function<float(float)> activation;
    std::function<float(float)> activation_derivative;

    void initialize_activation(const std::string& activation);
};


#endif // LAYER_HPP