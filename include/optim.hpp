#ifndef OPTIM_HPP
#define OPTIM_HPP

#include "layer.hpp"
#include "utils.hpp"
#include <vector>
#include <functional>

class Optimizer {
public:
    Optimizer(Matrix* gradients, Matrix* weights)
    : gradients(gradients), weights(weights) {}
    virtual void step() = 0;
    void zero_grad();

protected:
    Matrix* gradients;     // pointer to gradients in layer class
    Matrix* weights;       // pointer to weights
};

class SGD : public Optimizer {
public:
    SGD(Matrix* gradients, Matrix* weights, float lr)
        : Optimizer(gradients, weights), learning_rate(lr) {}
    
    void step() override;

private:
    float learning_rate;
};

class Adam : public Optimizer {
public:
    Adam(Matrix* gradients, Matrix* weights, float lr, 
        float beta1 = 0.9, float beta2 = 0.999, float epsilon = 1e-08);
    
    void step() override;

private:
    float learning_rate;
    float beta1;
    float beta2;
    float epsilon;
    int t = 0;
    std::vector<Matrix> m;
    std::vector<Matrix> v;
};

#endif // OPTIM_HPP

