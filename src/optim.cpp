#include "optim.hpp"


void Optimizer::zero_grad() {
    
}

void SGD::step() {
    
}

Adam::Adam(Matrix* gradients, Matrix* weights, float lr, float beta1, float beta2, float epsilon)
    : Optimizer(gradients, weights), learning_rate(lr), beta1(beta1), beta2(beta2), epsilon(epsilon) {
     
}

void Adam::step() {

}