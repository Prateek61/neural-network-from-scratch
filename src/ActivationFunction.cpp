#include <ActivationFunction.h>
#include "math.h"

float nn::activation_functions::Sigmoid::activate(float x){
    return 1/(1+exp(-x));
}

double nn::activation_functions::Sigmoid::activate(double x){
    return 1/(1+exp(-x));
}

float nn::activation_functions::Sigmoid::derivative(float x){

    return (1/(1+ exp(-x)))*(1-1/(1+exp(-x)));

}
double nn::activation_functions::Sigmoid::derivative(double x){

    return (1/(1+ exp(-x)))*(1-1/(1+exp(-x)));

}

float nn::activation_functions::ReLU::activate(float x){
    
    return (x>0) * x;
    
}
float nn::activation_functions::ReLU::derivative(float x){

    return (x>0);
    
}

double nn::activation_functions::ReLU::activate(double x){
    
    return (x>0) * x;
    
}
double nn::activation_functions::ReLU::derivative(double x){

    return (x>0);
    
}

float nn::activation_functions::Tanh::activate(float x){
    
    return (exp(x) - exp(-x)) / (exp(x) + exp(-x));

    
}
float nn::activation_functions::Tanh::derivative(float x){

    return 1.0f - tanh(x) * tanh(x);

    
}

double nn::activation_functions::Tanh::activate(double x){
    
    return (exp(x) - exp(-x)) / (exp(x) + exp(-x));
    
}   
double nn::activation_functions::Tanh::derivative(double x){

    return 1.0f - tanh(x) * tanh(x);
    
}