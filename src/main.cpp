#include <iostream>
#include "ActivationFunction.h"
#include "matrix.h"
#include "iostream"

int main()
{	

	nn::activation_functions::ReLU actfun;
	nn::Matrix<float> ma({{2.2,3.2},{-12.2,4.2}});
	// actfun.derivative(ma);
	
	std::cout<<ma;

	return 0;
}