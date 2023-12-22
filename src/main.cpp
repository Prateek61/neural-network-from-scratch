#include <iostream>
#include "ActivationFunction.h"
#include "matrix.h"

int main()
{	

	nn::activation_functions::ReLU actfun;
	nn::Matrix<float> ma({{2.2,3.2},{-1.2,4.2}});

	return 0;
}