#include <ActivationFunction.h>
#include <cmath>
#include "Matrix.h"

void nn::activation_functions::Sigmoid::activate(Matrix<float>& mat)
{
	mat.perform_element_wise_operation([](const float x) -> float
	{
		return 1 / (1 + exp(-x));
	});
}

void nn::activation_functions::Sigmoid::activate(Matrix<double>& mat)
{
	mat.perform_element_wise_operation([](const double x) -> double
	{
		return 1 / (1 + exp(-x));
	});
}

void nn::activation_functions::Sigmoid::derivative(Matrix<float>& mat)
{
	mat.perform_element_wise_operation([](const float x) -> float
	{
		return (1 / (1 + exp(-x))) * (1 - 1 / (1 + exp(-x)));
	});
}

void nn::activation_functions::Sigmoid::derivative(Matrix<double>& mat)
{
	mat.perform_element_wise_operation([](const double x) -> double
	{
		return (1 / (1 + exp(-x))) * (1 - 1 / (1 + exp(-x)));
	});
}

void nn::activation_functions::ReLU::activate(Matrix<float>& mat)
{
	mat.perform_element_wise_operation([](const float x) -> float
	{
		return (x > 0) * x;
	});
}

void nn::activation_functions::ReLU::derivative(Matrix<float>& mat)
{
	mat.perform_element_wise_operation([](const float x) -> float
	{
		return (x > 0);
	});
}

void nn::activation_functions::ReLU::activate(Matrix<double>& mat)
{
	mat.perform_element_wise_operation([](const double x) -> double
	{
		return (x > 0) * x;
	});
}

void nn::activation_functions::ReLU::derivative(Matrix<double>& mat)
{
	mat.perform_element_wise_operation([](const double x) -> double
	{
		return (x > 0);
	});
}

void nn::activation_functions::Tanh::activate(Matrix<float>& mat)
{
	mat.perform_element_wise_operation([](const float x) -> float
	{
		return (exp(x) - exp(-x)) / (exp(x) + exp(-x));
	});
}

void nn::activation_functions::Tanh::derivative(Matrix<float>& mat)
{
	mat.perform_element_wise_operation([](const float x) -> float
	{
		return 1.0f - tanh(x) * tanh(x);
	});
}

void nn::activation_functions::Tanh::activate(Matrix<double>& mat)
{
	mat.perform_element_wise_operation([](const double x) -> double
	{
		return (exp(x) - exp(-x)) / (exp(x) + exp(-x));
	});
}

void nn::activation_functions::Tanh::derivative(Matrix<double>& mat)
{
	mat.perform_element_wise_operation([](const double x) -> double
	{
		return 1.0f - tanh(x) * tanh(x);
	});
}
