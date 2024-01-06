// File: src/NeuralNetwork/ActivationFunction.cpp
// Purpose: Implementation file for ActivationFunction class.

#include "NeuralNetwork/ActivationFunction.h"

#include <cmath> // exp

#include "NeuralNetwork/Matrix.h" // nn::Matrix

float nn::activation_functions::Sigmoid::activation_function(const float x)
{
	return 1 / (1 + exp(-x));
}

void nn::activation_functions::Sigmoid::activate(Matrix<float>& mat)
{
	mat.perform_element_wise_operation([](const float x) -> float
	{
		return 1 / (1 + exp(-x));
	});
}

void nn::activation_functions::Sigmoid::derivative(Matrix<float>& mat)
{
	mat.perform_element_wise_operation([](const float x) -> float
	{
		return activation_function(x) * (1 - activation_function(x));
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

void nn::activation_functions::LeakyReLU::activate(Matrix<float>& mat)
{
	mat.perform_element_wise_operation([](const float x) -> float
	{
		return (x > 0) * x + (x <= 0) * 0.01f * x;
	});
}

void nn::activation_functions::LeakyReLU::derivative(Matrix<float>& mat)
{
	mat.perform_element_wise_operation([](const float x) -> float
	{
		return (x > 0) + (x <= 0) * 0.01f;
	});
}

void nn::activation_functions::SoftMax::activate(Matrix<float>& mat)
{
	for (size_t j = 0; j < mat.get_cols(); ++j)
	{
		float sum = 0.0f;
		for (size_t i = 0; i < mat.get_rows(); ++i)
		{
			sum += exp(mat(i, j));
		}

		for (size_t i = 0; i < mat.get_rows(); ++i)
		{
			mat(i, j) = exp(mat(i, j)) / sum;
		}
	}
}

void nn::activation_functions::SoftMax::derivative(Matrix<float>& mat)
{
	const auto temp = mat;
	mat.perform_element_wise_operation([](const float x) -> float
	{
		return 0;
	});

	for (size_t j = 0; j < mat.get_cols(); ++j)
	{
		for (size_t i = 0; i < mat.get_rows(); ++i)
		{
			for (size_t k = 0; k < mat.get_rows(); k++)
			{
				if (i == k)
				{
					mat(i, j) += temp(i, j) * (1 - temp(k, j));
				}
				else
				{
					mat(i, j) -= temp(i, j) * temp(k, j);
				}
			}
		}
	}
}
