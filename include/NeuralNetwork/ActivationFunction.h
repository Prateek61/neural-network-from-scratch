// File: src/NeuralNetwork/NeuralNetwork.cpp
// Purpose: Header file for ActivationFunction class.

#pragma once

#include "NeuralNetwork/Matrix.h" // nn::Matrix

namespace nn::activation_functions
{
	/// <summary>
	/// Interface for activation functions
	/// </summary>
	class ActivationFunction
	{
	public:
		/// <summary>
		/// Virtual destructor
		/// </summary>
		virtual ~ActivationFunction() = default;

		/// <summary>
		/// Performs the activation function on the input matrix
		/// </summary>
		/// <param name="mat">Input matrix</param>
		virtual void activate(Matrix<float> &mat) = 0;

		/// <summary>
		/// Performs the activation function derivative on the input matrix
		/// </summary>
		/// <param name="mat">Input matrix</param>
		virtual void derivative(Matrix<float> &mat) = 0;

		/// <summary>
		/// Performs the activation function on the input matrix
		/// </summary>
		/// <param name="mat">Input matrix</param>
		virtual void activate(Matrix<double> &mat) = 0;

		/// <summary>
		/// Performs the activation function derivative on the input matrix
		/// </summary>
		/// <param name="mat">Input matrix</param>
		virtual void derivative(Matrix<double> &mat) = 0;
	};


	/// <summary>
	/// Sigmoid activation function
	/// </summary>
	class Sigmoid : public ActivationFunction
	{
	public:
		/// <summary>
		/// Performs the activation function on the input matrix
		/// </summary>
		/// <param name="mat">Input matrix</param>
		void activate(Matrix<float> &mat) override;

		/// <summary>
		/// Performs the activation function derivative on the input matrix
		/// </summary>
		/// <param name="mat">Input matrix</param>
		void derivative(Matrix<float> &mat) override;

		/// <summary>
		/// Performs the activation function on the input matrix
		/// </summary>
		/// <param name="mat">Input matrix</param>
		void activate(Matrix<double> &mat) override;

		/// <summary>
		/// Performs the activation function derivative on the input matrix
		/// </summary>
		/// <param name="mat">Input matrix</param>
		void derivative(Matrix<double> &mat) override;
	};


	class ReLU : public ActivationFunction
	{
		const std::string activation_function_name_ = "ReLU";
	public:
		/// <summary>
		/// Performs the activation function on the input matrix
		/// </summary>
		/// <param name="mat">Input matrix</param>
		void activate(Matrix<float> &mat) override;

		/// <summary>
		/// Performs the activation function derivative on the input matrix
		/// </summary>
		/// <param name="mat">Input matrix</param>
		void derivative(Matrix<float> &mat) override;

		/// <summary>
		/// Performs the activation function on the input matrix
		/// </summary>
		/// <param name="mat">Input matrix</param>
		void activate(Matrix<double> &mat) override;

		/// <summary>
		/// Performs the activation function derivative on the input matrix
		/// </summary>
		/// <param name="mat">Input matrix</param>
		void derivative(Matrix<double> &mat) override;
	};


	class LeakyReLU : public ActivationFunction
	{
	public:
		/// <summary>
		/// Performs the activation function on the input matrix
		/// </summary>
		/// <param name="mat">Input matrix</param>
		void activate(Matrix<float> &mat) override;

		/// <summary>
		/// Performs the activation function derivative on the input matrix
		/// </summary>
		/// <param name="mat">Input matrix</param>
		void derivative(Matrix<float> &mat) override;

		/// <summary>
		/// Performs the activation function on the input matrix
		/// </summary>
		/// <param name="mat">Input matrix</param>
		void activate(Matrix<double> &mat) override;

		/// <summary>
		/// Performs the activation function derivative on the input matrix
		/// </summary>
		/// <param name="mat">Input matrix</param>
		void derivative(Matrix<double> &mat) override;
	};


	class Tanh : public ActivationFunction
	{
	public:
		/// <summary>
		/// Performs the activation function on the input matrix
		/// </summary>
		/// <param name="mat">Input matrix</param>
		void activate(Matrix<float> &mat) override;

		/// <summary>
		/// Performs the activation function derivative on the input matrix
		/// </summary>
		/// <param name="mat">Input matrix</param>
		void derivative(Matrix<float> &mat) override;

		/// <summary>
		/// Performs the activation function on the input matrix
		/// </summary>
		/// <param name="mat">Input matrix</param>
		void activate(Matrix<double> &mat) override;

		/// <summary>
		/// Performs the activation function derivative on the input matrix
		/// </summary>
		/// <param name="mat">Input matrix</param>
		void derivative(Matrix<double> &mat) override;
	};
}
