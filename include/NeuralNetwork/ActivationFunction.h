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
	};


	/// <summary>
	/// Sigmoid activation function
	/// </summary>
	class Sigmoid final : public ActivationFunction
	{
	public:
		/// <summary>
		/// Performs the activation function on the input value
		/// </summary>
		/// <param name="x">Input Value</param>
		/// <returns>Value of the function</returns>
		[[nodiscard]] static float activation_function(const float x);

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
	};


	class ReLU final : public ActivationFunction
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
	};


	class LeakyReLU final : public ActivationFunction
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
	};


	class Tanh final : public ActivationFunction
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
	};

	class SoftMax final : public ActivationFunction
	{
	public:
		/// <summary>
		/// Performs the activation function on the input matrix
		/// </summary>
		/// <param name="mat">Input matrix</param>
		void activate(Matrix<float>& mat) override;

		/// <summary>
		/// Performs the activation function derivative on the input matrix
		/// </summary>
		/// <param name="mat">Input matrix</param>
		void derivative(Matrix<float>& mat) override;
	};
}
