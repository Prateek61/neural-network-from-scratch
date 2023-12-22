#pragma once
#include "matrix.h"

namespace nn::activation_functions
{
	/// <summary>
	/// Interface for activation functions
	/// </summary>
	class ActivationFunction
	{
	public:
		/// <summary>
		/// Activation function
		/// </summary>
		/// <param name="x">Input</param>
		/// <returns>Function value at x</returns>
		virtual void activate(Matrix<float> &x) = 0;

		/// <summary>
		/// Derivative of the activation function
		/// </summary>
		/// <param name="x">Input</param>
		/// <returns>Derivative value at x</returns>
		virtual void derivative(Matrix<float> &x) = 0;

		/// <summary>
		/// Activation function
		/// </summary>
		/// <param name="x">Input</param>
		/// <returns>Function value at x</returns>
		virtual void activate(Matrix<double> &x) = 0;

		/// <summary>
		/// Derivative of the activation function
		/// </summary>
		/// <param name="x">Input</param>
		/// <returns>Derivative value at x</returns>
		virtual void derivative(Matrix<double> &x) = 0;
	};


	/// <summary>
	/// Sigmoid activation function
	/// </summary>
	class Sigmoid : public ActivationFunction
	{
	public:
		/// <summary>
		/// Activation function
		/// </summary>
		/// <param name="x">Input</param>
		/// <returns>Function value at x</returns>
		void activate(Matrix<float> &x) override;

		/// <summary>
		/// Derivative of the activation function
		/// </summary>
		/// <param name="x">Input</param>
		/// <returns>Derivative value at x</returns>
		void derivative(Matrix<float> &x) override;

		/// <summary>
		/// Activation function
		/// </summary>
		/// <param name="x">Input</param>
		/// <returns>Function value at x</returns>
		void activate(Matrix<double> &x) override;

		/// <summary>
		/// Derivative of the activation function
		/// </summary>
		/// <param name="x">Input</param>
		/// <returns>Derivative value at x</returns>
		void derivative(Matrix<double> &x) override;
	};


	class ReLU : public ActivationFunction
	{
	public:
		/// <summary>
		/// Activation function
		/// </summary>
		/// <param name="x">Input</param>
		/// <returns>Function value at x</returns>
		void activate(Matrix<float> &x) override;

		/// <summary>
		/// Derivative of the activation function
		/// </summary>
		/// <param name="x">Input</param>
		/// <returns>Derivative value at x</returns>
		void derivative(Matrix<float> &x) override;

		/// <summary>
		/// Activation function
		/// </summary>
		/// <param name="x">Input</param>
		/// <returns>Function value at x</returns>
		void activate(Matrix<double> &x) override;

		/// <summary>
		/// Derivative of the activation function
		/// </summary>
		/// <param name="x">Input</param>
		/// <returns>Derivative value at x</returns>
		void derivative(Matrix<double> &x) override;
	};


	class LeakyReLU : public ActivationFunction
	{
	public:
		/// <summary>
		/// Activation function
		/// </summary>
		/// <param name="x">Input</param>
		/// <returns>Function value at x</returns>
		void activate(Matrix<float> &x) override;

		/// <summary>
		/// Derivative of the activation function
		/// </summary>
		/// <param name="x">Input</param>
		/// <returns>Derivative value at x</returns>
		void derivative(Matrix<float> &x) override;

		/// <summary>
		/// Activation function
		/// </summary>
		/// <param name="x">Input</param>
		/// <returns>Function value at x</returns>
		void activate(Matrix<double> &x) override;

		/// <summary>
		/// Derivative of the activation function
		/// </summary>
		/// <param name="x">Input</param>
		/// <returns>Derivative value at x</returns>
		void derivative(Matrix<double> &x) override;
	};


	class Tanh : public ActivationFunction
	{
	public:
		/// <summary>
		/// Activation function
		/// </summary>
		/// <param name="x">Input</param>
		/// <returns>Function value at x</returns>
		void activate(Matrix<float> &x) override;

		/// <summary>
		/// Derivative of the activation function
		/// </summary>
		/// <param name="x">Input</param>
		/// <returns>Derivative value at x</returns>
		void derivative(Matrix<float> &x) override;

		/// <summary>
		/// Activation function
		/// </summary>
		/// <param name="x">Input</param>
		/// <returns>Function value at x</returns>
		void activate(Matrix<double> &x) override;

		/// <summary>
		/// Derivative of the activation function
		/// </summary>
		/// <param name="x">Input</param>
		/// <returns>Derivative value at x</returns>
		void derivative(Matrix<double> &x) override;
	};
}
