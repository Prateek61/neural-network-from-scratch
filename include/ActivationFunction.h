#pragma once

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
		virtual float activate(float x) = 0;

		/// <summary>
		/// Derivative of the activation function
		/// </summary>
		/// <param name="x">Input</param>
		/// <returns>Derivative value at x</returns>
		virtual float derivative(float x) = 0;

		/// <summary>
		/// Activation function
		/// </summary>
		/// <param name="x">Input</param>
		/// <returns>Function value at x</returns>
		virtual double activate(double x) = 0;

		/// <summary>
		/// Derivative of the activation function
		/// </summary>
		/// <param name="x">Input</param>
		/// <returns>Derivative value at x</returns>
		virtual double derivative(double x) = 0;
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
		float activate(float x) override;

		/// <summary>
		/// Derivative of the activation function
		/// </summary>
		/// <param name="x">Input</param>
		/// <returns>Derivative value at x</returns>
		float derivative(float x) override;

		/// <summary>
		/// Activation function
		/// </summary>
		/// <param name="x">Input</param>
		/// <returns>Function value at x</returns>
		double activate(double x) override;

		/// <summary>
		/// Derivative of the activation function
		/// </summary>
		/// <param name="x">Input</param>
		/// <returns>Derivative value at x</returns>
		double derivative(double x) override;
	};


	class ReLU : public ActivationFunction
	{
	public:
		/// <summary>
		/// Activation function
		/// </summary>
		/// <param name="x">Input</param>
		/// <returns>Function value at x</returns>
		float activate(float x) override;

		/// <summary>
		/// Derivative of the activation function
		/// </summary>
		/// <param name="x">Input</param>
		/// <returns>Derivative value at x</returns>
		float derivative(float x) override;

		/// <summary>
		/// Activation function
		/// </summary>
		/// <param name="x">Input</param>
		/// <returns>Function value at x</returns>
		double activate(double x) override;

		/// <summary>
		/// Derivative of the activation function
		/// </summary>
		/// <param name="x">Input</param>
		/// <returns>Derivative value at x</returns>
		double derivative(double x) override;
	};


	class LeakyReLU : public ActivationFunction
	{
	public:
		/// <summary>
		/// Activation function
		/// </summary>
		/// <param name="x">Input</param>
		/// <returns>Function value at x</returns>
		float activate(float x) override;

		/// <summary>
		/// Derivative of the activation function
		/// </summary>
		/// <param name="x">Input</param>
		/// <returns>Derivative value at x</returns>
		float derivative(float x) override;

		/// <summary>
		/// Activation function
		/// </summary>
		/// <param name="x">Input</param>
		/// <returns>Function value at x</returns>
		double activate(double x) override;

		/// <summary>
		/// Derivative of the activation function
		/// </summary>
		/// <param name="x">Input</param>
		/// <returns>Derivative value at x</returns>
		double derivative(double x) override;
	};


	class Tanh : public ActivationFunction
	{
	public:
		/// <summary>
		/// Activation function
		/// </summary>
		/// <param name="x">Input</param>
		/// <returns>Function value at x</returns>
		float activate(float x) override;

		/// <summary>
		/// Derivative of the activation function
		/// </summary>
		/// <param name="x">Input</param>
		/// <returns>Derivative value at x</returns>
		float derivative(float x) override;

		/// <summary>
		/// Activation function
		/// </summary>
		/// <param name="x">Input</param>
		/// <returns>Function value at x</returns>
		double activate(double x) override;

		/// <summary>
		/// Derivative of the activation function
		/// </summary>
		/// <param name="x">Input</param>
		/// <returns>Derivative value at x</returns>
		double derivative(double x) override;
	};
}
