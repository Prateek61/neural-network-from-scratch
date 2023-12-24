// File: include/Layer.h
// Purpose: Header file for Layer class

#pragma once

#include <memory> // std::unique_ptr

#include "NeuralNetwork/Matrix.h" // nn::Matrix
#include "NeuralNetwork/ActivationFunction.h" // nn::activation_functions::ActivationFunction

namespace nn
{
	class Layer
	{
	private:
		/// <summary>
		/// Activation matrix of this layer (columns are neurons, rows are batch size)
		/// </summary>
		std::unique_ptr<nn::Matrix<float>> activations_;

		/// <summary>
		/// Activation matrix produced by the sum of the weights and biases prior to the activation function
		/// </summary>
		std::unique_ptr<nn::Matrix<float>> sums_;

		/// <summary>
		/// Weights matrix of this layer
		/// </summary>
		std::unique_ptr<nn::Matrix<float>> weights_;

		/// <summary>
		/// Biases matrix of this layer
		/// </summary>
		std::unique_ptr<nn::Matrix<float>> biases_;

		/// <summary>
		/// Delta of the weights matrix of this layer
		/// </summary>
		std::unique_ptr<nn::Matrix<float>> delta_activations_;

		/// <summary>
		/// Delta of the biases matrix of this layer
		/// </summary>
		std::unique_ptr<nn::Matrix<float>> delta_sums_;

		/// <summary>
		/// Delta of the weights matrix of this layer
		/// </summary>
		std::unique_ptr<nn::Matrix<float>> delta_weights_;

		/// <summary>
		/// Delta of the biases matrix of this layer
		/// </summary>
		std::unique_ptr<nn::Matrix<float>> delta_biases_;

		/// <summary>
		/// Neuron count of this layer
		/// </summary>
		size_t neuron_count_;

		/// <summary>
		/// Batch size of this layer
		/// </summary>
		size_t batch_size_;

		/// <summary>
		/// Activation function of this layer (default is sigmoid) (owned)
		/// </summary>
		std::unique_ptr<nn::activation_functions::ActivationFunction> activation_function_;

	public:
		/// <summary>
		/// Default constructor
		/// </summary>
		Layer();

		/// <summary>
		/// Initializes the layer with the given neuron count and batch size
		/// </summary>
		/// <param name="neuron_count">Number of neurons in the layer</param>
		/// <param name="batch_size">The batch size for the layer</param>
		Layer(const size_t neuron_count, const size_t batch_size);

		/// <summary>
		/// Initializes the layer with the given neuron count, batch size and activation function
		/// </summary>
		/// <param name="neuron_count">Neuron count of this layer</param>
		/// <param name="batch_size">Batch size of this layer</param>
		/// <param name="previous_layer_neuron_count">Neuron count of the previous layer</param>
		Layer(const size_t neuron_count, const size_t batch_size, const size_t previous_layer_neuron_count);

		/// <summary>
		/// Initializes the layer with the given neuron count, batch size and activation function
		/// </summary>
		/// <param name="neuron_count">Neuron count of this layer</param>
		/// <param name="batch_size">Batch size of this layer</param>
		/// <param name="previous_layer_neuron_count">Neuron count of the previous layer</param>
		/// <param name="activation_function">Activation function for this layer</param>
		Layer(const size_t neuron_count, const size_t batch_size, const size_t previous_layer_neuron_count, std::unique_ptr<nn::activation_functions::ActivationFunction> activation_function);

		/// <summary>
		/// Deletes the copy constructor
		/// </summary>
		Layer(const Layer& other) = delete;

		/// <summary>
		/// Destructor
		/// </summary>
		~Layer();

		/// <summary>
		/// Deletes the assignment operator
		/// </summary>
		Layer& operator=(const Layer& other) = delete;

		/// <summary>
		/// Initializes the layer with the given neuron count and batch size
		/// </summary>
		/// <param name="neuron_count">Number of neurons in the layer</param>
		/// <param name="batch_size">The batch size for the layer</param>
		void initialize(const size_t neuron_count, const size_t batch_size);

		/// <summary>
		/// Initializes the layer with the given neuron count and batch size
		/// </summary>
		/// <param name="neuron_count">Number of neurons in the layer</param>
		/// <param name="batch_size">The batch size for the layer</param>
		/// <param name="previous_layer_neuron_count">Neuron count of previous layer</param>
		void initialize(const size_t neuron_count, const size_t batch_size, const size_t previous_layer_neuron_count);

		/// <summary>
		/// Initializes the layer with the given neuron count, batch size and activation function
		/// </summary>
		/// <param name="neuron_count">Number of neurons in the layer</param>
		/// <param name="batch_size">The batch size for the layer</param>
		///	<param name="previous_layer_neuron_count">Neuron count of the previous layer</param>
		/// <param name="activation_function">Activation function for this layer</param>
		void initialize(const size_t neuron_count, const size_t batch_size, const size_t previous_layer_neuron_count, std::unique_ptr<nn::activation_functions::ActivationFunction> activation_function);

		/// <summary>
		/// Sets the activation function for this layer
		/// </summary>
		/// <param name="activation_function">Activation function for this layer</param>
		void set_activation_function(std::unique_ptr<nn::activation_functions::ActivationFunction> activation_function);

		/// <summary>
		/// Sets the biases matrix of this layer (does not copy, takes ownership)
		/// </summary>
		/// <param name="activations">Activations matrix to set in this layer</param>
		void set_activations(std::unique_ptr<nn::Matrix<float>> activations);

		/// <summary>
		/// Sets the activation matrix of this layer (copies)
		/// </summary>
		/// <param name="activations">Activations matrix to set in this layer</param>
		void set_activations(const Matrix<float>& activations);

		/// <summary>
		/// Sets the weights matrix of this layer (does not copy, takes ownership)
		/// </summary>
		/// <param name="weights">Weights matrix to set in this layer</param>
		void set_weights(std::unique_ptr<nn::Matrix<float>> weights);

		/// <summary>
		/// Sets the weights matrix of this layer (copies)
		/// </summary>
		/// <param name="weights">Weights matrix to set in this layer</param>
		void set_weights(const Matrix<float>& weights);

		/// <summary>
		/// Sets the biases matrix of this layer (does not copy, takes ownership)
		/// </summary>
		/// <param name="biases">Biases matrix to set in this layer</param>
		void set_biases(std::unique_ptr<nn::Matrix<float>> biases);

		/// <summary>
		/// Sets the biases matrix of this layer (copies)
		/// </summary>
		/// <param name="biases">Biases matrix to set in this layer</param>
		void set_biases(const Matrix<float>& biases);

		/// <summary>
		/// Resets the batch size of this layer and re-initializes the affected matrices
		/// </summary>
		/// <param name="batch_size"></param>
		void change_batch_size(const size_t batch_size);

		/// <summary>
		/// Returns the neuron count of this layer
		/// </summary>
		[[nodiscard]] size_t get_neuron_count() const;

		/// <summary>
		/// Returns the batch size of this layer
		/// </summary>
		/// <returns></returns>
		[[nodiscard]] size_t get_batch_size() const;

		/// <summary>
		/// Returns the activation function of this layer
		/// </summary>
		[[nodiscard]] const nn::activation_functions::ActivationFunction* get_activation_function() const;

		/// <summary>
		/// Returns the activation matrix of this layer
		/// </summary>
		[[nodiscard]] const Matrix<float>& get_activations() const;

		/// <summary>
		/// Returns the sums matrix of this layer
		/// </summary>
		/// <returns></returns>
		[[nodiscard]] const Matrix<float>& get_sums() const;

		/// <summary>
		/// Returns the weights matrix of this layer
		/// </summary>
		[[nodiscard]] const Matrix<float>& get_weights() const;

		/// <summary>
		/// Returns the biases matrix of this layer
		/// </summary>
		[[nodiscard]] const Matrix<float>& get_biases() const;

		/// <summary>
		/// Returns the delta activations matrix of this layer
		/// </summary>
		[[nodiscard]] const Matrix<float>& get_delta_activations() const;

		/// <summary>
		/// Returns the delta sums matrix of this layer
		/// </summary>
		[[nodiscard]] const Matrix<float>& get_delta_sums() const;

		/// <summary>
		/// Returns the delta weights matrix of this layer
		/// </summary>
		[[nodiscard]] const Matrix<float>& get_delta_weights() const;

		/// <summary>
		/// Returns the delta biases matrix of this layer
		/// </summary>
		[[nodiscard]] const Matrix<float>& get_delta_biases() const;

		/// <summary>
		/// Resets the layer to uninitialized state
		/// </summary>
		void reset();

		/// <summary>
		/// Runs forward propagation on this layer
		/// </summary>
		/// <param name="previous_layer">Previous Layer</param>
		void feed_forward(const Layer& previous_layer);

		/// <summary>
		/// Runs back propagation on this layer
		/// </summary>
		/// <param name="next_layer">Next layer</param>
		/// <param name="previous_layer">Previous Layer</param>
		void back_propagate(const Layer& next_layer, const Layer& previous_layer);

		/// <summary>
		/// Runs back propagation on this layer with the given expected activations
		/// </summary>
		/// <param name="expected_activations">Expected output of the network</param>
		void back_propagate(const Matrix<float>& expected_activations);

		/// <summary>
		/// Updates the weights and biases of this layer
		/// </summary>
		void update_weights_and_biases(const float learning_rate);
	};
}