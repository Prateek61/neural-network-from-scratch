// File: src/Layer.cpp
// Purpose: Source file for Layer objects, which represent a single layer of a neural network.

#include "Layer.h"

nn::Layer::Layer() = default;

nn::Layer::Layer(const size_t neuron_count, const size_t batch_size)
{
	this->initialize(neuron_count, batch_size);
}

nn::Layer::Layer(const size_t neuron_count, const size_t batch_size, const size_t previous_layer_neuron_count)
{
	this->initialize(neuron_count, batch_size, previous_layer_neuron_count);
}

nn::Layer::Layer(const size_t neuron_count, const size_t batch_size, const size_t previous_layer_neuron_count,
                 std::unique_ptr<activation_functions::ActivationFunction> activation_function)
{
	this->initialize(neuron_count, batch_size, previous_layer_neuron_count, std::move(activation_function));
}

nn::Layer::~Layer() = default;

void nn::Layer::initialize(const size_t neuron_count, const size_t batch_size)
{
	// Check if the layer has already been initialized.
	if (this->neuron_count_ != 0)
	{
		throw std::runtime_error("Layer has already been initialized.");
	}

	this->neuron_count_ = neuron_count;
	this->batch_size_ = batch_size;

	// Initialize the matrices.
	this->activations_ = std::make_unique<Matrix<float>>(neuron_count, batch_size);
}

void nn::Layer::initialize(const size_t neuron_count, const size_t batch_size, const size_t previous_layer_neuron_count)
{
	// Check if the layer has already been initialized.
	if (this->neuron_count_ != 0)
	{
		throw std::runtime_error("Layer has already been initialized.");
	}

	this->neuron_count_ = neuron_count;
	this->batch_size_ = batch_size;

	// Initialize the matrices
	this->activations_ = std::make_unique<Matrix<float>>(neuron_count, batch_size);
	this->weights_ = std::make_unique<Matrix<float>>(neuron_count, previous_layer_neuron_count);
	this->biases_ = std::make_unique<Matrix<float>>(neuron_count, 1);
	this->sums_ = std::make_unique<Matrix<float>>(neuron_count, batch_size);
	// Initialize the delta matrices
	this->delta_activations_ = std::make_unique<Matrix<float>>(neuron_count, batch_size);
	this->delta_weights_ = std::make_unique<Matrix<float>>(neuron_count, previous_layer_neuron_count);
	this->biases_ = std::make_unique<Matrix<float>>(neuron_count, 1);
	this->delta_sums_ = std::make_unique<Matrix<float>>(neuron_count, batch_size);

	// Randomize the weights and biases
	this->weights_->randomize(-1.0f, 1.0f);
	this->biases_->randomize(-1.0f, 1.0f);

	// Initialize the activation function to the sigmoid function
	this->activation_function_ = std::make_unique<activation_functions::Sigmoid>();
}

void nn::Layer::initialize(const size_t neuron_count, const size_t batch_size, const size_t previous_layer_neuron_count, std::unique_ptr<nn::activation_functions::ActivationFunction> activation_function)
{
	this->initialize(neuron_count, batch_size, previous_layer_neuron_count);

	// Set the activation function
	this->activation_function_ = std::move(activation_function);
}
