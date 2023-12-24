// File: src/NeuralNetwork/Layer.cpp
// Purpose: Source file for Layer class, which represent a single layer of a neural network.

#include "NeuralNetwork/Layer.h"

nn::Layer::Layer() = default;

nn::Layer::Layer(const size_t neuron_count, const size_t batch_size)
	: neuron_count_(0), batch_size_(0)
{
	this->initialize(neuron_count, batch_size);
}

nn::Layer::Layer(const size_t neuron_count, const size_t batch_size, const size_t previous_layer_neuron_count)
	: neuron_count_(0), batch_size_(0)
{
	this->initialize(neuron_count, batch_size, previous_layer_neuron_count);
}

nn::Layer::Layer(const size_t neuron_count, const size_t batch_size, const size_t previous_layer_neuron_count,
                 std::unique_ptr<activation_functions::ActivationFunction> activation_function)
	                 : neuron_count_(0), batch_size_(0)
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
	// Set the actication function to Sigmoid.
	this->activation_function_ = std::make_unique<activation_functions::Sigmoid>();
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

void nn::Layer::initialize(const size_t neuron_count, const size_t batch_size, const size_t previous_layer_neuron_count,
                           std::unique_ptr<activation_functions::ActivationFunction> activation_function)
{
	this->initialize(neuron_count, batch_size, previous_layer_neuron_count);

	// Delete the previous activation function
	this->activation_function_.reset();
	// Set the activation function
	this->activation_function_ = std::move(activation_function);
}

void nn::Layer::set_activation_function(std::unique_ptr<activation_functions::ActivationFunction> activation_function)
{
	// Delete the previous activation function
	this->activation_function_.reset();
	// Set the activation function
	this->activation_function_ = std::move(activation_function);
}

void nn::Layer::set_activations(std::unique_ptr<Matrix<float>> activations)
{
	// Check if the layer is initialized
	if (this->neuron_count_ == 0)
	{
		throw std::runtime_error("Layer is not initialized.");
	}
	// Check if the activations matrix is the correct size
	if (activations->get_rows() != this->neuron_count_ || activations->get_cols() != this->batch_size_)
	{
		throw std::runtime_error("Activations matrix is not the correct size.");
	}

	// Delete the previous activations matrix
	this->activations_.reset();
	// Set the activations matrix
	this->activations_ = std::move(activations);
}

void nn::Layer::set_activations(const Matrix<float>& activations)
{
	// Check if the layer is initialized
	if (this->neuron_count_ == 0)
	{
		throw std::runtime_error("Layer is not initialized.");
	}
	// Check if the activations matrix is the correct size
	if (activations.get_rows() != this->neuron_count_ || activations.get_cols() != this->batch_size_)
	{
		throw std::runtime_error("Activations matrix is not the correct size.");
	}

	// Loop through the activations matrix and set the activations
	for (size_t i = 0; i < activations.get_rows() * activations.get_cols(); i++)
	{
		this->activations_->operator[](i) = activations[i];
	}
}

void nn::Layer::set_weights(std::unique_ptr<Matrix<float>> weights)
{
	// Check if weights is initialized
	if (this->weights_ == nullptr)
	{
		throw std::runtime_error("Weights matrix is not initialized.");
	}
	// Check if the size of the weights matrix is correct
	if (weights->get_rows() != this->neuron_count_ || weights->get_cols() != this->weights_->get_cols())
	{
		throw std::runtime_error("Weights matrix is not the correct size.");
	}

	// Delete the previous weights matrix
	this->weights_.reset();
	// Set the weights matrix
	this->weights_ = std::move(weights);
}

void nn::Layer::set_weights(const Matrix<float>& weights)
{
	// Check if weights is initialized
	if (this->weights_ == nullptr)
	{
		throw std::runtime_error("Weights matrix is not initialized.");
	}
	// Check if the size of the weights matrix is correct
	if (weights.get_rows() != this->neuron_count_ || weights.get_cols() != this->weights_->get_cols())
	{
		throw std::runtime_error("Weights matrix is not the correct size.");
	}

	// Loop through the weights matrix and set the weights
	for (size_t i = 0; i < weights.get_rows() * weights.get_cols(); i++)
	{
		this->weights_->operator[](i) = weights[i];
	}
}

void nn::Layer::set_biases(std::unique_ptr<nn::Matrix<float>> biases)
{
	// Check if biases is initialized
	if (this->biases_ == nullptr)
	{
		throw std::runtime_error("Biases matrix is not initialized.");
	}
	// Check if the size of the biases matrix is correct
	if (biases->get_rows() != this->neuron_count_ || biases->get_cols() != this->biases_->get_cols())
	{
		throw std::runtime_error("Biases matrix is not the correct size.");
	}

	// Delete the previous biases matrix
	this->biases_.reset();
	// Set the biases matrix
	this->biases_ = std::move(biases);
}

void nn::Layer::set_biases(const Matrix<float>& biases)
{
// Check if biases is initialized
	if (this->biases_ == nullptr)
	{
		throw std::runtime_error("Biases matrix is not initialized.");
	}
	// Check if the size of the biases matrix is correct
	if (biases.get_rows() != this->neuron_count_ || biases.get_cols() != this->biases_->get_cols())
	{
		throw std::runtime_error("Biases matrix is not the correct size.");
	}

	// Loop through the biases matrix and set the biases
	for (size_t i = 0; i < biases.get_rows() * biases.get_cols(); i++)
	{
		this->biases_->operator[](i) = biases[i];
	}
}

void nn::Layer::change_batch_size(const size_t batch_size)
{
	// Check if the layer is initialized
	if (this->neuron_count_ == 0)
	{
		throw std::runtime_error("Layer is not initialized.");
	}
	// Check if the batch size is the same
	if (batch_size == this->batch_size_)
	{
		return;
	}

	// Change the batch size
	this->batch_size_ = batch_size;

	// Delete the activations matrix and create a new one
	this->activations_.reset();
	this->activations_ = std::make_unique<Matrix<float>>(this->neuron_count_, batch_size);

	// Check if the layer is a hidden layer
	if (this->weights_ == nullptr)
	{
		return;
	}

	// Delete remaining matrices and create new ones
	this->sums_.reset();
	this->sums_ = std::make_unique<Matrix<float>>(this->neuron_count_, batch_size);
	this->delta_activations_.reset();
	this->delta_activations_ = std::make_unique<Matrix<float>>(this->neuron_count_, batch_size);
	this->delta_sums_.reset();
	this->delta_sums_ = std::make_unique<Matrix<float>>(this->neuron_count_, batch_size);
}

size_t nn::Layer::get_neuron_count() const
{
	return this->neuron_count_;
}

size_t nn::Layer::get_batch_size() const
{
	return this->batch_size_;
}

const nn::activation_functions::ActivationFunction* nn::Layer::get_activation_function() const
{
	return this->activation_function_.get();
}

const nn::Matrix<float>& nn::Layer::get_activations() const
{
	// Check if activations is initialized
	if (this->activations_ == nullptr)
	{
		throw std::runtime_error("Activations matrix is not initialized.");
	}

	return *this->activations_;
}

const nn::Matrix<float>& nn::Layer::get_sums() const
{
	if (this->sums_ == nullptr)
	{
		throw std::runtime_error("Sums matrix is not initialized.");
	}

	return *this->sums_;
}

const nn::Matrix<float>& nn::Layer::get_weights() const
{
	if (this->weights_ == nullptr)
	{
		throw std::runtime_error("Weights matrix is not initialized.");
	}

	return *this->weights_;
}

const nn::Matrix<float>& nn::Layer::get_biases() const
{
	if (this->biases_ == nullptr)
	{
		throw std::runtime_error("Biases matrix is not initialized.");
	}

	return *this->biases_;
}

const nn::Matrix<float>& nn::Layer::get_delta_activations() const
{
	if (this->delta_activations_ == nullptr)
	{
		throw std::runtime_error("Delta activations matrix is not initialized.");
	}

	return *this->delta_activations_;
}

const nn::Matrix<float>& nn::Layer::get_delta_sums() const
{
	if (this->delta_sums_ == nullptr)
	{
		throw std::runtime_error("Delta sums matrix is not initialized.");
	}

	return *this->delta_sums_;
}

const nn::Matrix<float>& nn::Layer::get_delta_weights() const
{
	if (this->delta_weights_ == nullptr)
	{
		throw std::runtime_error("Delta weights matrix is not initialized.");
	}

	return *this->delta_weights_;
}

const nn::Matrix<float>& nn::Layer::get_delta_biases() const
{
	if (this->delta_biases_ == nullptr)
	{
		throw std::runtime_error("Delta biases matrix is not initialized.");
	}

	return *this->delta_biases_;
}

void nn::Layer::reset()
{
	// Check if the layer is initialized
	if (this->neuron_count_ == 0)
	{
		return;
	}

	// Resets neuron count and batch size
	this->neuron_count_ = 0;
	this->batch_size_ = 0;

	// Resets the activation matrix
	this->activations_.reset();

	// Check if the layer is a hidden layer
	if (this->weights_ == nullptr)
	{
		return;
	}

	// Resets the weights, biases, sums, and delta matrices
	this->weights_.reset();
	this->biases_.reset();
	this->sums_.reset();
	this->delta_activations_.reset();
	this->delta_weights_.reset();
	this->delta_biases_.reset();
	this->delta_sums_.reset();
}

void nn::Layer::feed_forward(const Layer& previous_layer)
{
	// Check if this layer is initialized and is not the input layer
	if (this->neuron_count_ == 0 || previous_layer.neuron_count_ == 0 || this->weights_ == nullptr)
	{
		throw std::runtime_error("Layer is not initialized.");
	}

	// Calculate the sums
	this->sums_->calculate_sums_for_forward_propagation(*this->weights_, previous_layer.get_activations(), previous_layer.get_activations());
	// Copy the sums to the activations matrix
	this->activations_->operator=(*this->sums_);
	// Apply the activation function to the activations matrix
	this->activation_function_->activate(*this->activations_);
}

void nn::Layer::back_propagate(const Layer& next_layer, const Layer& previous_layer)
{
	// Check if this layer is initialized and is not the input layer
	if (next_layer.weights_ == nullptr || previous_layer.activations_ == nullptr || this->weights_ == nullptr)
	{
		throw std::runtime_error("Layer is not initialized.");
	}

	// Calculate the delta activations
	this->delta_activations_->calculate_delta_activation_for_back_propagation(next_layer.get_weights(), next_layer.get_delta_sums());

	// Calculate the delta sums
	*(this->delta_sums_) = *(this->sums_);
	this->activation_function_->derivative(*this->delta_sums_);
	this->delta_sums_->hadamard_product(*this->delta_activations_);

	// Calculate delta biases
	this->delta_biases_->calculate_delta_biases_for_back_propagation(*this->delta_sums_);

	// Calculate delta weights
	this->delta_weights_->calculate_delta_weights_for_back_propagation(previous_layer.get_activations(), *this->delta_sums_);
}

void nn::Layer::back_propagate(const Matrix<float>& expected_activations)
{
// Check if this layer is initialized and is not the input layer
	if (this->activations_ == nullptr || this->weights_ == nullptr)
	{
		throw std::runtime_error("Layer is not initialized.");
	}

	// Calculate the delta activations
	this->delta_activations_->calculate_delta_activation_from_expected_output(this->get_activations(), expected_activations);

	// Calculate the delta sums
	*(this->delta_sums_) = *(this->sums_);
	this->activation_function_->derivative(*this->delta_sums_);
	this->delta_sums_->hadamard_product(*this->delta_activations_);

	// Calculate delta biases
	this->delta_biases_->calculate_delta_biases_for_back_propagation(*this->delta_sums_);

	// Calculate delta weights
	this->delta_weights_->calculate_delta_weights_for_back_propagation(*this->activations_, *this->delta_sums_);
}

void nn::Layer::update_weights_and_biases(const float learning_rate)
{
	// Check if this layer is initialized and is not the input layer
	if (this->weights_ == nullptr || this->biases_ == nullptr)
	{
		throw std::runtime_error("Layer is not initialized.");
	}

	// Update the weights and biases
	this->weights_->perform_element_wise_operation(*this->delta_weights_,
		[learning_rate](const float weight, const float delta_weight) -> float
		{
			return weight - learning_rate * delta_weight;
		}
	);
	this->biases_->perform_element_wise_operation(*this->delta_biases_,
		[learning_rate](const float bias, const float delta_bias) -> float
		{
			return bias - learning_rate * delta_bias;
		}
	);
}
