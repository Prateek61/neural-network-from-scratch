// File: src/NeuralNetwork/NeuralNetwork.cpp
// Purpose: Implementation file for NeuralNetwork class.

#include "NeuralNetwork/NeuralNetwork.h"

#include <fstream> // std::ofstream

nn::NeuralNetwork::NeuralNetwork()
	: batch_size_(1), learning_rate_(0.01f)
{
}

nn::NeuralNetwork::NeuralNetwork(const float learning_rate, const size_t batch_size)
	: batch_size_(batch_size), learning_rate_(learning_rate)
{
}

void nn::NeuralNetwork::set_learning_rate(const float learning_rate)
{
	learning_rate_ = learning_rate;
}

void nn::NeuralNetwork::set_batch_size(const size_t batch_size)
{
	batch_size_ = batch_size;
}

void nn::NeuralNetwork::set_data_set(std::unique_ptr<DataSet> training_set)
{
	data_set_ = std::move(training_set);
}

void nn::NeuralNetwork::add_layer(std::unique_ptr<Layer> layer)
{
	layers_.push_back(std::move(layer));
}

void nn::NeuralNetwork::feed_forward()
{
	if (!this->is_ready())
	{
		throw std::runtime_error("Neural network is not ready to be fed forward.");
	}

	// first item of the list
	this->layers_.front()->set_activations(this->data_set_->get_batch_input());

	// iterate through the layers except the first one
	for (auto it = std::next(this->layers_.begin()); it != this->layers_.end(); ++it)
	{
		const auto previous_layer = std::prev(it);
		(*it)->feed_forward(*previous_layer->get());
	}
}

void nn::NeuralNetwork::feed_forward_with_input(const Matrix<float>& input)
{
	// Check if the input is valid
	if (input.get_rows() != this->layers_.front()->get_neuron_count() || input.get_cols() != this->batch_size_)
	{
		throw std::runtime_error("Invalid input size.");
	}

	// first item of the list
	this->layers_.front()->set_activations(input);

	// iterate through the layers except the first one
	for (auto it = std::next(this->layers_.begin()); it != this->layers_.end(); ++it)
	{
		const auto previous_layer = std::prev(it);
		(*it)->feed_forward(*previous_layer->get());
	}
}

void nn::NeuralNetwork::back_propagate()
{
	if (!this->is_ready())
	{
		throw std::runtime_error("Neural network is not ready to be back propagated.");
	}

	// last item of the list
	auto last_layer = std::prev(this->layers_.end());
	auto second_to_last_layer = std::prev(last_layer);
	(*last_layer)->back_propagate(this->data_set_->get_batch_output(), *second_to_last_layer->get());

	// iterate through the second to the last layer to the second layer
	for (auto it = std::prev(this->layers_.end(), 2); it != this->layers_.begin(); --it)
	{
		const auto next_layer = std::next(it);
		const auto previous_layer = std::prev(it);
		(*it)->back_propagate(*next_layer->get(), *previous_layer->get());
	}
}

void nn::NeuralNetwork::update_weights_and_biases()
{
	if (!this->is_ready())
	{
		throw std::runtime_error("Neural network is not ready to update weights and biases.");
	}

	// iterate through the layers except the first one
	for (auto it = std::next(this->layers_.begin()); it != this->layers_.end(); ++it)
	{
		(*it)->update_weights_and_biases(this->learning_rate_);
	}
}

void nn::NeuralNetwork::train(const size_t epochs)
{
	for (size_t i = 0; i < epochs; ++i)
	{
		train_one_epoch();
	}
}

void nn::NeuralNetwork::train_one_epoch()
{
	this->data_set_->reset();
	while (!this->data_set_->is_end())
	{
		this->feed_forward();
		this->back_propagate();
		this->update_weights_and_biases();

		this->data_set_->go_to_next_batch();
	}
	this->data_set_->reset();
}

float nn::NeuralNetwork::calculate_accuracy()
{
	this->data_set_->reset();

	size_t correct = 0;

	while (!this->data_set_->is_end())
	{
		this->feed_forward();

		auto activation_matrix = this->get_output();
		auto expected_matrix = this->data_set_->get_batch_output();

		for (size_t i = 0; i < activation_matrix.get_cols(); ++i)
		{
			size_t max_index = 0;
			float max_value = activation_matrix(0, i);
			for (size_t j = 1; j < activation_matrix.get_rows(); ++j)
			{
				if (activation_matrix(j, i) > max_value)
				{
					max_index = j;
					max_value = activation_matrix(j, i);
				}
			}

			if (expected_matrix(max_index, i) == 1.0f)
			{
				++correct;
			}
		}

		this->data_set_->go_to_next_batch();
	}

	const auto accuracy = static_cast<float>(correct) / static_cast<float>(this->data_set_->get_total_size());
	return accuracy;
}

float nn::NeuralNetwork::get_loss()
{
	float loss = 0.0f;

	this->data_set_->reset();

	while (!this->data_set_->is_end())
	{
		this->feed_forward();

		auto activation_matrix = this->get_output();
		auto expected_matrix = this->data_set_->get_batch_output();

		for (size_t i = 0; i < activation_matrix.get_rows() * activation_matrix.get_cols(); ++i)
		{
			const auto temp = static_cast<float>(pow(activation_matrix[i] - expected_matrix[i], 2));
			loss += temp * temp;
		}
		data_set_->go_to_next_batch();
	}

	this->data_set_->reset();

	loss /= static_cast<float>(this->data_set_->get_total_size());
	return loss;
}

void nn::NeuralNetwork::save_to_file(const std::string& file_name) const
{
	// Open file
	std::ofstream file(file_name);
	if (!file.is_open())
	{
		throw std::runtime_error("Could not open file.");
	}

	// Write general information
	file << this->learning_rate_ << "\n";
	// Write number of layers
	file << this->layers_.size() << "\n";

	// Write information for each layer
	for (const auto& layer : this->layers_)
	{
		// Write number of neurons
		file << layer->get_neuron_count() << "\n";

		// Check if its not the first layer
		if (layer != layers_.front())
		{
			// Write dimensions of the weights matrix
			file << layer->get_weights().get_rows() << " " << layer->get_weights().get_cols() << "\n";
			// Write the weights matrix
			file << layer->get_weights();
			// Write dimensions of the biases matrix
			file << layer->get_biases().get_rows() << " " << layer->get_biases().get_cols() << "\n";
			// Write the biases matrix
			file << layer->get_biases();
		}
	}
}

void nn::NeuralNetwork::load_from_file(const std::string& file_name)
{
	// Open file
	std::ifstream file(file_name);
	if (!file.is_open())
	{
		throw std::runtime_error("Could not open file.");
	}

	size_t num_layers = 0;
	float learning_rate = 0.0f;

	// Read general information
	file >> learning_rate;
	file >> num_layers;
	// Set the learning rate
	this->learning_rate_ = learning_rate;

	// Read information for each layer
	for (size_t i = 0; i < num_layers; ++i)
	{
		size_t num_neurons = 0;
		file >> num_neurons;

		// First layer
		if (i == 0)
		{
			auto layer = std::make_unique<Layer>(num_neurons, 1);
			this->add_layer(std::move(layer));
		}
		// Other layers
		else
		{
			// Weights matrix
			size_t w_cols, w_rows;
			file >> w_rows >> w_cols;
			Matrix<float> weights(w_rows, w_cols);
			float temp;
			// Read the weights matrix
			for (size_t j = 0; j < w_rows * w_cols; ++j)
			{
				file >> temp;
				weights[j] = temp;
			}

			// Biases matrix
			size_t b_cols, b_rows;
			file >> b_rows >> b_cols;
			Matrix<float> biases(b_rows, b_cols);
			// Read the biases matrix
			for (size_t j = 0; j < b_rows * b_cols; ++j)
			{
				file >> biases[j];
			}

			// Add the layer
			auto layer = std::make_unique<Layer>(num_neurons, 1, w_cols);
			// Set the weights and biases
			layer->set_weights(weights);
			layer->set_biases(biases);

			// Add the layer
			this->add_layer(std::move(layer));
		}
	}
}

bool nn::NeuralNetwork::is_ready() const
{
	// Check if the data set is ready
	if (!this->data_set_ || !this->data_set_->is_ready())
	{
		return false;
	}
	// Check if the layers are ready
	if (this->layers_.size() < 2)
	{
		return false;
	}
	// Check if the layers are of valid size
	if (this->layers_.front()->get_neuron_count() != this->data_set_->get_input_size() || this->layers_.back()->
		get_neuron_count() != this->data_set_->get_output_size())
	{
		return false;
	}

	return true;
}

const nn::Matrix<float>& nn::NeuralNetwork::get_output() const
{
	return this->layers_.back()->get_activations();
}

nn::DataSet* nn::NeuralNetwork::get_data_set()
{
	return this->data_set_.get();
}

std::list<std::unique_ptr<nn::Layer>>& nn::NeuralNetwork::get_layers()
{
	return this->layers_;
}
