#include "NeuralNetwork.h"

#include <fstream> // std::ofstream

nn::NeuralNetwork::NeuralNetwork()
	: batch_size_(1), learning_rate_(0.01f)
{}

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

void nn::NeuralNetwork::set_data_set(std::unique_ptr<nn::DataSet> training_set)
{
	data_set_ = std::move(training_set);
}

void nn::NeuralNetwork::add_layer(std::unique_ptr<nn::Layer> layer)
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
	this->layers_.front()->set_activations(this->data_set_->get_batch_input(this->batch_size_));

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
	this->layers_.back()->back_propagate(this->data_set_->get_batch_output(this->batch_size_));

	// iterate through the second to the last layer to the second layer
	for (auto it = std::prev(this->layers_.end(), 2); it != std::next(this->layers_.begin()); --it)
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
		this->data_set_->reset();
		while (!this->data_set_->is_end())
		{
			this->feed_forward();
			this->back_propagate();
			this->update_weights_and_biases();
		}
	}
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
		// Write the name of activation function class with RTTI
		file << typeid(*layer->get_activation_function()).name() << "\n";

		// Check if the layer is the first layer
		if (layer == this->layers_.front())
		{
			// Write the input size
			file << layer->get_neuron_count() << "\n";
		}
		else // Not the first layer
		{
			// Write dimensions of the weights matrix
			file << layer->get_weights().get_rows() << " " << layer->get_weights().get_cols() << "\n";
			// Write the weights matrix
			file << layer->get_weights() << "\n";
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
	if (this->layers_.front()->get_neuron_count() != this->data_set_->get_input_size() || this->layers_.back()->get_neuron_count() != this->data_set_->get_output_size())
	{
		return false;
	}

	return true;
}

const nn::Matrix<float>& nn::NeuralNetwork::get_output() const
{
	return this->layers_.back()->get_activations();
}

nn::DataSet& nn::NeuralNetwork::get_training_set()
{
	return *this->data_set_;
}

std::list<std::unique_ptr<nn::Layer>>& nn::NeuralNetwork::get_layers()
{
	return this->layers_;
}
