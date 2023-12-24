// File: include/NeuralNetwork/Matrix.h
// Purpose: Header file for NeuralNetwork class.

#pragma once

#include <memory> // std::unique_ptr
#include <list> // std::list
#include <string> // std::string

#include "NeuralNetwork/Layer.h" // nn::Layer
#include "NeuralNetwork/DataSet.h" // nn::TrainingSet


namespace nn
{
	class NeuralNetwork
	{
	private:
		/// <summary>
		/// The layers of the neural network. (owned)
		/// </summary>
		std::list<std::unique_ptr<nn::Layer>> layers_;

		/// <summary>
		/// The training set of the neural network. (owned)
		/// </summary>
		std::unique_ptr<nn::DataSet> data_set_;

		/// <summary>
		/// Training batch size.
		/// </summary>
		size_t batch_size_;

		/// <summary>
		/// Learning rate of the neural network.
		/// </summary>
		float learning_rate_;

	public:
		/// <summary>
		/// Default constructor.
		/// </summary>
		NeuralNetwork();

		/// <summary>
		/// Delete the copy constructor.
		/// </summary>
		NeuralNetwork(const NeuralNetwork&) = delete;

		/// <summary>
		/// Constructor that sets the learning rate and batch size.
		/// </summary>
		/// <param name="learning_rate">Learning rate of the network</param>
		/// <param name="batch_size">Batch size of the network</param>
		NeuralNetwork(const float learning_rate, const size_t batch_size);

		/// <summary>
		/// Sets the learning rate of the neural network.
		/// </summary>
		void set_learning_rate(const float learning_rate);

		/// <summary>
		/// Sets the batch size of the neural network.
		/// </summary>
		/// <param name="batch_size"></param>
		void set_batch_size(const size_t batch_size);

		/// <summary>
		/// Sets the data set of the neural network. (Takes ownership)
		/// </summary>
		void set_data_set(std::unique_ptr<nn::DataSet> training_set);

		/// <summary>
		/// Add a layer to the neural network. (Takes ownership)
		/// </summary>
		void add_layer(std::unique_ptr<nn::Layer> layer);

		/// <summary>
		/// Runs the forward propagation algorithm on the neural network.
		/// </summary>
		void feed_forward();

		/// <summary>
		/// Runs the back propagation algorithm on the neural network.
		/// </summary>
		void back_propagate();

		/// <summary>
		/// Updates the weights and biases of the neural network.
		/// </summary>
		void update_weights_and_biases();

		/// <summary>
		/// Trains the neural network for the given number of epochs.
		/// </summary>
		/// <param name="epochs">The number of times the network trains the whole data set</param>
		void train(const size_t epochs);

		/// <summary>
		/// Saves the neural network to a file.
		/// </summary>
		/// <param name="file_name">Name of the file (relative to executable)</param>
		void save_to_file(const std::string& file_name) const;

		/// <summary>
		/// Loads the neural network from a file.
		/// </summary>
		/// <param name="file_name">Name of the file (relative to executable)</param>
		void load_from_file(const std::string& file_name);

		/// <summary>
		/// Returns if the neural network is ready to be trained or used.
		/// </summary>
		[[nodiscard]] bool is_ready() const;

		/// <summary>
		/// Returns the output of the neural network.
		/// </summary>
		/// <returns>Output matrix</returns>
		[[nodiscard]] const nn::Matrix<float>& get_output() const;

		/// <summary>
		/// Gets the data set of the neural network.
		/// </summary>
		/// <returns>DataSet Class the network is using</returns>
		nn::DataSet& get_training_set();

		/// <summary>
		/// Returns the layers of the neural network.
		/// </summary>
		/// <returns>A std::list of Layer pointers</returns>
		std::list<std::unique_ptr<nn::Layer>>& get_layers();
	};
}