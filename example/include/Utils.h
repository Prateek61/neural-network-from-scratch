#pragma once

#include <chrono>
#include <fstream>
#include <vector>
#include <ostream>

#include "TrainSet.h"
#include "NeuralNetwork/NeuralNetwork.h"

inline void setup_network(const std::vector<int>& structure, nn::NeuralNetwork& net, const size_t batch_size)
{
	for (size_t i = 0; i < structure.size(); ++i)
	{
		if (i == 0)
		{
			auto layer = std::make_unique<nn::Layer>(structure[i], batch_size);
			net.add_layer(std::move(layer));
		}
		else if (i == structure.size() - 1)
		{
			auto layer = std::make_unique<nn::Layer>(structure[i], batch_size, structure[i - 1]);
			layer->set_activation_function(std::make_unique<nn::activation_functions::Sigmoid>());
			net.add_layer(std::move(layer));
		}
		else
		{
			auto layer = std::make_unique<nn::Layer>(structure[i], batch_size, structure[i - 1]);
			layer->set_activation_function(std::make_unique<nn::activation_functions::Sigmoid>());
			net.add_layer(std::move(layer));
		}
	}
}

inline void train_and_test(const int batch_size, const int num_epochs, const float learning_rate, const int print_every,
                           const std::vector<int>& structure, const bool print, const bool print_to_file,
                           const std::string& file_name)
{
	// Create a clock
	const auto start = std::chrono::high_resolution_clock::now();

	// Setup Variables
	float train_set_loss = 0.0f;
	float train_set_accuracy = 0.0f;
	float test_set_loss = 0.0f;
	float test_set_accuracy = 0.0f;
	long long time_taken_sec = 0;

	nn::NeuralNetwork nn(learning_rate, batch_size);

	// Setup TrainSet
	auto train_set = std::make_unique<TrainSet>("dataset/train-images.idx3-ubyte", "dataset/train-labels.idx1-ubyte");
	train_set->initialize(batch_size);
	nn.set_data_set(std::move(train_set));

	// Setup NeuralNetwork
	setup_network(structure, nn, batch_size);
	if (print)
		std::cout << "Network is setup\n";

	if (nn.is_ready())
	{
		if (print)
		{
			std::cout << "Neural network is ready.\n";
			std::cout << "Initial Loss: " << nn.get_loss() << '\n';
		}

		for (int i = 0; i < num_epochs; ++i)
		{
			auto train_start = std::chrono::high_resolution_clock::now();

			nn.train_one_epoch();

			auto train_end = std::chrono::high_resolution_clock::now();
			std::chrono::duration<double> train_time = train_end - train_start;

			if (print && (i % print_every == 0))
			{
				std::cout << "Epoch: " << i << '\n';
				std::cout << "Loss: " << nn.get_loss() << '\n';
				std::cout << "Time: " << train_time.count() << "s\n";
			}

			if (i == num_epochs - 1)
			{
				train_set_loss = nn.get_loss();
				train_set_accuracy = nn.calculate_accuracy();
				if (print)
				{
					std::cout << "Epoch: " << i << '\n';
					std::cout << "Loss: " << train_set_loss << '\n';
					std::cout << "Time: " << train_time.count() << "s\n";
				}
			}
		}
	}

	// Setup TestSet
	if (print)
		std::cout << "\nTesting...\n";
	auto test_set = std::make_unique<TrainSet>("dataset/t10k-images.idx3-ubyte", "dataset/t10k-labels.idx1-ubyte");
	test_set->initialize(batch_size);
	nn.set_data_set(std::move(test_set));

	test_set_loss = nn.get_loss();
	test_set_accuracy = nn.calculate_accuracy();
	if (print)
	{
		std::cout << "Loss: " << test_set_loss << '\n';
		std::cout << "Accuracy: " << test_set_accuracy << '\n';
	}

	const auto end = std::chrono::high_resolution_clock::now();
	time_taken_sec = std::chrono::duration_cast<std::chrono::seconds>(end - start).count();
	if (print)
		std::cout << "Time: " << time_taken_sec << "s\n";

	if (print_to_file)
	{
		// Format: Structure, Batch Size, Learning Rate, Num Epochs, Train Set Loss, Train Set Accuracy, Test Set Loss, Test Set Accuracy, Time Taken (sec)
		std::ofstream file(file_name, std::ios::app);

		file << "\"";
		for (size_t i = 0; i < structure.size(); ++i)
		{
			file << structure[i];
			if (i != structure.size() - 1)
				file << ',';
		}
		file << "\",";
		file << batch_size << ',' << num_epochs << ',' << learning_rate << ',' << train_set_loss << ',' << train_set_accuracy << ',' << test_set_loss << ',' << test_set_accuracy << ',' << time_taken_sec << '\n';

		file.close();
	}
}

inline int get_completed()
{
	std::ifstream file("completed.txt");
	int completed = 0;
	if (file.is_open())
	{
		file >> completed;
		file.close();
	}
	else
	{
		completed = 0;
		// Create the file
		std::ofstream file_new("completed.txt");
		file_new << completed;
		file_new.close();
	}

	return completed;
}

inline void update_completed()
{
	int completed = get_completed();
	completed++;

	std::ofstream file("completed.txt");
	file << completed;
	file.close();
}

inline void trainer()
{
	std::vector<int> batch_sizes = { 8, 32, 50, 100, 250};
	std::vector<int> num_epochs = {30, 250, 100, 50, 10};
	std::vector<float> learning_rates = { 0.5f, 0.25f, 0.1f, 0.05f, 0.01f, 0.001f };
	std::vector<std::vector<int>> structures = {
		{784, 64, 64, 10},
		{784, 128, 64, 10},
		{784, 64, 64, 64, 10},
		{784, 64, 32, 10},
		{784, 128, 128, 10},
		{784, 128, 128, 64, 10}
	};

	int completed = get_completed();

	for (const auto& structure : structures)
	{
		for (const auto& batch_size : batch_sizes)
		{
			for (const auto& epoch : num_epochs)
			{
				for (const auto& learning_rate : learning_rates)
				{
					if (completed > 0)
					{
						completed--;
						continue;
					}
					train_and_test(batch_size, epoch, learning_rate, 5, structure, false, true, "results.csv");
					update_completed();
				}
			}
		}
	}
}
