// File: example/main.cpp
// Purpose: Example file for NeuralNetwork Library.

#include <chrono>

#include "TrainSet.h"

#include <NeuralNetwork/NeuralNetwork.h>
#include <NeuralNetwork/ActivationFunction.h>

int main()
{
	constexpr int batch_size = 20;

	nn::NeuralNetwork nn(0.01f, batch_size);
	auto train_set = std::make_unique<TrainSet>("dataset/train-images.idx3-ubyte", "dataset/train-labels.idx1-ubyte");
	train_set->initialize(batch_size);
	nn.set_data_set(std::move(train_set));
	auto layer1 = std::make_unique<nn::Layer>(784, batch_size);
	auto layer2 = std::make_unique<nn::Layer>(16, batch_size, 784);
	layer2->set_activation_function(std::make_unique<nn::activation_functions::Sigmoid>());
	auto layer3 = std::make_unique<nn::Layer>(16, batch_size, 16);
	auto layer4 = std::make_unique<nn::Layer>(10, batch_size, 784);

	std::cout << "Initialized\n";

	nn.add_layer(std::move(layer1));
	// nn.add_layer(std::move(layer2));
	// nn.add_layer(std::move(layer3));
	nn.add_layer(std::move(layer4));

	nn.save_to_file("nn.txt");

	// create a clock
	const auto start = std::chrono::high_resolution_clock::now();


	if (nn.is_ready())
	{
		std::cout << "Neural network is ready.\n";

		std::cout << "Loss: " << nn.get_loss() << '\n';
		std::cout << "Accuracy: " << nn.calculate_accuracy() << '\n';
		for (size_t i = 0; i < 2; i++)
		{
			nn.train_one_epoch();
		}

		std::cout << "Loss: " << nn.get_loss() << '\n';
		std::cout << "Accuracy: " << nn.calculate_accuracy() << '\n';
	}
	else
	{
		std::cout << "Neural network is not ready.\n";
	}

	nn.save_to_file("gg.txt");

	const auto end = std::chrono::high_resolution_clock::now();
	std::cout << "Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms\n";

	return 0;
}