// File: example/main.cpp
// Purpose: Example file for NeuralNetwork Library.

#include <chrono>

#include "TrainSet.h"

#include <NeuralNetwork/NeuralNetwork.h>
#include <NeuralNetwork/ActivationFunction.h>

int main()
{
	constexpr int batch_size = 1;
    constexpr int num_epochs = 10;
    constexpr float learning_rate = 1.0f;
    constexpr int print_every = 1;

	nn::NeuralNetwork nn(0.01f, batch_size);
	auto train_set = std::make_unique<TrainSet>("dataset/train-images.idx3-ubyte", "dataset/train-labels.idx1-ubyte");
	train_set->initialize(batch_size);
	nn.set_data_set(std::move(train_set));
	auto layer1 = std::make_unique<nn::Layer>(784, batch_size);
	auto layer2 = std::make_unique<nn::Layer>(16, batch_size, 784);
	layer2->set_activation_function(std::make_unique<nn::activation_functions::Sigmoid>());
	auto layer3 = std::make_unique<nn::Layer>(16, batch_size, 16);
	auto layer4 = std::make_unique<nn::Layer>(10, batch_size, 16);

	std::cout << "Initialized\n";

	nn.add_layer(std::move(layer1));
	nn.add_layer(std::move(layer2));
	nn.add_layer(std::move(layer3));
	nn.add_layer(std::move(layer4));

	std::cout << "Network is setup\n";

	// create a clock
	const auto start = std::chrono::high_resolution_clock::now();

    auto train_start = std::chrono::high_resolution_clock::now();

	if (nn.is_ready())
	{
		std::cout << "Neural network is ready.\n";

		std::cout << "Initial Loss: " << nn.get_loss() << '\n';
		for (int i = 0; i < num_epochs; ++i)
        {
            train_start = std::chrono::high_resolution_clock::now();
            nn.train_one_epoch();
            const auto train_end = std::chrono::high_resolution_clock::now();
            if (i % print_every == 0 || i == num_epochs - 1)
            {
                std::cout << "Epoch: " << i << "\tTime: " << std::chrono::duration_cast<std::chrono::milliseconds>(train_end - train_start).count() << "ms\n";
                std::cout << "Loss: " << nn.get_loss() << '\n';
            }
        }
	}
	else
	{
		std::cout << "Neural network is not ready.\n";
	}

	const auto end = std::chrono::high_resolution_clock::now();
	std::cout << "Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms\n";

    std::cout << "\nTesting...\n";
    auto test_set = std::make_unique<TrainSet>("dataset/t10k-images.idx3-ubyte", "dataset/t10k-labels.idx1-ubyte");
    test_set->initialize(batch_size);
    nn.set_data_set(std::move(test_set));
    std::cout << "Loss: " << nn.get_loss() << '\n';
    std::cout << "Accuracy: " << nn.calculate_accuracy() << '\n';

	return 0;
}