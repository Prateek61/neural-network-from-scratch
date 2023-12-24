// File: example/main.cpp
// Purpose: Example file for NeuralNetwork Library.

#include <iostream>

#include <NeuralNetwork/NeuralNetwork.h>

int main()
{	
	auto first_layer = std::make_unique<nn::Layer>(784, 1);
	auto second_layer = std::make_unique<nn::Layer>(64, 1, 768);
	second_layer->set_activation_function(std::make_unique<nn::activation_functions::ReLU>());

	auto third_layer = std::make_unique<nn::Layer>(10, 1, 64);

	nn::NeuralNetwork nn(0.01f, 1);
	nn.add_layer(std::move(first_layer));
	nn.add_layer(std::move(second_layer));
	nn.add_layer(std::move(third_layer));

	nn.save_to_file("network.txt");

    std::cout << "Hello World!\n";

	return 0;
}