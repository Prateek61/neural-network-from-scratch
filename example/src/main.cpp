// File: example/main.cpp
// Purpose: Example file for NeuralNetwork Library.

#ifndef OLC_PGE_APPLICATION
#define OLC_PGE_APPLICATION
#endif


#include <chrono>

#include "GUI.h"
#include "Utils.h"

int main()
{
	// train_and_test(8, 30, 0.01f, 1, { 784, 128, 64, 10 }, true, false, "", "neural_network.txt");

	// return 0;
	GUI gui;
	if (gui.Construct(540, 360, 2, 2))
	{
		gui.Start();
	}

	return 0;
}
