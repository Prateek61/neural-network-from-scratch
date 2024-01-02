// File: example/main.cpp
// Purpose: Example file for NeuralNetwork Library.

#define OLC_PGE_APPLICATION
#include "GUI.h"

int main()
{
	GUI gui;
	if (gui.Construct(540, 360, 2, 2))
		gui.Start();

	return 0;
}
