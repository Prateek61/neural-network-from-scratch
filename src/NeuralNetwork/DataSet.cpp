// File: src/NeuralNetwork/DataSet.cpp
// Purpose: Implementation file for DataSet class.

#include "NeuralNetwork/DataSet.h"

nn::DataSet::DataSet()
	: current_index_(0)
{
}

size_t nn::DataSet::get_current_index() const
{
	return current_index_;
}