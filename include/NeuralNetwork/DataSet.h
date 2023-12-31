// File: include/NeuralNetwork/Dataset.h
// Purpose: Header file for Dataset virtual class.

#pragma once

#include "NeuralNetwork/Matrix.h" // nn::Matrix

namespace nn
{
	class DataSet
	{
	protected:
		/// <summary>
		/// The current index of the training set.
		/// </summary>
		size_t current_index_;

	public:
		/// <summary>
		/// Default constructor. (initializes the current index to 0)
		/// </summary>
		DataSet();

		/// <summary>
		/// Virtual destructor.
		/// </summary>
		virtual ~DataSet() = default;

		/// <summary>
		/// Initializes the training set. (e.g. load data from file)
		/// </summary>
		virtual void initialize(const size_t batch_size) = 0;

		/// <summary>
		/// Gets the next batch of input of training data.
		/// </summary>
		/// <returns>Training data</returns>
		virtual nn::Matrix<float>& get_batch_input() = 0;

		/// <summary>
		/// Gets the next batch of output of training data.
		/// </summary>
		/// <returns>Output data</returns>
		virtual nn::Matrix<float>& get_batch_output() = 0;

		/// <summary>
		/// Indicates whether the training set has reached the end.
		/// </summary>
		[[nodiscard]] virtual bool is_end() const = 0;

		/// <summary>
		/// Indicates whether the training set is ready to be used.
		/// </summary>
		[[nodiscard]] virtual bool is_ready() const = 0;

		/// <summary>
		/// Resets the training set to the beginning.
		/// </summary>
		virtual void reset() = 0;

		/// <summary>
		/// Returns the input size of the training set.
		/// </summary>
		[[nodiscard]] virtual size_t get_input_size() const = 0;

		/// <summary>
		/// Returns the output size of the training set.
		/// </summary>
		[[nodiscard]] virtual size_t get_output_size() const = 0;

		/// <summary>
		/// Returns the total size of the training set.
		/// </summary>
		[[nodiscard]] virtual size_t get_total_size() const = 0;

		/// <summary>
		/// Gets the current index of the training set
		/// </summary>
		[[nodiscard]] size_t get_current_index() const;

		/// <summary>
		/// Increments the current index of the training set
		/// </summary>
		void go_to_next_batch();
	};
}