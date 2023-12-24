// File: include/Matrix.h
// Purpose: Header file for Matrix.cpp.

#pragma once

#include <random> // std::random_device, std::mt19937, std::uniform_real_distribution
#include <functional> // std::function
#include <stdexcept> // std::runtime_error
#include <vector>	 // std::vector
#include <iostream> // std::ostream

#include "AlignedMemoryAllocator.h" // nn::utils::AlignedMemoryAllocator



namespace nn
{
	/// <summary>
	/// Class for a matrix.
	/// </summary>
	/// <typeparam name="T">Type of data in matrix.</typeparam>
	template <typename T>
	class Matrix
	{
	private:
		/// <summary>
		/// Rows in the matrix.
		/// </summary>
		size_t rows_;

		/// <summary>
		/// Columns in the matrix.
		/// </summary>
		size_t cols_;

		/// <summary>
		/// Aligned memory allocator to allocate memory for elements in the matrix.
		/// </summary>
		utils::AlignedMemoryAllocator<T, 64> allocator_;

	public:
		/// <summary>
		/// Default constructor.
		/// </summary>
		Matrix();

		/// <summary>
		/// Constructor for a matrix of size rows x cols.
		/// </summary>
		/// <param name="rows">Rows in the matrix</param>
		/// <param name="cols">Columns in the matrix</param>
		Matrix(size_t rows, size_t cols);

		/// <summary>
		/// Constructor for a matrix of size rows x cols.
		/// </summary>
		/// <param name="data">2d vector representing the matrix</param>
		explicit Matrix(const std::vector<std::vector<T>>& data);

		/// <summary>
		/// Constructor for a matrix of size rows x cols.
		/// </summary>
		/// <param name="data">1d vector representing the matrix</param>
		/// <param name="rows">Number of rows in this matrix</param>
		/// <param name="cols">Number of columns in this matrix</param>
		Matrix(const std::vector<T>& data, size_t rows, size_t cols);

		/// <summary>
		/// Copy constructor.
		/// </summary>
		Matrix(const Matrix<T>& other);

		/// <summary>
		/// Assignment operator. Copies the data from other matrix to this matrix.
		/// </summary>
		Matrix& operator=(const Matrix<T>& other);

		/// <summary>
		/// Returns the element at row, col.
		/// </summary>
		/// <returns>Reference to the element at row, col</returns>
		[[nodiscard]] T& operator()(size_t row, size_t col);

		/// <summary>
		/// Returns the element at row, col.
		/// </summary>
		/// <returns>Copy of element at row, col</returns>
		[[nodiscard]] const T& operator()(size_t row, size_t col) const;

		/// <summary>
		/// Returns the element at index of the matrix data array.
		/// </summary>
		/// <returns>Reference to the element at index</returns>
		[[nodiscard]] T& operator[](size_t index);

		/// <summary>
		/// Returns the element at index of the matrix data array.
		/// </summary>
		/// <returns>Copy of the element at index</returns>
		[[nodiscard]] const T& operator[](size_t index) const;

		/// <summary>
		/// Returns the element at row, col.
		/// </summary>
		/// <returns>Copy of the element at row, col</returns>
		[[nodiscard]] T at(size_t row, size_t col) const;

		/// <summary>
		/// Returns the element at index of the matrix data array.
		/// </summary>
		/// <returns>Copy of the element at index</returns>
		[[nodiscard]] T at(size_t index) const;

		/// <summary>
		/// Returns the matrix data.
		/// </summary>
		/// <returns>Array to type T</returns>
		[[nodiscard]] T* get_data();

		/// <summary>
		/// Returns the matrix data.
		/// </summary>
		/// <returns>Array to the type T</returns>
		[[nodiscard]] const T* get_data() const;

		/// <summary>
		/// Returns the number of rows in the matrix.
		/// </summary>
		[[nodiscard]] size_t get_rows() const;

		/// <summary>
		/// Returns the number of columns in the matrix.
		/// </summary>
		[[nodiscard]] size_t get_cols() const;

		/// <summary>
		/// Clears the matrix.
		/// </summary>
		void clear();

		/// <summary>
		/// Initializes the matrix with size rows x cols.
		/// </summary>
		/// <param name="rows">Rows in the matrix</param>
		/// <param name="cols">Columns in the matrix</param>
		void init(size_t rows, size_t cols);

		/// <summary>
		/// Performs matrix multiplication on matrix1 and matrix2 and stores the result in result.
		/// </summary>
		/// <param name="matrix1">First Matrix</param>
		/// <param name="matrix2">Second Matrix</param>
		/// <param name="result">Result matrix</param>
		static void multiply(const Matrix<T>& matrix1, const Matrix<T>& matrix2, Matrix<T>& result);

		/// <summary>
		/// Performs matrix multiplication on matrix1 and matrix2 and stores the result in this matrix.
		/// </summary>
		/// <param name="matrix1">First Matrix</param>
		/// <param name="matrix2">Second Matrix</param>
		void multiply(const Matrix<T>& matrix1, const Matrix<T>& matrix2);

		/// <summary>
		/// Perform element wise multiplication on this matrix with other matrix and stores the result in this matrix.
		/// </summary>
		/// <param name="other">Other matrix</param>
		void hadamard_product(const Matrix<T>& other);

		/// <summary>
		/// Performs an element wise operation on this matrix with other matrix and stores the result in this matrix.
		/// </summary>
		/// <param name="other">Other matrix</param>
		/// <param name="operation">Function (First argument is value of this and Second is value of other)</param>
		void perform_element_wise_operation(const Matrix<T>& other, const std::function<T(T, T)>& operation);

		/// <summary>
		/// Performs an element wise operation on this matrix.
		/// </summary>
		/// <param name="operation">Function</param>
		void perform_element_wise_operation(const std::function<T(T)>& operation);

		/// <summary>
		/// Randomizes the contents of this matrix between min and max.
		/// </summary>
		void randomize(const T& min, const T& max);

		/// <summary>
		/// Transposes the matrix and returns the result.
		/// </summary>
		/// <returns>Transpose matrix of this matrix</returns>
		Matrix<T> transpose() const;

		/// <summary>
		/// Calculates the sums matrix for forward propagation and stores the result in this matrix(for layer class).
		///	sum = weights * input + biases
		/// </summary>
		/// <param name="weights">Weights of this layer</param>
		/// <param name="biases">Biases of this Layer</param>
		/// <param name="input">Activations of previous Layer</param>
		void calculate_sums_for_forward_propagation(const Matrix<T>& weights, const Matrix<T>& biases,
		                                            const Matrix<T>& input);

		/// <summary>
		/// Calculates the delta activation matrix and stores the result in this matrix(for layer class).
		///	delta_activation = transpose(weights) * delta_sums
		/// </summary>
		/// <param name="next_layer_weights">Weights matrix of next layer</param>
		/// <param name="next_layer_delta_sums">Delta sums matrix of next layer</param>
		void calculate_delta_activation_for_back_propagation(const Matrix<T>& next_layer_weights,
		                                                     const Matrix<T>& next_layer_delta_sums);

		/// <summary>
		/// Calculates the delta biases matrix and stores the result in this matrix(for layer class).
		///	delta_biases = mean of delta_sums so it becomes a vector(1d matrix)
		/// </summary>
		/// <param name="this_layer_delta_sums"></param>
		void calculate_delta_biases_for_back_propagation(const Matrix<T>& this_layer_delta_sums);

		/// <summary>
		/// Calculates the delta weights matrix and stores the result in this matrix(for layer class).
		///	delta_weights = (delta_sums * transpose(previous_layer_activation)) / batch_size
		/// </summary>
		/// <param name="previous_layer_activations">Activation Matrix of previous layer</param>
		/// <param name="this_layer_delta_sums">Delta sums matrix of this layer</param>
		void calculate_delta_weights_for_back_propagation(const Matrix<T>& previous_layer_activations,
		                                                  const Matrix<T>& this_layer_delta_sums);

		/// <summary>
		/// Calculates the delta activation matrix from the expected output matrix and stores the result in this matrix(for layer class).
		/// </summary>
		/// <param name="this_layer_activations">Activation matrix of this layer</param>
		/// <param name="expected_output">Expected Output of the training session</param>
		void calculate_delta_activation_from_expected_output(const Matrix<T>& this_layer_activations, const Matrix<T>& expected_output);
	};
}

#pragma region Implementation
template <typename T>
nn::Matrix<T>::Matrix()
	: rows_(0), cols_(0)
{
}


template <typename T>
nn::Matrix<T>::Matrix(const size_t rows, const size_t cols)
	: rows_(0), cols_(0)
{
	this->init(rows, cols);
}

template <typename T>
nn::Matrix<T>::Matrix(const std::vector<std::vector<T>>& data)
	: rows_(0), cols_(0)
{
	// Initializes the matrix with the size of the data.
	this->init(data.size(), data[0].size());

	// Copy data from 2d vector to matrix.
	for (size_t i = 0; i < data.size(); i++)
	{
		for (size_t j = 0; j < data[0].size(); j++)
		{
			this->operator()(i, j) = data[i][j];
		}
	}
}

template <typename T>
nn::Matrix<T>::Matrix(const std::vector<T>& data, const size_t rows, const size_t cols)
	: rows_(0), cols_(0)
{
	// Initializes the matrix with the size of the data.
	this->init(rows, cols);

	// Copy data from 1d vector to matrix.
	for (size_t i = 0; i < data.size(); i++)
	{
		this->operator[](i) = data[i];
	}
}

template <typename T>
nn::Matrix<T>::Matrix(const Matrix<T>& other)
	: rows_(0), cols_(0)
{
	this->init(other.get_rows(), other.get_cols());

	// Copy data from other matrix.
	this->allocator_.copy_data(other.allocator_);
}

template<typename T>
inline nn::Matrix<T>& nn::Matrix<T>::operator=(const Matrix<T>& other)
{
	// Check if the matrix is initialized.
	if (this->get_rows() == 0 || this->get_cols() == 0 || !this->allocator_.is_initialized())
	{
		throw std::runtime_error("Cannot copy to an uninitialized matrix.");
	}
	// Check if the dimensions are compatible
    if (this->get_rows() != other.get_rows() || this->get_cols() != other.get_cols())
	{
		throw std::runtime_error("Cannot copy matrices with incompatible dimensions.");
	}

	this->allocator_.copy_data(other.allocator_);
	return *this;
}

template <typename T>
T& nn::Matrix<T>::operator()(const size_t row, const size_t col)
{
	return this->allocator_.get()[row * this->cols_ + col];
}

template <typename T>
const T& nn::Matrix<T>::operator()(const size_t row, const size_t col) const
{
	return this->allocator_.get()[row * this->cols_ + col];
}

template <typename T>
T& nn::Matrix<T>::operator[](const size_t index)
{
	return this->allocator_.get()[index];
}

template <typename T>
const T& nn::Matrix<T>::operator[](const size_t index) const
{
	return this->allocator_.get()[index];
}

template <typename T>
T nn::Matrix<T>::at(const size_t row, const size_t col) const
{
	return this->allocator_.get()[row * this->cols_ + col];
}

template <typename T>
T nn::Matrix<T>::at(const size_t index) const
{
	return this->allocator_.get()[index];
}

template <typename T>
T* nn::Matrix<T>::get_data()
{
	return this->allocator_.get();
}

template <typename T>
const T* nn::Matrix<T>::get_data() const
{
	return this->allocator_.get();
}

template <typename T>
size_t nn::Matrix<T>::get_rows() const
{
	return this->rows_;
}

template <typename T>
size_t nn::Matrix<T>::get_cols() const
{
	return this->cols_;
}

template <typename T>
void nn::Matrix<T>::clear()
{
	this->allocator_.delete_data();
	this->rows_ = 0;
	this->cols_ = 0;
}

template <typename T>
void nn::Matrix<T>::init(const size_t rows, const size_t cols)
{
	// Check if rows and cols are valid.
	if (rows == 0 || cols == 0)
	{
		throw std::runtime_error("Cannot initialize matrix with 0 rows or 0 columns.");
	}

	// Check if matrix is already initialized.
	if (this->rows_ != 0 || this->cols_ != 0 || this->allocator_.is_initialized())
	{
		throw std::runtime_error("Matrix already initialized.");
	}

	this->allocator_.init(rows * cols);
	this->rows_ = rows;
	this->cols_ = cols;
}

template <typename T>
void nn::Matrix<T>::multiply(const Matrix<T>& matrix1, const Matrix<T>& matrix2, Matrix<T>& result)
{
	// Initialize the result matrix to default values.
	for (size_t i = 0; i < result.rows_ * result.cols_; i++)
	{
		result[i] = T();
	}

	// Perform matrix multiplication.
	for (size_t i = 0; i < matrix1.get_rows(); i++)
	{
		for (size_t k = 0; k < matrix1.get_cols(); k++)
		{
			for (size_t j = 0; j < matrix2.get_cols(); j++)
			{
				result(i, j) += matrix1.at(i, k) * matrix2.at(k, j);
			}
		}
	}
}

template <typename T>
void nn::Matrix<T>::multiply(const Matrix<T>& matrix1, const Matrix<T>& matrix2)
{
	if (matrix1.get_cols() != matrix2.get_rows() || this->get_rows() != matrix1.get_rows() || this->get_cols() !=
		matrix2.get_cols())
	{
		throw std::runtime_error("Cannot multiply matrices with incompatible dimensions.");
	}

	Matrix<T>::multiply(matrix1, matrix2, *this);
}

template<typename T>
inline void nn::Matrix<T>::hadamard_product(const Matrix<T>& other)
{
	this->perform_element_wise_operation(other, [](const T& value1, const T& value2) -> T
	{
		return value1 * value2;
	});
}

template <typename T>
void nn::Matrix<T>::perform_element_wise_operation(const Matrix<T>& other, const std::function<T(T, T)>& operation)
{
	if (this->get_rows() != other.get_rows() || this->get_cols() != other.get_cols())
	{
		throw std::runtime_error("Cannot perform element wise operation on matrices with incompatible dimensions.");
	}

	for (size_t i = 0; i < this->get_rows() * this->get_cols(); i++)
	{
		this->operator[](i) = operation(this->operator[](i), other[i]);
	}
}

template <typename T>
void nn::Matrix<T>::perform_element_wise_operation(const std::function<T(T)>& operation)
{
	for (size_t i = 0; i < this->get_rows() * this->get_cols(); i++)
	{
		this->operator[](i) = operation(this->operator[](i));
	}
}

template <typename T>
void nn::Matrix<T>::randomize(const T& min, const T& max)
{
	// Create a thread safe random number generator.
	const auto random_number_generator = [min, max]() -> T
	{
		std::random_device random_device;
		std::mt19937 random_engine(random_device());
		std::uniform_real_distribution<T> distribution(min, max);

		return distribution(random_engine);
	};

	for (size_t i = 0; i < this->get_rows() * this->get_cols(); i++)
	{
		this->operator[](i) = random_number_generator();
	}
}

template <typename T>
nn::Matrix<T> nn::Matrix<T>::transpose() const
{
	// Initialize the result matrix.
	Matrix<T> result(this->get_cols(), this->get_rows());

	for (size_t i = 0; i < this->get_rows(); i++)
	{
		for (size_t j = 0; j < this->get_cols(); j++)
		{
			result(j, i) = this->at(i, j);
		}
	}

	return result;
}

template<typename T>
inline void nn::Matrix<T>::calculate_sums_for_forward_propagation(const Matrix<T>& weights, const Matrix<T>& biases, const Matrix<T>& input)
{
	// Check if dimensions are compatible.
	if (biases.get_rows() != this->get_rows() || biases.get_rows() != 1)
	{
		throw std::runtime_error("Cannot calculate sums for forward propagation with incompatible dimensions.");
	}

	// Calculate sums.
	Matrix<T>::multiply(weights, input, *this);
	this->perform_element_wise_operation([&](const T& value) -> T
	{
		return value + biases[&value - &this->operator[](0)];
	});
}

template<typename T>
inline void nn::Matrix<T>::calculate_delta_activation_for_back_propagation(const Matrix<T>& next_layer_weights, const Matrix<T>& next_layer_delta_sums)
{
	// Transpose the weights matrix.
	const auto transpose_next_layer_weights = next_layer_weights.transpose();
	// Multiply the transposed weights matrix with the delta sums matrix.
	this->multiply(transpose_next_layer_weights, next_layer_delta_sums);
}

template<typename T>
inline void nn::Matrix<T>::calculate_delta_biases_for_back_propagation(const Matrix<T>& this_layer_delta_sums)
{
	// Check if dimensions are compatible.
	if (this->get_cols() != 1 || this->get_rows() != this_layer_delta_sums.get_rows())
	{
		throw std::runtime_error("Cannot calculate delta biases for back propagation with incompatible dimensions.");
	}

	// Calculate delta biases.
	for (size_t i = 0; i < this->get_rows(); i++)
	{
		T res = T();
		for (size_t j = 0; j < this_layer_delta_sums.get_cols(); j++)
		{
			res += this_layer_delta_sums.at(i, j);
		}
		this->operator[](i) = res / this_layer_delta_sums.get_cols();
	}
}

template<typename T>
inline void nn::Matrix<T>::calculate_delta_weights_for_back_propagation(const Matrix<T>& previous_layer_activations, const Matrix<T>& this_layer_delta_sums)
{
	const auto transpose_previous_layer_activation = previous_layer_activations.transpose();

	this->multiply(this_layer_delta_sums, transpose_previous_layer_activation);

	// delta_weights / batch_size
	this->perform_element_wise_operation([&](const T& value) -> T
	{
		return value / this_layer_delta_sums.get_cols();
	});
}

template<typename T>
inline void nn::Matrix<T>::calculate_delta_activation_from_expected_output(const Matrix<T>& this_layer_activations, const Matrix<T>& expected_output)
{
	*this = this_layer_activations;

	this->perform_element_wise_operation(expected_output, [&](const T& value1, const T& value2) -> T
	{
		return (value1 - value2) * static_cast<T>(2.0);
	});
}


/// <summary>
/// Overload of the << operator to print the matrix.
/// </summary>
template <typename T>
std::ostream& operator<<(std::ostream& os, const nn::Matrix<T>& matrix)
{
	for (size_t i = 0; i < matrix.get_rows(); i++)
	{
		for (size_t j = 0; j < matrix.get_cols(); j++)
		{
			os << matrix(i, j) << " ";
		}
		os << "\n";
	}

	return os;
}
#pragma endregion
