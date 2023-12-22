// File: include/Matrix.h
// Purpose: Header file for Matrix.cpp.

#pragma once

#include <random>
#include <functional>

#include "AlignedMemoryAllocator.h" // nn::utils::AlignedMemoryAllocator

#include <stdexcept> // std::runtime_error
#include <Matrix.h>	 // nn::Matrix
#include <vector>	 // std::vector
#include <iostream>

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
		explicit Matrix(const std::vector<std::vector<T>> &data);

		/// <summary>
		/// Constructor for a matrix of size rows x cols.
		/// </summary>
		/// <param name="data">1d vector representing the matrix</param>
		/// <param name="rows">Number of rows in this matrix</param>
		/// <param name="cols">Number of columns in this matrix</param>
		Matrix(const std::vector<T> &data, size_t rows, size_t cols);

		/// <summary>
		/// Copy constructor.
		/// </summary>
		Matrix(const Matrix<T> &other);

		/// <summary>
		/// Delete the assignment operator.
		/// </summary>
		Matrix &operator=(const Matrix<T> &other) = delete;

		/// <summary>
		/// Returns the element at row, col.
		/// </summary>
		/// <returns>Reference to the element at row, col</returns>
		[[nodiscard]] T &operator()(size_t row, size_t col);

		/// <summary>
		/// Returns the element at row, col.
		/// </summary>
		/// <returns>Copy of element at row, col</returns>
		[[nodiscard]] const T &operator()(size_t row, size_t col) const;

		/// <summary>
		/// Returns the element at index of the matrix data array.
		/// </summary>
		/// <returns>Reference to the element at index</returns>
		[[nodiscard]] T &operator[](size_t index);

		/// <summary>
		/// Returns the element at index of the matrix data array.
		/// </summary>
		/// <returns>Copy of the element at index</returns>
		[[nodiscard]] const T &operator[](size_t index) const;

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
		[[nodiscard]] T *get_data();

		/// <summary>
		/// Returns the matrix data.
		/// </summary>
		/// <returns>Array to the type T</returns>
		[[nodiscard]] const T *get_data() const;

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
		static void multiply(const Matrix<T> &matrix1, const Matrix<T> &matrix2, Matrix<T> &result);

		/// <summary>
		/// Performs matrix multiplication on matrix1 and matrix2 and stores the result in this matrix.
		/// </summary>
		/// <param name="matrix1">First Matrix</param>
		/// <param name="matrix2">Second Matrix</param>
		void multiply(const Matrix<T> &matrix1, const Matrix<T> &matrix2);

		/// <summary>
		/// Performs an element wise operation on this matrix with other matrix and stores the result in this matrix.
		/// </summary>
		/// <param name="other">Other matrix</param>
		/// <param name="operation">Function (First argument is value of this and Second is value of other)</param>
		void perform_element_wise_operation(const Matrix<T> &other, const std::function<T(T, T)> &operation);

		/// <summary>
		/// Performs an element wise operation on this matrix.
		/// </summary>
		/// <param name="operation">Function</param>
		void perform_element_wise_operation(const std::function<T(T)> &operation);

		/// <summary>
		/// Randomizes the contents of this matrix between min and max.
		/// </summary>
		void randomize(const T &min, const T &max);

		friend std::ostream &operator<<(std::ostream &OUT, const Matrix<T> &mat)
		{

			for (int x = 0; x < mat.get_cols(); x++)
			{
				for (int y = 0; y < mat.get_rows(); y++)
				{

					OUT << mat(x, y) << "\t";
				}
				OUT << std::endl;
			}
			return OUT;
		}
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
nn::Matrix<T>::Matrix(const std::vector<std::vector<T>> &data)
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
nn::Matrix<T>::Matrix(const std::vector<T> &data, const size_t rows, const size_t cols)
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
nn::Matrix<T>::Matrix(const Matrix<T> &other)
	: rows_(0), cols_(0)
{
	this->init(other.get_rows(), other.get_cols());

	// Copy data from other matrix.
	this->allocator_.copy_data(other.allocator_);
}

template <typename T>
T &nn::Matrix<T>::operator()(const size_t row, const size_t col)
{
	return this->allocator_.get()[row * this->cols_ + col];
}

template <typename T>
const T &nn::Matrix<T>::operator()(const size_t row, const size_t col) const
{
	return this->allocator_.get()[row * this->cols_ + col];
}

template <typename T>
T &nn::Matrix<T>::operator[](const size_t index)
{
	return this->allocator_.get()[index];
}

template <typename T>
const T &nn::Matrix<T>::operator[](const size_t index) const
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
T *nn::Matrix<T>::get_data()
{
	return this->allocator_.get();
}

template <typename T>
const T *nn::Matrix<T>::get_data() const
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
void nn::Matrix<T>::multiply(const Matrix<T> &matrix1, const Matrix<T> &matrix2, Matrix<T> &result)
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
void nn::Matrix<T>::multiply(const Matrix<T> &matrix1, const Matrix<T> &matrix2)
{
	if (matrix1.get_cols() != matrix2.get_rows() || this->get_rows() != matrix1.get_rows() || this->get_cols() != matrix2.get_cols())
	{
		throw std::runtime_error("Cannot multiply matrices with incompatible dimensions.");
	}

	Matrix<T>::multiply(matrix1, matrix2, *this);
}

template <typename T>
void nn::Matrix<T>::perform_element_wise_operation(const Matrix<T> &other, const std::function<T(T, T)> &operation)
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
void nn::Matrix<T>::perform_element_wise_operation(const std::function<T(T)> &operation)
{
	for (size_t i = 0; i < this->get_rows() * this->get_cols(); i++)
	{
		this->operator[](i) = operation(this->operator[](i));
	}
}

template <typename T>
inline void nn::Matrix<T>::randomize(const T &min, const T &max)
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

#pragma endregion