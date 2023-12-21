// File: include/Matrix.h
// Purpose: Header file for Matrix.cpp.

#pragma once

#include "AlignedMemoryAllocator.h" // nn::utils::AlignedMemoryAllocator

#include <stdexcept> // std::runtime_error
#include <Matrix.h> // nn::Matrix

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
		/// Copy constructor.
		/// </summary>
		Matrix(const Matrix<T>& other);

		/// <summary>
		/// Delete the assignment operator.
		/// </summary>
		Matrix& operator=(const Matrix<T>& other) = delete;

		/// <summary>
		/// Returns the element at row, col.
		/// </summary>
		/// <returns>Reference to the element at row, col</returns>
		[[nodiscard]] T& operator()(size_t row, size_t col);

		/// <summary>
		/// Returns the element at row, col.
		/// </summary>
		/// <returns>Copy of element at row, col</returns>
		[[nodiscard]] T operator()(size_t row, size_t col) const;

		/// <summary>
		/// Returns the element at index of the matrix data array.
		/// </summary>
		/// <returns>Reference to the element at index</returns>
		[[nodiscard]] T& operator[](size_t index);

		/// <summary>
		/// Returns the element at index of the matrix data array.
		/// </summary>
		/// <returns>Copy of the element at index</returns>
		[[nodiscard]] T operator[](size_t index) const;

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
nn::Matrix<T>::Matrix(const Matrix<T>& other)
	: rows_(0), cols_(0)
{
	this->init(other.get_rows(), other.get_cols());

	// Copy data from other matrix.
	this->allocator_.copy_data(other.allocator_);
}


template <typename T>
T& nn::Matrix<T>::operator()(size_t row, size_t col)
{
	return this->allocator_.get()[row * this->cols_ + col];
}


template <typename T>
T nn::Matrix<T>::operator()(size_t row, size_t col) const
{
	return this->allocator_.get()[row * this->cols_ + col];
}


template <typename T>
T& nn::Matrix<T>::operator[](size_t index)
{
	return this->allocator_.get()[index];
}

template <typename T>
T nn::Matrix<T>::operator[](size_t index) const
{
	return this->allocator_.get()[index];
}


template <typename T>
T nn::Matrix<T>::at(size_t row, size_t col) const
{
	return this->allocator_.get()[row * this->cols_ + col];
}

template <typename T>
T nn::Matrix<T>::at(size_t index) const
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
void nn::Matrix<T>::init(size_t rows, size_t cols)
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
	if (matrix1.get_cols() != matrix2.get_rows())
	{
		throw std::runtime_error("Cannot multiply matrices with incompatible dimensions.");
	}

	Matrix<T>::multiply(matrix1, matrix2, *this);
}

#pragma endregion
