// File: include/matrix.h
// Purpose: Header file for matrix.cpp.

#pragma once
#include <stdexcept>

namespace nn
{
	namespace utils
	{
		// TODO: Implement this class.

		/// <summary>
		/// Class for allocating aligned memory.
		/// </summary>
		/// <typeparam name="T">Type of data to allocate memory for.</typeparam>
		/// <typeparam name="Alignment">Alignment of memory in bytes.</typeparam>
		template <typename T, size_t Alignment>
		class AlignedMemoryAllocator
		{
		private:
			/// <summary>
			/// Is the memory initialized?
			/// </summary>
			bool initialized_;

			/// <summary>
			/// The whole of the memory.
			/// </summary>
			T* data_;

			/// <summary>
			/// The aligned memory.
			/// </summary>
			T* aligned_data_;

			/// <summary>
			/// Size of the memory.
			/// </summary>
			size_t size_;

		public:
			/// <summary>
			/// Default constructor.
			/// </summary>
			AlignedMemoryAllocator();

			/// <summary>
			/// Allocates memory of size * sizeof(T) and aligns it to alignment.
			/// </summary>
			/// <param name="size">Number of elements to allocate the memory for</param>
			explicit AlignedMemoryAllocator(size_t size);

			/// <summary>
			/// Delete the copy constructor.
			/// </summary>
			AlignedMemoryAllocator(const AlignedMemoryAllocator<T, Alignment>&) = delete;

			/// <summary>
			/// Delete the move constructor.
			/// </summary>
			AlignedMemoryAllocator(AlignedMemoryAllocator<T, Alignment>&&) = delete;

			/// <summary>
			/// Delete the copy assignment operator.
			/// </summary>
			AlignedMemoryAllocator& operator=(const AlignedMemoryAllocator<T, Alignment>&) = delete;

			/// <summary>
			/// Delete the move assignment operator.
			/// </summary>
			AlignedMemoryAllocator& operator=(AlignedMemoryAllocator<T, Alignment>&&) = delete;

			/// <summary>
			/// Frees the memory.
			/// </summary>
			~AlignedMemoryAllocator();

			/// <summary>
			/// Allocates memory of size * sizeof(T) and aligns it to alignment.
			/// </summary>
			/// <param name="size">Number of elements to allocate the memory for</param>
			void init(size_t size);

			/// <summary>
			/// Frees the memory.
			/// </summary>
			void delete_data();

			/// <summary>
			/// Copies the data from other to this.
			/// </summary>
			void copy_data(const AlignedMemoryAllocator<T, Alignment>& other);

			/// <summary>
			/// Copies the data from source to destination.
			/// </summary>
			/// <typeparam name="Alignment2">Alignment of destination allocator</typeparam>
			template <size_t Alignment2>
			static void copy_data_between_alignments(const AlignedMemoryAllocator<T, Alignment>& source,
			                                         AlignedMemoryAllocator<T, Alignment2>& destination);

			/// <summary>
			/// Returns if the memory is initialized.
			/// </summary>
			[[nodiscard]] bool is_initialized() const;

			/// <summary>
			/// Returns the aligned memory.
			/// </summary>
			[[nodiscard]] T* get_aligned_data() const;

			/// <summary>
			/// Returns the size of the memory.(number of elements)
			/// </summary>
			[[nodiscard]] size_t get_size() const;
		};
	}

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
		/// Assignment operator.
		/// </summary>
		Matrix& operator=(const Matrix<T>& other);

		/// <summary>
		/// Returns the element at row, col.
		/// </summary>
		/// <returns>Reference to the element at row, col</returns>
		[[nodiscard]] T& operator()(size_t row, size_t col);

		/// <summary>
		/// Returns the element at index of the matrix data array.
		/// </summary>
		/// <returns>Reference to the element at index</returns>
		[[nodiscard]] T& operator[](size_t index);

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
		/// Returns the number of rows in the matrix.
		/// </summary>
		size_t get_rows() const;

		/// <summary>
		/// Returns the number of columns in the matrix.
		/// </summary>
		size_t get_cols() const;

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
template <typename T, size_t Alignment>
nn::utils::AlignedMemoryAllocator<T, Alignment>::AlignedMemoryAllocator()
	: initialized_(false), data_(nullptr), aligned_data_(nullptr), size_(0)
{
}

template <typename T, size_t Alignment>
nn::utils::AlignedMemoryAllocator<T, Alignment>::AlignedMemoryAllocator(const size_t size)
	: initialized_(false), data_(nullptr), aligned_data_(nullptr), size_(0)
{
	this->init(size);
}

template <typename T, size_t Alignment>
nn::utils::AlignedMemoryAllocator<T, Alignment>::~AlignedMemoryAllocator<T, Alignment>()
{
	this->delete_data();
}

template <typename T, size_t Alignment>
void nn::utils::AlignedMemoryAllocator<T, Alignment>::init(const size_t size)
{
	// If the memory is already initialized, throw exception
	if (this->initialized_)
	{
		throw std::logic_error("Memory already initialized.");
	}

	this->initialized_ = true;
	this->size_ = size;
	this->data_ = new T[size];
	this->aligned_data_ = this->data_;
}

template <typename T, size_t Alignment>
void nn::utils::AlignedMemoryAllocator<T, Alignment>::delete_data()
{
	this->initialized_ = false;
	this->size_ = 0;

	if (this->data_)
	{
		delete this->data_;
	}

	this->data_ = nullptr;
	this->aligned_data_ = nullptr;
}

template <typename T, size_t Alignment>
void nn::utils::AlignedMemoryAllocator<T, Alignment>::copy_data(const AlignedMemoryAllocator<T, Alignment>& other)
{
	if (this->size_ != other.size_)
	{
		throw std::logic_error("Cannot copy, size doesn't match");
	}

	// Copies memory from other.aligned_data to this->aligned_data
	memcpy(static_cast<void*>(this->aligned_data_), static_cast<void*>(other.aligned_data_), this->size_ * sizeof(T));
}

template <typename T, size_t Alignment>
template <size_t Alignment2>
void nn::utils::AlignedMemoryAllocator<T, Alignment>::copy_data_between_alignments(
	const AlignedMemoryAllocator<T, Alignment>& source, AlignedMemoryAllocator<T, Alignment2>& destination)
{
	if (source.get_size() != destination.get_size())
	{
		throw std::logic_error("Cannot copy, size doesn't match");
	}

	// Copies memory from source.aligned_data to destination.aligned_data
	memcpy(static_cast<void*>(destination.get_aligned_data()), static_cast<void*>(source.get_aligned_data()),
	       source.size_ * sizeof(T));
}

template <typename T, size_t Alignment>
bool nn::utils::AlignedMemoryAllocator<T, Alignment>::is_initialized() const
{
	return this->initialized_;
}

template <typename T, size_t Alignment>
T* nn::utils::AlignedMemoryAllocator<T, Alignment>::get_aligned_data() const
{
	return this->aligned_data_;
}

template<typename T, size_t Alignment>
inline size_t nn::utils::AlignedMemoryAllocator<T, Alignment>::get_size() const
{
	return this->size_;
}

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

	throw std::runtime_error("Not implemented");
}

#pragma endregion
