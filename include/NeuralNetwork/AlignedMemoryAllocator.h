// File: include/NeuralNetwork/AlignedMemoryAllocator.h
// Purpose: Header file for AlignedMemoryAllocator template class.

#pragma once

#include <stdexcept> // std::logic_error
#include <iostream> // std::ostream
#include <cstring> // memcpy

namespace nn::utils
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
		[[nodiscard]] T* get();

		/// <summary>
		/// Returns the aligned memory
		/// </summary>
		[[nodiscard]] const T* get() const;

		/// <summary>
		/// Returns the size of the memory.(number of elements)
		/// </summary>
		[[nodiscard]] size_t get_size() const;
	};
}


#pragma region Template Implementation

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
nn::utils::AlignedMemoryAllocator<T, Alignment>::~AlignedMemoryAllocator()
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

#ifdef _WIN32
	this->data_ = static_cast<T*>(_aligned_malloc(size * sizeof(T), Alignment));
	this->aligned_data_ = this->data_;
#else
	this->data_ = new T[size];
	this->aligned_data_ = this->data_;
#endif
}

template <typename T, size_t Alignment>
void nn::utils::AlignedMemoryAllocator<T, Alignment>::delete_data()
{
	this->initialized_ = false;
	this->size_ = 0;



	if (this->data_)
	{
#ifdef _WIN32
		_aligned_free(this->data_);
#else
		delete[] this->data_;
#endif
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
	memcpy(static_cast<void*>(destination.get()), static_cast<const void*>(source.get()),
		source.size_ * sizeof(T));
}

template <typename T, size_t Alignment>
bool nn::utils::AlignedMemoryAllocator<T, Alignment>::is_initialized() const
{
	return this->initialized_;
}

template <typename T, size_t Alignment>
T* nn::utils::AlignedMemoryAllocator<T, Alignment>::get()
{
	return this->aligned_data_;
}

template<typename T, size_t Alignment>
inline const T* nn::utils::AlignedMemoryAllocator<T, Alignment>::get() const
{
	return this->aligned_data_;
}

template<typename T, size_t Alignment>
inline size_t nn::utils::AlignedMemoryAllocator<T, Alignment>::get_size() const
{
	return this->size_;
}

template <typename T, size_t Alignment>
std::ostream& operator<<(std::ostream& os, const nn::utils::AlignedMemoryAllocator<T, Alignment>& allocator)
{
	// Loop through elements and print
	for (size_t i = 0; i < allocator.size_; i++)
	{
		os << allocator.aligned_data_[i] << " ";
	}

	return os;
}

#pragma endregion