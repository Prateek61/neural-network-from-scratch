// File: test/AlignedMemoryAllocatorTest.cpp
// Purpose: Test file for AlignedMemoryAllocator.cpp.

#include <gtest/gtest.h>

#include "AlignedMemoryAllocator.h"

// Test case for default constructor
TEST(AlignedMemoryAllocatorTest, DefaultConstructor) {
    nn::utils::AlignedMemoryAllocator<int, 32> allocator;
    ASSERT_FALSE(allocator.is_initialized());
    ASSERT_EQ(allocator.get(), nullptr);
    ASSERT_EQ(allocator.get_size(), 0);
}

// Test case for initialization with a specific size
TEST(AlignedMemoryAllocatorTest, InitializationWithSize) {
    nn::utils::AlignedMemoryAllocator<int, 32> allocator(100);
    ASSERT_TRUE(allocator.is_initialized());
    ASSERT_NE(allocator.get(), nullptr);
    ASSERT_EQ(allocator.get_size(), 100);
    ASSERT_EQ(allocator.get(), allocator.get());
    ASSERT_NO_THROW(allocator.delete_data());
    ASSERT_FALSE(allocator.is_initialized());
    ASSERT_EQ(allocator.get(), nullptr);
    ASSERT_EQ(allocator.get_size(), 0);
}

// Test case for copying data between two allocators of same alignment
TEST(AlignedMemoryAllocatorTest, CopyDataBetweenSameAlignment) {
    nn::utils::AlignedMemoryAllocator<int, 32> source_allocator(100);
    nn::utils::AlignedMemoryAllocator<int, 32> destination_allocator(100);

    // Modify source data
    for (int i = 0; i < 100; ++i) {
        source_allocator.get()[i] = i;
    }

    auto copy_data_between_alignments = [&]()
    {
        nn::utils::AlignedMemoryAllocator<int, 32>::copy_data_between_alignments(source_allocator, destination_allocator);
    };

    // Copy data from source to destination
    ASSERT_NO_THROW(copy_data_between_alignments());

    // Ensure the data is copied correctly
    for (int i = 0; i < 100; ++i) {
        ASSERT_EQ(source_allocator.get()[i], destination_allocator.get()[i]);
    }
}

// Test case for copying data when sizes do not match
TEST(AlignedMemoryAllocatorTest, CopyDataSizeMismatch) {
    nn::utils::AlignedMemoryAllocator<int, 32> source_allocator(100);
    nn::utils::AlignedMemoryAllocator<int, 32> destination_allocator(50);

    auto copy_data_between_alignments = [&]()
    {
    	nn::utils::AlignedMemoryAllocator<int, 32>::copy_data_between_alignments(source_allocator, destination_allocator);
	};

    // Ensure copying data with different sizes throws an exception
    ASSERT_THROW(copy_data_between_alignments(), std::logic_error);
}

// Test case for checking copy_data method with same size
TEST(AlignedMemoryAllocatorTest, CopyDataSameSize) {
    nn::utils::AlignedMemoryAllocator<int, 32> source_allocator(100);
    nn::utils::AlignedMemoryAllocator<int, 32> destination_allocator(100);

    // Modify source data
    for (int i = 0; i < 100; ++i) {
        source_allocator.get()[i] = i;
    }

    // Copy data from source to destination
    destination_allocator.copy_data(source_allocator);

    // Ensure the data is copied correctly
    for (int i = 0; i < 100; ++i) {
        ASSERT_EQ(source_allocator.get()[i], destination_allocator.get()[i]);
    }
}

// Test case for checking copy_data method with different sizes
TEST(AlignedMemoryAllocatorTest, CopyDataDifferentSize) {
    const nn::utils::AlignedMemoryAllocator<int, 32> source_allocator(100);
    nn::utils::AlignedMemoryAllocator<int, 32> destination_allocator(50);

    // Ensure copying data with different sizes throws an exception
    ASSERT_THROW(destination_allocator.copy_data(source_allocator), std::logic_error);
}

