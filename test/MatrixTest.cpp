// File: test/MatrixTest.cpp
// Purpose: Test file for Matrix.cpp.

#include <gtest/gtest.h>

#include "Matrix.h"

// Test case for default constructor
TEST(MatrixTest, DefaultConstructor) {
    nn::Matrix<int> matrix;
    ASSERT_EQ(matrix.get_rows(), static_cast<size_t>(0));
    ASSERT_EQ(matrix.get_cols(), static_cast<size_t>(0));
    ASSERT_EQ(matrix.get_data(), nullptr);
}

// Test case for initialization with specific size
TEST(MatrixTest, InitializationWithSize) {
    nn::Matrix<int> matrix(2, 3);
    ASSERT_EQ(matrix.get_rows(), 2);
    ASSERT_EQ(matrix.get_cols(), 3);
    ASSERT_NE(matrix.get_data(), nullptr);
}

// Test case for copy constructor
TEST(MatrixTest, CopyConstructor) {
    nn::Matrix<int> matrix1(2, 3);

    // Initialize matrix data
    for (size_t i = 0; i < 6; ++i) {
        matrix1[i] = static_cast<int>(i) + 1;
    }

    nn::Matrix<int> matrix2(matrix1);

    ASSERT_EQ(matrix2.get_rows(), 2);
    ASSERT_EQ(matrix2.get_cols(), 3);
    ASSERT_NE(matrix2.get_data(), nullptr);

    for (size_t i = 0; i < 6; ++i) {
        ASSERT_EQ(matrix1[i], matrix2[i]);
    }
}

// Test case for element access using () operator
TEST(MatrixTest, ElementAccessOperator) {
    nn::Matrix<int> matrix(2, 2);

    matrix(0, 0) = 10;
    matrix(0, 1) = 20;
    matrix(1, 0) = 30;
    matrix(1, 1) = 40;

    ASSERT_EQ(matrix(0, 0), 10);
    ASSERT_EQ(matrix(0, 1), 20);
    ASSERT_EQ(matrix(1, 0), 30);
    ASSERT_EQ(matrix(1, 1), 40);
}

// Test case for matrix multiplication
TEST(MatrixTest, MatrixMultiplication) {
    // Initialize matrices
    nn::Matrix<int> mat1(2, 3);
    nn::Matrix<int> mat2(3, 2);
    nn::Matrix<int> result(2, 2);

    // Initialize matrix data (you may use loops or fill the matrices appropriately)

    // Fill mat1 with values
    mat1(0, 0) = 1;
    mat1(0, 1) = 2;
    mat1(0, 2) = 3;
    mat1(1, 0) = 4;
    mat1(1, 1) = 5;
    mat1(1, 2) = 6;

    // Fill mat2 with values
    mat2(0, 0) = 7;
    mat2(0, 1) = 8;
    mat2(1, 0) = 9;
    mat2(1, 1) = 10;
    mat2(2, 0) = 11;
    mat2(2, 1) = 12;

    // Expected result for the multiplication
    constexpr int expectedResult[4] = { 58, 64, 139, 154 };

    // Perform matrix multiplication
    ASSERT_NO_THROW(nn::Matrix<int>::multiply(mat1, mat2, result));

    // Validate the result
    for (size_t i = 0; i < result.get_rows(); ++i) {
        for (size_t j = 0; j < result.get_cols(); ++j) {
            ASSERT_EQ(result(i, j), expectedResult[i * result.get_cols() + j]);
        }
    }
}
