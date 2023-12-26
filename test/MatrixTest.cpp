// File: test/MatrixTest.cpp
// Purpose: Test file for Matrix.cpp.

#include <gtest/gtest.h>
#include <NeuralNetwork/Matrix.h>

#include <vector>
#include <iostream>

#define VEC std::vector

// Test case for default constructor
TEST(MatrixTest, DefaultConstructor)
{
	nn::Matrix<int> matrix;
	ASSERT_EQ(matrix.get_rows(), static_cast<size_t>(0));
	ASSERT_EQ(matrix.get_cols(), static_cast<size_t>(0));
	ASSERT_EQ(matrix.get_data(), nullptr);
}

// Test case for initialization with specific size
TEST(MatrixTest, InitializationWithSize)
{
	nn::Matrix<int> matrix(2, 3);
	ASSERT_EQ(matrix.get_rows(), 2);
	ASSERT_EQ(matrix.get_cols(), 3);
	ASSERT_NE(matrix.get_data(), nullptr);
}

// Test case for copy constructor
TEST(MatrixTest, CopyConstructor)
{
	nn::Matrix<int> matrix1(2, 3);

	// Initialize matrix data
	for (size_t i = 0; i < 6; ++i)
	{
		matrix1[i] = static_cast<int>(i) + 1;
	}

	nn::Matrix<int> matrix2(matrix1);

	ASSERT_EQ(matrix2.get_rows(), 2);
	ASSERT_EQ(matrix2.get_cols(), 3);
	ASSERT_NE(matrix2.get_data(), nullptr);

	for (size_t i = 0; i < 6; ++i)
	{
		ASSERT_EQ(matrix1[i], matrix2[i]);
	}
}

// Test case for element access using () operator
TEST(MatrixTest, ElementAccessOperator)
{
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
TEST(MatrixTest, MatrixMultiplication)
{
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
	constexpr int expected_result[4] = {58, 64, 139, 154};

	// Perform matrix multiplication
	ASSERT_NO_THROW(nn::Matrix<int>::multiply(mat1, mat2, result));

	// Validate the result
	for (size_t i = 0; i < result.get_rows(); ++i)
	{
		for (size_t j = 0; j < result.get_cols(); ++j)
		{
			ASSERT_EQ(result(i, j), expected_result[i * result.get_cols() + j]);
		}
	}
}

// Test case for perform_element_wise_operation 2 matrices
TEST(MatrixTest, PerformElementWiseOperation2)
{
	// Initialize matrices
	nn::Matrix mat1 = VEC<VEC<float>>({{1.0f, 2.0f}, {3.0f, 4.0f}});
	const nn::Matrix mat2 = VEC<VEC<float>>({{5.0f, 6.0f}, {7.0f, 8.0f}});

	nn::Matrix result = VEC<VEC<float>>({{6.0f, 8.0f}, {10.0f, 12.0f}});

	mat1.perform_element_wise_operation(mat2, [](const float a, const float b) { return a + b; });

	// Validate the result
	for (size_t i = 0; i < result.get_rows(); ++i)
	{
		for (size_t j = 0; j < result.get_cols(); ++j)
		{
			ASSERT_NEAR(mat1(i, j), result(i, j), 0.0001f);
		}
	}
}

// Test for calculate_sums_for_forward_propagation
TEST(MatrixTest, CalculateSumsForForwardPropagation)
{
	// Initialize matrices
	nn::Matrix<float> sums(2, 1);
	const nn::Matrix weights = VEC<VEC<float>>({{1.0f, 2.0f}, {3.0f, 4.0f}});
	const nn::Matrix biases = VEC<VEC<float>>({{-5.0f}, {6.0f}});
	const nn::Matrix input = VEC<VEC<float>>({{1.0f}, {-2.0f}});

	nn::Matrix result = VEC<VEC<float>>({{-8.0f}, {1.0f}});

	// Calculate sums
	sums.calculate_sums_for_forward_propagation(weights, biases, input);

	// Validate the result
	for (size_t i = 0; i < result.get_rows(); ++i)
	{
		for (size_t j = 0; j < result.get_cols(); ++j)
		{
			ASSERT_NEAR(sums(i, j), result(i, j), 0.0001f);
		}
	}
}

// Test for calculate_delta_activation_from_expected_output
TEST(MatrixTest, CalculateDeltaActivationFromExpectedOutput)
{
	// Initialize matrices
	nn::Matrix<float> delta_activation(2, 1);
	const nn::Matrix this_layer_activation = VEC<VEC<float>>({{-1.0f}, {-1.0f}});
	const nn::Matrix expected_output = VEC<VEC<float>>({{2.0f}, {-1.0f}});

	const nn::Matrix result = VEC<VEC<float>>({{-6.0f}, {0.0f}});

	// Calculate delta activation
	delta_activation.calculate_delta_activation_from_expected_output(this_layer_activation, expected_output);

	// Validate the result
	for (size_t i = 0; i < result.get_rows(); ++i)
	{
		for (size_t j = 0; j < result.get_cols(); ++j)
		{
			ASSERT_NEAR(delta_activation(i, j), result(i, j), 0.0001f);
		}
	}
}

// Test for calculate_delta_bias_for_back_propagation
TEST(MatrixTest, CalculateDeltaBiasForBackPropagation)
{
	// Initialize matrices
	nn::Matrix<float> delta_bias(2, 1);
	const nn::Matrix delta_activation = VEC<VEC<float>>(
		{
			{-2.0f, -3.0f},
			{4.0f, 5.0f}
		}
	);

	const nn::Matrix result = VEC<VEC<float>>({{-2.5f}, {4.5f}});

	// Calculate delta bias
	delta_bias.calculate_delta_biases_for_back_propagation(delta_activation);

	// Validate the result
	for (size_t i = 0; i < result.get_rows(); ++i)
	{
		for (size_t j = 0; j < result.get_cols(); ++j)
		{
			ASSERT_NEAR(delta_bias(i, j), result(i, j), 0.0001f);
		}
	}
}

// Test for calculate_delta_weights_for_back_propagation
TEST(MatrixTest, CalculateDeltaWeightsForBackPropagation)
{
	// Initialize matrices
	nn::Matrix<float> delta_weights(2, 2);

	nn::Matrix previous_layer_activation = VEC<VEC<float>>(
		{
			{-4.0f, 2.0f},
			{3.0f, -1.0f}
		}
	);

	nn::Matrix this_layer_delta_sums = VEC<VEC<float>>(
		{
			{1.0f, -2.0f},
			{-3.0f, 4.0f}
		}
	);

	nn::Matrix result = VEC<VEC<float>>(
		{
			{-4.0f, 2.5f},
			{10.0f, -6.5f}
		}
	);

	// Calculate delta weights
	delta_weights.calculate_delta_weights_for_back_propagation(previous_layer_activation, this_layer_delta_sums);

	// Validate the result
	for (size_t i = 0; i < result.get_rows(); ++i)
	{
		for (size_t j = 0; j < result.get_cols(); ++j)
		{
			ASSERT_NEAR(delta_weights(i, j), result(i, j), 0.0001f);
		}
	}
}

// Test for calculate_delta_activation_for_back_propagation
TEST(MatrixTest, CalculateDeltaActivationForBackPropagation)
{
	// Initialize matrices
	nn::Matrix<float> delta_activation(2, 2);

	const nn::Matrix next_layer_weights = VEC<VEC<float>>(
		{
			{1.0f, 2.0f},
			{-2.0f, 1.0f}
		}
	);

	const nn::Matrix next_layer_delta_activation = VEC<VEC<float>>(
		{
			{1.0f, 2.0f},
			{3.0f, 4.0f}
		}
	);

	nn::Matrix result = VEC<VEC<float>>(
		{
			{-5.0f, -6.0f},
			{5.0f, 8.0f}
		}
	);

	// Calculate delta activation
	delta_activation.calculate_delta_activation_for_back_propagation(next_layer_weights, next_layer_delta_activation);

	// Validate the result
	for (size_t i = 0; i < result.get_rows(); ++i)
	{
		for (size_t j = 0; j < result.get_cols(); ++j)
		{
			ASSERT_NEAR(delta_activation(i, j), result(i, j), 0.0001f);
		}
	}
}
