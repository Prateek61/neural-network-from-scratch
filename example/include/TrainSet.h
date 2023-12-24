#pragma once

#include <NeuralNetwork/DataSet.h>
#include <NeuralNetwork/Matrix.h>
#include <vector>
#include <memory>

#include <string_view>

class TrainSet : public nn::DataSet
{
public:
	std::vector<std::unique_ptr<nn::Matrix<float>>> images_in_batches;
	std::vector<std::unique_ptr<nn::Matrix<float>>> labels_in_batches;
	std::string_view images_file_path;
	std::string_view labels_file_path;

	size_t num_batches;
	size_t output_size;
	size_t num_images;
	size_t num_labels;

	size_t num_rows;
	size_t num_cols;

	TrainSet(const std::string_view images_file_path, const std::string_view labels_file_path);

	void initialize(const size_t batch_size) override;
	nn::Matrix<float>& get_batch_input() override;
	nn::Matrix<float>& get_batch_output() override;
	[[nodiscard]] bool is_end() const override;
	[[nodiscard]] bool is_ready() const override;
	void reset() override;
	[[nodiscard]] size_t get_input_size() const override;
	[[nodiscard]] size_t get_output_size() const override;
	[[nodiscard]] size_t get_total_size() const override;

	[[nodiscard]] bool is_files_good() const;
};