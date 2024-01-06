#include "TrainSet.h"

#include <iostream>
#include <fstream>
#include <cmath>

#include <chrono>

TrainSet::TrainSet(const std::string_view images_file_path, const std::string_view labels_file_path)
	: DataSet(), images_file_path(images_file_path), labels_file_path(labels_file_path), num_batches(0), output_size(0), num_images(0), num_labels(0), num_rows(0), num_cols(0)
{
	if (!this->is_files_good())
	{
		throw std::runtime_error("Files don't exists or couldn't be read.");
	}

	// Open file
	std::ifstream images_file(this->images_file_path.data(), std::ios::in | std::ios::binary);
	std::ifstream labels_file(this->labels_file_path.data(), std::ios::in | std::ios::binary);

	uint32_t magic_number_images, magic_number_labels, l_num_images, l_num_labels, l_num_rows, l_num_cols;

	// Read images file header	
	images_file.read(reinterpret_cast<char*>(&magic_number_images), sizeof(magic_number_images));
	images_file.read(reinterpret_cast<char*>(&l_num_images), sizeof(l_num_images));
	images_file.read(reinterpret_cast<char*>(&l_num_rows), sizeof(l_num_rows));
	images_file.read(reinterpret_cast<char*>(&l_num_cols), sizeof(l_num_cols));

	// Read labels file header
	labels_file.read(reinterpret_cast<char*>(&magic_number_labels), sizeof(magic_number_labels));
	labels_file.read(reinterpret_cast<char*>(&l_num_labels), sizeof(l_num_labels));

	// swap endianness
	magic_number_images = _byteswap_ulong(magic_number_images);
	l_num_images = _byteswap_ulong(l_num_images);
	l_num_rows = _byteswap_ulong(l_num_rows);
	l_num_cols = _byteswap_ulong(l_num_cols);

	magic_number_labels = _byteswap_ulong(magic_number_labels);
	l_num_labels = _byteswap_ulong(l_num_labels);

	// Print info
	std::cout << "Images file header:\n";
	std::cout << "Magic number: " << magic_number_images << "\n";
	std::cout << "Number of images: " << l_num_images << "\n";
	std::cout << "Number of rows: " << l_num_rows << "\n";
	std::cout << "Number of columns: " << l_num_cols << "\n";

	std::cout << "Labels file header:\n";
	std::cout << "Magic number: " << magic_number_labels << "\n";
	std::cout << "Number of labels: " << l_num_labels << "\n";

	if (!l_num_images || !l_num_labels)
	{
		throw std::runtime_error("Invalid file header.");
	}

	// Set number of images and labels
	this->num_images = l_num_images;
	this->num_labels = l_num_labels;
	this->num_rows = l_num_rows;
	this->num_cols = l_num_cols;
	this->output_size = 10;

	// Close files
	images_file.close();
	labels_file.close();
}

void TrainSet::initialize(const size_t batch_size)
{
	if (!this->is_files_good())
	{
		throw std::runtime_error("Files don't exists or couldn't be read.");
	}

	// Open file
	std::ifstream images_file(this->images_file_path.data(), std::ios::in | std::ios::binary);
	std::ifstream labels_file(this->labels_file_path.data(), std::ios::in | std::ios::binary);

	uint32_t temp;
	images_file.read(reinterpret_cast<char*>(&temp), sizeof(temp));
	images_file.read(reinterpret_cast<char*>(&temp), sizeof(temp));
	images_file.read(reinterpret_cast<char*>(&temp), sizeof(temp));
	images_file.read(reinterpret_cast<char*>(&temp), sizeof(temp));
	labels_file.read(reinterpret_cast<char*>(&temp), sizeof(temp));
	labels_file.read(reinterpret_cast<char*>(&temp), sizeof(temp));



	// Initialize matrices
	this->images_in_batches.clear();
	const auto l_num_batches = static_cast<size_t>(std::ceil(static_cast<float>(this->num_images) / static_cast<float>(batch_size)));
	this->num_batches = l_num_batches;
	this->images_in_batches.resize(num_batches);
	this->labels_in_batches.resize(num_batches);

	auto pixel_buffer = std::make_unique<uint8_t[]>(this->num_rows * this->num_cols * batch_size);
	auto label_buffer = std::make_unique<uint8_t[]>(batch_size);

	for (size_t iterations = 0; iterations < this->num_batches; ++iterations)
	{

		// Initialize matrix
		this->images_in_batches[iterations] = std::make_unique<nn::Matrix<float>>(this->num_rows * this->num_cols, batch_size);
		this->labels_in_batches[iterations] = std::make_unique<nn::Matrix<float>>(this->output_size, batch_size);
		this->labels_in_batches[iterations]->perform_element_wise_operation([](const float) -> float { return 0.0f; });

		images_file.read(reinterpret_cast<char*>(pixel_buffer.get()), this->num_rows * this->num_cols * batch_size);
		labels_file.read(reinterpret_cast<char*>(label_buffer.get()), batch_size);

		for (size_t j = 0; j < batch_size; ++j)
		{
			for (size_t i = 0; i < this->num_rows * this->num_cols; ++i)
			{
				this->images_in_batches[iterations]->operator()(i, j) = static_cast<float>(pixel_buffer[i + j * this->num_rows * this->num_cols]) / 255.0f;
			}

			this->labels_in_batches[iterations]->operator()(label_buffer[j], j) = 1.0f;
		}
	}

	// Close files
	if (images_file.is_open())
	{
		images_file.close();
	}
	if (labels_file.is_open())
	{
		labels_file.close();
	}
}

nn::Matrix<float>& TrainSet::get_batch_input()
{
	return *this->images_in_batches[this->current_index_];
}

nn::Matrix<float>& TrainSet::get_batch_output()
{
	return *this->labels_in_batches[this->current_index_];
}

bool TrainSet::is_end() const
{
	return this->current_index_ >= this->num_batches;
}

bool TrainSet::is_ready() const
{
	return true;
}

void TrainSet::reset()
{
	this->current_index_ = 0;
}

size_t TrainSet::get_input_size() const
{
	return this->num_rows * this->num_cols;
}

size_t TrainSet::get_output_size() const
{
	return this->output_size;
}

size_t TrainSet::get_total_size() const
{
	return this->num_images;
}

bool TrainSet::is_files_good() const
{
	// Open file
	const auto in = std::ifstream(this->images_file_path.data(), std::ios::in | std::ios::binary);
	const auto out = std::ifstream(this->labels_file_path.data(), std::ios::in | std::ios::binary);

	// Check if files are open
	if (!in.is_open() || !out.is_open())
	{
		return false;
	}
	// Check if files are good
	if (!in.good() || !out.good())
	{
		return false;
	}

	return true;
}