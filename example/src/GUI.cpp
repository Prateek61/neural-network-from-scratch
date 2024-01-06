#include "GUI.h"
#include <iostream>

void GUI::calculate_predicted_number()
{
	this->is_calculating_ = true;
	this->is_finished_ = false;

	// Get the data from the drawing
	const std::array<float, 784> data = this->drawing_->get_data();
	// Convert the data to a matrix
	nn::Matrix<float> data_matrix(784, 1);
	for (size_t i = 0; i < data.size(); ++i)
	{
		data_matrix(i, 0) = data[i];
	}

	// Feed forward the data
	this->neural_network_->feed_forward_with_input(data_matrix);

	// Get the output of the neural network
	const nn::Matrix<float>& output = this->neural_network_->get_output();

	// Find the index of the highest value in the output matrix
	size_t max_index = 0;
	float max_value = output(0, 0);
	for (size_t i = 1; i < output.get_rows(); ++i)
	{
		if (output(i, 0) > max_value)
		{
			max_index = i;
			max_value = output(i, 0);
		}
	}

	std::cout << "Output:\n " << output << std::endl;

	this->predicted_number_ = static_cast<int>(max_index);
	this->is_calculating_ = false;
	this->is_finished_ = true;
}

GUI::GUI()
{
	sAppName = "Neural Network";
}


GUI::~GUI() = default;

bool GUI::OnUserCreate()
{
	// Initialize the drawing array
	this->drawing_ = std::make_unique<Drawing>(olc::vi2d(10, 10));
	// Initialize the predict button
	this->predict_ = std::make_unique<Button>(olc::vi2d(400, 10), olc::vi2d(130, 50), "Predict", olc::GREEN, [&]()
	{
		this->calculate_predicted_number();
		std::cout << "Predicted number: " << this->predicted_number_ << std::endl;
	});
	// Initialize the clear button
	this->clear_ = std::make_unique<Button>(olc::vi2d(400, 70), olc::vi2d(130, 50), "Clear", olc::RED, [&]()
	{
		this->drawing_->clear();
		this->is_finished_ = false;
	});
	this->is_calculating_ = false;
	this->is_finished_ = false;
	this->predicted_number_ = -1;
	this->initialize_neural_network("neural_network.txt");
	return true;
}

bool GUI::OnUserUpdate(float elapsed_time)
{
	// Clear the screen
	Clear(olc::BLACK);

	// Disabled when the neural network is calculating
	if (!this->is_calculating_)
	{
		if (GetMouse(0).bHeld)
		{
			// Update the drawing array 
			this->drawing_->update(olc::vi2d(GetMouseX(), GetMouseY()));
			// Predict and Clear buttons
			this->predict_->update(olc::vi2d(GetMouseX(), GetMouseY()), GetMouse(0).bPressed);
			this->clear_->update(olc::vi2d(GetMouseX(), GetMouseY()), GetMouse(0).bPressed);
		}
	}

	if (this->is_finished_)
	{
		// Draw the predicted number
		DrawString(400, 140, "Prediction: " + std::to_string(this->predicted_number_), olc::WHITE, 1);
	}
	if (this->is_calculating_)
	{
		// Draw the calculating text
		DrawString(400, 140, "Calculating...", olc::WHITE, 1);
	}

	// Draw the drawing array
	this->drawing_->draw(*this);
	// Draw the predict button
	this->predict_->draw(*this);
	// Draw the clear button
	this->clear_->draw(*this);

	return true;
}

void GUI::initialize_neural_network(const char* file_name)
{
	// Initialize the neural network
	this->neural_network_ = std::make_unique<nn::NeuralNetwork>();
	neural_network_->load_from_file(file_name);
	neural_network_->set_batch_size(1);
}

Button::Button(const olc::vi2d& position, const olc::vi2d& size, const std::string_view& text, const olc::Pixel& color,
               const std::function<void()>& callback)
	: position_(position), size_(size), text_(text), color_(color), is_hovered_(false), is_clicked_(false),
	  callback_(callback), text_position_(0, 0), text_scale_(0)
{
}

void Button::update(const olc::vi2d& mouse_position, const bool& is_mouse_clicked)
{
	// Check if mouse is hovered
	if (mouse_position.x >= this->position_.x && mouse_position.x <= this->position_.x + this->size_.x &&
		mouse_position.y >= this->position_.y && mouse_position.y <= this->position_.y + this->size_.y)
	{
		this->is_hovered_ = true;

		// Check if mouse is clicked
		if (is_mouse_clicked)
		{
			is_clicked_ = true;
			this->callback_();
		}
		else
		{
			is_clicked_ = false;
		}
	}
	else
	{
		is_hovered_ = false;
		is_clicked_ = false;
	}
}

void Button::draw(olc::PixelGameEngine& pge)
{
	// Draw button
	if (this->is_hovered_)
	{
		pge.FillRect(this->position_, this->size_, this->color_ * 0.8f);
	}
	else
	{
		pge.FillRect(this->position_, this->size_, this->color_ * 0.5f);
	}

	if (this->text_scale_ == 0)
	{
		// Calculate the proper position for the text to center it
		olc::vi2d text_size = pge.GetTextSizeProp(this->text_);
		this->text_scale_ = 2;
		text_size *= this->text_scale_;
		this->text_position_ = this->position_ + (this->size_ / 2) - (text_size / 2);
	}

	// Draw text
	pge.DrawString(this->text_position_, this->text_, olc::WHITE, this->text_scale_);
}

Drawing::Drawing(const olc::vi2d& offset)
	: offset(offset)
{
	this->clear();
}

void Drawing::draw(olc::PixelGameEngine& pge) const
{
	// Draw a border around the drawing area
	pge.DrawRect(this->offset.x - 1, this->offset.y - 1, 337, 337, olc::WHITE);

	// Draw the drawing array
	for (int x = 0; x < this->drawing.size(); ++x)
	{
		for (int y = 0; y < this->drawing[0].size(); ++y)
		{
			if (this->drawing[x][y] == true)
			{
				pge.Draw(x + this->offset.x, y + this->offset.y, olc::WHITE);
			}
		}
	}
}

void Drawing::update(const olc::vi2d& mouse_position)
{
	// Update the drawing array which is 336 * 336 pixels but offset by this->offset pixels
	if (mouse_position.x >= this->offset.x && mouse_position.x <= this->offset.x + this->drawing.size() &&
		mouse_position.y >= this->offset.y && mouse_position.y <= this->offset.y + this->drawing[0].size())
	{
		// Set the pixels in 10 radius to true
		const int x = mouse_position.x - this->offset.x;
		const int y = mouse_position.y - this->offset.y;

		constexpr int radius = 12;

		const int x_bottom_bound = std::max(0, x - radius);
		const int x_top_bound = std::min(336, x + radius);
		const int y_bottom_bound = std::max(0, y - radius);
		const int y_top_bound = std::min(336, y + radius);

		// Set the pixels in a 10 radius circle to true
		for (int x_ = x_bottom_bound; x_ < x_top_bound; ++x_)
		{
			for (int y_ = y_bottom_bound; y_ < y_top_bound; ++y_)
			{
				if (std::pow(x_ - x, 2) + std::pow(y_ - y, 2) <= radius * radius)
				{
					this->drawing[x_][y_] = true;
				}
			}
		}
	}
}

void Drawing::clear()
{
	for (int x = 0; x < this->drawing.size(); ++x)
	{
		for (int y = 0; y < this->drawing[0].size(); ++y)
		{
			this->drawing[x][y] = false;
		}
	}
}

std::array<float, 784> Drawing::get_data() const
{
	std::array<float, 784> data{};
	// Initialize the data array to zeros
	for (float& i : data)
	{
		i = 0.0f;
	}

	// Of the 336 * 336 pixel array, each 12 * 12 pixel square is a pixel in the data array
	// We need to iterate through the 336 * 336 array and update the corresponding pixel in the data array by adding 1 if the pixel is true
	for (int x = 0; x < this->drawing.size(); ++x)
	{
		for (int y = 0; y < this->drawing[0].size(); ++y)
		{
			// Calculate the position in the data array
			const int data_x = y / 12;
			const int data_y = x / 12;

			// Update the data array
			data[data_x * 28 + data_y] += static_cast<float>(this->drawing[x][y]);
		}
	}

	// Normalize the data array
	for (float& i : data)
	{
		i /= 144.0f;
	}

	return data;
}
