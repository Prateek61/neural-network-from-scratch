#include "GUI.h"
#include <iostream>

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
	this->predict_ = std::make_unique<Button>(olc::vi2d(400, 10), olc::vi2d(130, 50), "Predict", olc::GREEN, []()
	{
		std::cout << "Predict button clicked!" << std::endl;
	});
	// Initialize the clear button
	this->clear_ = std::make_unique<Button>(olc::vi2d(400, 70), olc::vi2d(130, 50), "Clear", olc::RED, [&]()
	{
		this->drawing_->clear();
	});

	return true;
}

bool GUI::OnUserUpdate(float elapsed_time)
{
	// Clear the screen
	Clear(olc::BLACK);

	if (GetMouse(0).bHeld)
	{
		this->drawing_->update(olc::vi2d(GetMouseX(), GetMouseY()));
	}
	this->drawing_->draw(*this);

	// Update and draw the predict button
	this->predict_->update(olc::vi2d(GetMouseX(), GetMouseY()), GetMouse(0).bPressed);
	this->predict_->draw(*this);
	// Update the draw the clear button
	this->clear_->update(olc::vi2d(GetMouseX(), GetMouseY()), GetMouse(0).bPressed);
	this->clear_->draw(*this);

	return true;
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
	pge.DrawRect(this->offset.x - 1, this->offset.y - 1, 341, 341, olc::WHITE);

	// Draw the drawing array
	for (int x = 0; x < 340; ++x)
	{
		for (int y = 0; y < 340; ++y)
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
	// Update the drawing array which is 340 * 340 pixels but offset by this->offset pixels
	if (mouse_position.x >= this->offset.x && mouse_position.x <= this->offset.x + 340 &&
		mouse_position.y >= this->offset.y && mouse_position.y <= this->offset.y + 340)
	{
		// Set the pixels in 10 radius to true
		const int x = mouse_position.x - this->offset.x;
		const int y = mouse_position.y - this->offset.y;

		const int x_bottom_bound = std::max(0, x - 25);
		const int x_top_bound = std::min(340, x + 25);
		const int y_bottom_bound = std::max(0, y - 25);
		const int y_top_bound = std::min(340, y + 25);

		for (int i = x_bottom_bound; i < x_top_bound; ++i)
		{
			for (int j = y_bottom_bound; j < y_top_bound; ++j)
			{
				this->drawing[i][j] = true;
			}
		}
	}
}

void Drawing::clear()
{
	for (int x = 0; x < 340; ++x)
	{
		for (int y = 0; y < 340; ++y)
		{
			this->drawing[x][y] = false;
		}
	}
}
