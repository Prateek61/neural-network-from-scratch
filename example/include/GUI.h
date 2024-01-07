// File: example/include/GUI.h
// Purpose: Header file for GUI class.

#pragma once

#include "olcPixelGameEngine.h"

#include <memory>
#include <string_view>

#include <NeuralNetwork/NeuralNetwork.h>

class Button;
class Drawing;

class GUI final : public olc::PixelGameEngine
{
private:
	// Neural Network
	std::unique_ptr<nn::NeuralNetwork> neural_network_;
	// Predict Button
	std::unique_ptr<Button> predict_;
	// Button for clearing the drawing
	std::unique_ptr<Button> clear_;
	// 336 * 336 boolean array for drawing
	std::unique_ptr<Drawing> drawing_;
	// Is the neural network calculating
	bool is_calculating_;
	// Has the neural network finished calculating
	bool is_finished_;
	// The predicted number
	 int predicted_number_;

	// Calculate the predicted number
	 void calculate_predicted_number();

public:

	GUI();
	~GUI() override;

	bool OnUserCreate() override;
	bool OnUserUpdate(float elapsed_time) override;

	void initialize_neural_network(const char* file_name);
};

// Button class
class Button
{
private:
	// Position
	olc::vi2d position_;

	// Size
	olc::vi2d size_;

	// Text
	std::string text_;

	// Color
	olc::Pixel color_;

	// Is Hovered
	bool is_hovered_;

	// Is Clicked
	bool is_clicked_;

	// Callback function
	std::function<void()> callback_;

	// Text position
	olc::vi2d text_position_;

	// Text scale
	int text_scale_;
public:
	Button(const olc::vi2d& position, const olc::vi2d& size, const std::string_view& text, const olc::Pixel& color, const std::function<void()>& callback);
	~Button() = default;

	// Update
	void update(const olc::vi2d& mouse_position, const bool& is_mouse_clicked);
	// Draw
	void draw(olc::PixelGameEngine& pge);
};

class Drawing
{
public:
	explicit Drawing(const olc::vi2d& offset);
	~Drawing() = default;

	// Draw
	void draw(olc::PixelGameEngine& pge) const;
	// Update
	void update(const olc::vi2d& mouse_position);
	// Clear
	void clear();
	// Get data ( Image )
	[[nodiscard]] std::array<float, 784> get_data() const;

	// Drawing board ( 28 * 12, 28 * 12 )
	std::array<std::array<bool, 336>, 336> drawing;
	// Offset
	olc::vi2d offset;
};