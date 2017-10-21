#pragma once

#include <iostream>
#include <stdio.h>
#include <vector>

enum class Type { Bi, Y, O, Non };

class Position
{
	double x_;
	double y_;
	double z_;

	Type type_;

public:
	Position() :
		x_(0.0),
		y_(0.0),
		z_(0.0),
		type_(Type::Non)
	{};

	Position(const double x, const double y, const double z, const Type type) :
		x_(x),
		y_(y),
		z_(z),
		type_(type)
	{};

	std::vector<double> Data() const {
		std::vector<double> temp(3);
		temp[0] = x_;
		temp[1] = y_;
		temp[2] = z_;
		return temp;
	}

	Type GetType() const {
		return type_;
	}

	friend std::ostream& operator<<(std::ostream& os, const Position& position);
};

std::ostream& operator<<(std::ostream& os, const Position& position) {
	switch (position.GetType())
	{
		case Type::Bi:
			os << "Bi";
			break;
		case Type::Y:
			os << "Y";
			break;
		case Type::O:
			os << "O";
			break;
		case Type::Non:
			os << "Non";
			break;
	default:
			os << "Non";
		break;
	}
	os << "\t";
	for (auto position : position.Data()) {
		os << position << "\t";
	}
	os << "\n";
	return os;
};
