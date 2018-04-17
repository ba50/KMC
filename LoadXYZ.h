#pragma once

#include <vector>
#include <string>
#include <sstream>
#include <fstream>

#include "Position.h"

namespace Load {
	static void XYZ(std::vector<Type> &types, std::vector<std::vector<double>> &positions, const std::string &file_name) {
		std::ifstream file;
		file.open(file_name);
		double temp_double;
		std::vector<double> temp_vector;
		std::string temp_string;
		
		file >> temp_string;
		std::stringstream iss(temp_string);

		size_t atom_number;
		iss >> atom_number;

		std::getline(file, temp_string);

		for (size_t i = 0; i < atom_number; i++) {
			file >> temp_string;
			if (temp_string == "Bi") {
				types.push_back(Type::Bi);
			}
			if (temp_string == "Y") {
				types.push_back(Type::Y);
			}
			if (temp_string == "O") {
				types.push_back(Type::O);
			}

			temp_vector.clear();

			file >> temp_double;
			temp_vector.push_back(temp_double);
			file >> temp_double;
			temp_vector.push_back(temp_double);
			file >> temp_double;
			temp_vector.push_back(temp_double);

			positions.push_back(temp_vector);
		}
	}
}
