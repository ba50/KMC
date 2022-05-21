#pragma once

#include <vector>
#include <map>

#include "Position.h"

// TODO: need to fix this, heh
#define CELL_SIZE 1.0

class Configuration
{
	std::vector<Position> positions_;
	std::map<Type, float> stoichiometry_;
	std::map<Type, std::vector<Position>> sort_map_;
	size_t oxygen_number_;
	size_t kation_number_;

public:

	Configuration() {};

	Configuration(const std::vector<std::vector<double>> & positions, const std::vector<Type> & types) {

		Position temp_position;
		for (size_t i = 0; i < positions.size(); i++) {
			temp_position = Position(positions[i][0], positions[i][1], positions[i][2], types[i]);
			positions_.push_back(temp_position);
			sort_map_[types[i]].push_back(temp_position);
			stoichiometry_[types[i]]++;
		}

		oxygen_number_ = static_cast<size_t>(stoichiometry_[Type::O]);
		kation_number_ = static_cast<size_t>(stoichiometry_[Type::Bi] + stoichiometry_[Type::Y]);
		float Y_temp = stoichiometry_[Type::Y];
		for (auto& stoich : stoichiometry_) {
			stoich.second /= Y_temp;
		}
		//because
		stoichiometry_[Type::O] *= 2.f;
	};

	void Stoichiometry() {
		for (auto sto : stoichiometry_) {
			switch (sto.first)
			{
			case Type::Bi:
				std::cout << "Bi(" << sto.second << ")";
				break;
			case Type::Y:
				std::cout << "Y(" << sto.second << ")";
				break;
			case Type::O:
				std::cout << "O(" << sto.second << ")";
				break;
            default:
                throw("No case in Configuration");
                break;
			}
		}
	}

	std::vector<Position> GetPositions() const {
		return positions_;
	};

	size_t GetOxygenNumber() const {
		return oxygen_number_;
	};

	size_t GetKationNumber() const {
		return kation_number_;
	};

	std::map<Type, std::vector<Position>> GetSortMap() const {
		return sort_map_;
	};

	friend std::ostream& operator<<(std::ostream& os, Configuration& configuration);
};

std::ostream& operator<<(std::ostream& os, Configuration& configuration) {

	for (auto position : configuration.positions_) {
		os << position;
	}
	return os;
}
