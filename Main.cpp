#include <memory>
#include <fstream>
#include <vector>
#include <time.h>
#include <cmath>
#include <sstream>

#include "LoadXYZ.h"
#include "Configuration.h"
#include "Core.h"

int main(int argc, char *argv[]) {

	int cells;
	{
		std::istringstream ss(argv[1]);
		if (!(ss >> cells))
			std::cerr << "Invalid number " << argv[1] << '\n';
	}

	long double time_end;
	{
		std::string::size_type sz;
		time_end = std::stold(argv[2], &sz); 
	}

	double delta_energy = std::atof(argv[3]);
	std::string file_name_in(argv[4]);
	std::string file_name_out(argv[5]);
	srand(static_cast<unsigned>(time(NULL)));

	std::vector<Type> types;
	std::vector<std::vector<double>> positions;

	Load::XYZ(types, positions, file_name_in);

	std::unique_ptr<Configuration> sample = std::make_unique<Configuration>(positions, types);

	std::unique_ptr<Core> core = std::make_unique<Core>(*sample, cells, time_end, types, delta_energy);
	core->Run();

	std::ofstream file_out("when_which_where_"+file_name_out+".dat", std::ios::binary | std::ios::out);
	for (auto pos : core->when_which_where) {
		file_out << pos[0] << "\t" << pos[1] << "\t" << pos[2] << "\n";
	}
	
	std::ofstream file_out_heat_map("heat_map_"+file_name_out+".dat", std::ios::binary | std::ios::out);
	for (size_t z = 0; z < core->heat_map_array_size_; z++){
		for (size_t y = 0; y < core->heat_map_array_size_; y++){
			for (size_t x = 0; x < core->heat_map_array_size_; x++){
				file_out_heat_map << core->heat_map_array_[z][y][x] << "\t"; 
			}
			file_out_heat_map << std::endl;
		}
		file_out_heat_map << std::endl << std::endl;
	}
}

