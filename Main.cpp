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

	std::unique_ptr<Core> core = std::make_unique<Core>(*sample, cells, time_end, types, delta_energy, file_name_out);
	core->Run();
	
	std::ofstream file_out_heat_map("heat_map_"+file_name_out+".dat");
	for (size_t z = 1; z < core->heat_map_array_size_-1; z++)
		for (size_t y = 1; y < core->heat_map_array_size_-1; y++)
			for (size_t x = 1; x < core->heat_map_array_size_-1; x++)
				file_out_heat_map << x << " " << y << " " << z << " " << core->heat_map_array_[z][y][x] << std::endl;
	file_out_heat_map.close();

	std::ofstream file_out_param("param_"+file_name_out+".dat");
	file_out_param << core->steps+1 << std::endl;
	file_out_param << sample->GetOxygenNumber() << std::endl;
}

