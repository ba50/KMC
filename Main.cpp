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

	std::vector<size_t> cells(3,5);
	{
		std::istringstream ss(argv[1]);
		if (!(ss >> cells[0]))
			std::cerr << "Invalid number " << argv[1] << '\n';
	}
	int cells_y;
	{
		std::istringstream ss(argv[2]);
		if (!(ss >> cells[1]))
			std::cerr << "Invalid number " << argv[2] << '\n';
	}
	int cells_z;
	{
		std::istringstream ss(argv[3]);
		if (!(ss >> cells[2]))
			std::cerr << "Invalid number " << argv[3] << '\n';
	}

	long double time_end;
	{
		std::string::size_type sz;
		time_end = std::stold(argv[4], &sz); 
	}

	double delta_energy = std::atof(argv[5]);
	std::string file_name_in(argv[6]);
	std::string file_name_out(argv[7]);
	srand(static_cast<unsigned>(time(NULL)));

	std::vector<Type> types;
	std::vector<std::vector<double>> positions;

	Load::XYZ(types, positions, file_name_in);
	std::unique_ptr<Configuration> sample = std::make_unique<Configuration>(positions, types);

	std::unique_ptr<Core> core = std::make_unique<Core>(*sample, cells, time_end, types, delta_energy, file_name_out);
	/*
	core->Run();

	std::ofstream file_out_param("param_"+file_name_out+".dat");
	file_out_param << core->steps+1 << std::endl;
	file_out_param << sample->GetOxygenNumber() << std::endl;
	*/
}

