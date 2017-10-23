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

	long steps;
	{
		std::istringstream ss(argv[2]);
		if (!(ss >> steps))
			std::cerr << "Invalid number " << argv[2] << '\n';
	}

	double delta_energy = std::atof(argv[3]);
	std::string file_name_in(argv[4]);
	std::string file_name_out(argv[5]);

	srand(static_cast<unsigned>(time(NULL)));

	std::vector<Type> types;
	std::vector<std::vector<double>> positions;

	Load::XYZ(types, positions, file_name_in);

	std::unique_ptr<Configuration> sample = std::make_unique<Configuration>(positions, types);

	std::unique_ptr<Core> core = std::make_unique<Core>(*sample, cells, steps, types, delta_energy);
	core->Run();

    std::ofstream file_out("when_which_where_"+file_name_out+".dat", std::ios::binary | std::ios::out);
    for (auto pos : core->when_which_where) {
        file_out << pos[0] << "\t" << pos[1] << "\t" << pos[2] << "\n";
    }
}
