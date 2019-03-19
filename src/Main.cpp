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
	// srand(static_cast<unsigned>(time(NULL)));
	srand(1234);

	std::string data_path(argv[1]);
	std::string line;

	std::ifstream input;
	std::ofstream file_out_param;

	std::vector<std::string> input_vector;
	std::vector<size_t> cells(3,5);
	std::vector<Type> types;
	std::vector<std::vector<double>> positions;
	std::vector<bool> contact_switch(2, false);
	std::vector<unsigned int> contact(2, 100);

	input.open(data_path+"/input.kmc");
	while(!input.eof()){
		getline(input, line);
		input_vector.push_back(line.substr(0, line.find('#', 0)));
	}
	
	{
		std::istringstream ss(input_vector[1]);
		if (!(ss >> cells[0]))
			std::cerr << "Invalid number " << input_vector[1] << '\n';
	}

	{
		std::istringstream ss(input_vector[2]);
		if (!(ss >> cells[1]))
			std::cerr << "Invalid number " << input_vector[2] << '\n';
	}

	{
		std::istringstream ss(input_vector[3]);
		if (!(ss >> cells[2]))
			std::cerr << "Invalid number " << input_vector[3] << '\n';
	}

	long double time_end;
	{
		std::string::size_type sz;
		time_end = std::stold(input_vector[4], &sz);
	}

	switch(std::atoi(input_vector[5].c_str())){
		case 0:
			contact_switch[0] = false;
			break;
		case 2:
			contact_switch[0] = true;
			break;
		default:
			throw("Error in contact switch");
			break;
	}

	switch(std::atoi(input_vector[6].c_str())){
		case 0:
			contact_switch[1] = false;
			break;
		case 2:
			contact_switch[1] = true;
			break;
		default:
			throw("Error in contact switch");
			break;
	}

	{
		std::istringstream ss(input_vector[7]);
		if (!(ss >> contact[0]))
			std::cerr << "Invalid number " << input_vector[7] << '\n';
	}

	{
		std::istringstream ss(input_vector[8]);
		if (!(ss >> contact[1]))
			std::cerr << "Invalid number " << input_vector[8] << '\n';
	}

	double A = std::atof(input_vector[9].c_str());
	double period = std::atof(input_vector[10].c_str());
	double delta_energy_base = std::atof(input_vector[11].c_str());

	Load::XYZ(types, positions, data_path);
	std::unique_ptr<Configuration> sample = std::make_unique<Configuration>(positions, types);

	std::unique_ptr<Core> core = std::make_unique<Core>(*sample, cells, types, contact_switch, contact, data_path);
	core->Run(time_end, A, period, delta_energy_base);

	//Save params/
	/*
	file_out_param.open(data_path+"/params.dat");
	file_out_param << core->steps+1 << std::endl;
	file_out_param << sample->GetOxygenNumber() << std::endl;
	file_out_param.close();
	*/
}

