#pragma once

#include <iostream>
#include <chrono>
#include <ctime> 
#include <map>
#include <algorithm>
#include <fstream>
#include <cmath>
#include <numeric>

#include "Configuration.h"


class Core {
	std::vector<Type> types;

	double*** oxygen_array_;
	std::vector<size_t> oxygen_array_size_; 
	std::vector<std::vector<int>> oxygen_positions_;
	std::vector<std::vector<int>> oxygen_positions_inf_;

	double*** kation_array_;
	std::vector<size_t> kation_array_size_; 

	double*** residence_time_array_;
	std::vector<size_t> residence_time_array_size_;
	std::vector<double> residence_time_;

	std::vector<std::vector<double>> jump_rate_vector_;
	std::vector<double> jumpe_rate_sume_vector_;

	std::vector<std::vector<int>> direction_vector;
	std::vector<double> jumpe_direction_sume_vector_;
	const double temperature_scale;
	const std::string data_path;

	std::vector<double> v_total;
	std::vector<double> v_elec;
	std::vector<double> v_sc;
	std::vector<double> e_field;

public:
	long double steps;

	Core(
		const Configuration& configuration,
			const std::vector<size_t> cells,
			const std::vector<Type> types,
			const std::vector<bool> contact_switch,
			const std::vector<unsigned int> contact,
			const double& temperature_scale,
			const std::string& data_path):
		types( types ), 
		temperature_scale( temperature_scale ),
		data_path( data_path ),
		steps( 0 )
	{
		// Define OXYGENE
		// with bourdery conditions
		
		oxygen_array_size_ = std::vector<size_t>(3, 5);
		kation_array_size_ = std::vector<size_t>(3, 5);
		residence_time_array_size_ = std::vector<size_t>(3, 5);

		oxygen_array_size_[0] = 2 * cells[0] + 2;
		oxygen_array_size_[1] = 2 * cells[1] + 2;
		oxygen_array_size_[2] = 2 * cells[2] + 2;


		oxygen_array_ = new double**[oxygen_array_size_[2]];
		for (size_t z = 0; z < oxygen_array_size_[2]; z++)
			oxygen_array_[z] = new double*[oxygen_array_size_[1]];
		for (size_t z = 0; z < oxygen_array_size_[2]; z++)
			for (size_t y = 0; y < oxygen_array_size_[1]; y++)
				oxygen_array_[z][y] = new double[oxygen_array_size_[0]];

		for (size_t z = 0; z < oxygen_array_size_[2]; z++)
			for (size_t y = 0; y < oxygen_array_size_[1]; y++)
				for (size_t x = 0; x < oxygen_array_size_[0]; x++)
					oxygen_array_[z][y][x] = 1.0;
		
		{
			std::vector<double> temp_position(3);
			std::vector<int> temp_site(3);
			for (auto& sort_map : configuration.GetSortMap()) {
				if (sort_map.first == Type::O) {
					for (auto &position : sort_map.second) {
						temp_position = position.GetPosition();

						//+1 because bourdery conditions
						temp_site[0] = static_cast<int>(floor(temp_position[0]/CELL_SIZE)) + 1;
						temp_site[1] = static_cast<int>(floor(temp_position[1]/CELL_SIZE)) + 1;
						temp_site[2] = static_cast<int>(floor(temp_position[2]/CELL_SIZE)) + 1;
						oxygen_positions_.push_back(temp_site);
						oxygen_positions_inf_.push_back(temp_site);
						oxygen_array_[temp_site[2]][temp_site[1]][temp_site[0]] = 0.0;
					}
				}
			}
		}

		std::vector<std::vector<double>> temp_vector;
		for (size_t i = 0; i < configuration.GetKationNumber(); i++) {
			temp_vector.push_back(std::vector<double>(configuration.GetPositions()[i].GetPosition()));
		}

		for (size_t i = configuration.GetKationNumber(); i < configuration.GetPositions().size(); i++) {
			temp_vector.push_back(std::vector<double>(3));
		}

		// Define KATION
		kation_array_size_[0] = 2 * cells[0] + 1;
		kation_array_size_[1] = 2 * cells[1] + 1;
		kation_array_size_[2] = 2 * cells[2] + 1;

		kation_array_ = new double**[kation_array_size_[2]];
		for (size_t z = 0; z < kation_array_size_[2]; z++)
			kation_array_[z] = new double*[kation_array_size_[1]];
		for (size_t z = 0; z < kation_array_size_[2]; z++)
			for (size_t y = 0; y < kation_array_size_[1]; y++)
				kation_array_[z][y] = new double[kation_array_size_[0]];

		for (size_t z = 0; z < kation_array_size_[2]; z++)
			for (size_t y = 0; y < kation_array_size_[1]; y++)
				for (size_t x = 0; x < kation_array_size_[0]; x++)
					kation_array_[z][y][x] = 0.0;

		{
			std::vector<double> temp_position(3);
			std::vector<size_t> temp_site(3);
			for (auto& sort_map : configuration.GetSortMap()) {
				if (sort_map.first == Type::Bi) {
					for (auto &position : sort_map.second) {
						temp_position = position.GetPosition();
						temp_site[0] = static_cast<size_t>(floor(temp_position[0]/CELL_SIZE));
						temp_site[1] = static_cast<size_t>(floor(temp_position[1]/CELL_SIZE));
						temp_site[2] = static_cast<size_t>(floor(temp_position[2]/CELL_SIZE));
						kation_array_[temp_site[2]][temp_site[1]][temp_site[0]] = 1;
					}
				}
				if (sort_map.first == Type::Y) {
					for (auto& position : sort_map.second) {
						temp_position = position.GetPosition();
						temp_site[0] = static_cast<size_t>(floor(temp_position[0]/CELL_SIZE));
						temp_site[1] = static_cast<size_t>(floor(temp_position[1]/CELL_SIZE));
						temp_site[2] = static_cast<size_t>(floor(temp_position[2]/CELL_SIZE));
						kation_array_[temp_site[2]][temp_site[1]][temp_site[0]] = 2;
					}
				}
			}
		}

		// Define RESIDENCE TIME
		residence_time_array_size_[0]= 2 * cells[0];
		residence_time_array_size_[1]= 2 * cells[1];
		residence_time_array_size_[2]= 2 * cells[2];

		residence_time_array_ = new double**[residence_time_array_size_[2]];
		for (size_t z = 0; z < residence_time_array_size_[2]; z++)
			residence_time_array_[z] = new double*[residence_time_array_size_[1]];
		for (size_t z = 0; z < residence_time_array_size_[2]; z++)
			for (size_t y = 0; y < residence_time_array_size_[1]; y++)
				residence_time_array_[z][y] = new double[residence_time_array_size_[0]];

		auto oxygen_environment = [](const size_t z, const size_t y, const size_t x, double*** &kation_array) {
			return static_cast<int>(kation_array[z][y][x] + kation_array[z][y][x+1] + kation_array[z][y+1][x] + kation_array[z][y+1][x+1]
				+ kation_array[z+1][y][x] + kation_array[z+1][y][x+1] + kation_array[z+1][y+1][x] + kation_array[z+1][y+1][x+1] - 4.0);
		};

		residence_time_ = { 9.0, 16.0, 22.0, 33.0, 48.0 };
		for (size_t i = 0; i < residence_time_.size(); ++i) {
			residence_time_[i] /= temperature_scale;
		}

		for (size_t z = 0; z < residence_time_array_size_[2]; z++) {
			for (size_t y = 0; y < residence_time_array_size_[1]; y++) {
				for (size_t x = 0; x < residence_time_array_size_[0]; x++) {
					switch (oxygen_environment(z, y, x, kation_array_))
					{
						case 0:
							residence_time_array_[z][y][x] = residence_time_[0];
							break;
						case 1:
							residence_time_array_[z][y][x] = residence_time_[1];
							break;
						case 2:
							residence_time_array_[z][y][x] = residence_time_[2];
							break;
						case 3:
							residence_time_array_[z][y][x] = residence_time_[3];
							break;
						case 4:
							residence_time_array_[z][y][x] = residence_time_[4];
							break;
						default:
							std::cout << oxygen_environment(z, y, x, kation_array_) << "\n";
							throw("Error in Residencec time value");
							break;
					}
				}
			}
		}

		if(contact_switch[0]){
			for (size_t z = 0; z < residence_time_array_size_[2]; z++) {
				for (size_t y = 0; y < residence_time_array_size_[1]; y++) {
					residence_time_array_[z][y][0] = contact[0];
				}
			}
		}

		if(contact_switch[1]){
			for (size_t z = 0; z < residence_time_array_size_[2]; z++) {
				for (size_t y = 0; y < residence_time_array_size_[1]; y++) {
					residence_time_array_[z][y][residence_time_array_size_[0]-1] = contact[1];
				}
			}
		}

		for (size_t i = 0; i < configuration.GetOxygenNumber(); i++)
			jump_rate_vector_.push_back(std::vector<double>(6));

		for (size_t i = 0; i < configuration.GetOxygenNumber()+1; i++)
			jumpe_rate_sume_vector_.push_back(0.0);

		direction_vector.push_back(std::vector<int>{ 1, 0, 0});
		direction_vector.push_back(std::vector<int>{-1, 0, 0});
		direction_vector.push_back(std::vector<int>{ 0, 1, 0});
		direction_vector.push_back(std::vector<int>{ 0,-1, 0});
		direction_vector.push_back(std::vector<int>{ 0, 0, 1});
		direction_vector.push_back(std::vector<int>{ 0, 0,-1});

		for (size_t i = 0; i < direction_vector.size()+1; i++) 
			jumpe_direction_sume_vector_.push_back(0.0);
	}

	~Core() {

		for (size_t z = 0; z < oxygen_array_size_[2]; z++) {
			for (size_t y = 0; y < oxygen_array_size_[1]; y++)
				delete[] oxygen_array_[z][y];

			delete[] oxygen_array_[z];
		}
		delete[] oxygen_array_;

		for (size_t z = 0; z < kation_array_size_[2]; z++) {
			for (size_t y = 0; y < kation_array_size_[1]; y++)
				delete[] kation_array_[z][y];

			delete[] kation_array_[z];
		}
		delete[] kation_array_;

		for (size_t z = 0; z < residence_time_array_size_[2]; z++) {
			for (size_t y = 0; y < residence_time_array_size_[1]; y++)
				delete[] residence_time_array_[z][y];

			delete[] residence_time_array_[z];
		}
		delete[] residence_time_array_;

	}

	// Da sie to lepiej rozwiazac!!
	void Run(
		long double thermalization_time,
		long double time_start,
		long double time_end,
		const long double window,
		const long double window_epsilon,
		const double Amp,
		const double frequency,
		const double period,
		const double static_potential
	){
		if( remove(std::string(data_path + "time.log").c_str()) != 0 )
			std::cout<<"Error deleting file: time.log"<<std::endl;
		else
			std::cout<< "File time.log successfully deleted"<<std::endl;

		std::cout << "Starting thermalization\n";
		auto start_timer = std::chrono::system_clock::now();
		Thermalization(thermalization_time);

		std::cout << "Starting simulation" << "\n";
		SimulationLoop(
			time_start,
			time_end,
			window,
			window_epsilon,
			Amp,
			frequency,
			static_potential
		);
		std::cout << data_path << ": Core exit.\n";
		auto end_timer = std::chrono::system_clock::now();

		std::chrono::duration<double> elapsed_time = end_timer-start_timer;

		std::time_t end_time = std::chrono::system_clock::to_time_t(end_timer);

		std::ofstream time_log(data_path + "/time.log");
		time_log << "Finished simulation at:" << std::ctime(&end_time)
			     << "elapsed time: " << elapsed_time.count() << "[s]\n";
	}

	void Thermalization(long double time_end){
		double random_for_atom, random_for_direction;
		double random_for_time;

		std::vector<double>::iterator selected_atom_temp, selected_direction_temp;
		size_t selected_atom, seleced_direction;

		size_t id, i;
		long double time{ 0.0 };

		while(time < time_end){
			BourderyConditions(oxygen_array_, oxygen_array_size_);

			for (id = 0; id < jump_rate_vector_.size(); id++) {
				jump_rate_vector_[id][0] = jump_rate(id, 0, 0, 1, oxygen_array_, oxygen_positions_, residence_time_array_);
				jump_rate_vector_[id][1] = jump_rate(id, 0, 0, -1, oxygen_array_, oxygen_positions_, residence_time_array_);
				jump_rate_vector_[id][2] = jump_rate(id, 0, 1, 0, oxygen_array_, oxygen_positions_, residence_time_array_);
				jump_rate_vector_[id][3] = jump_rate(id, 0, -1, 0, oxygen_array_, oxygen_positions_, residence_time_array_);
				jump_rate_vector_[id][4] = jump_rate(id, 1, 0, 0, oxygen_array_, oxygen_positions_, residence_time_array_);
				jump_rate_vector_[id][5] = jump_rate(id, -1, 0, 0, oxygen_array_, oxygen_positions_, residence_time_array_);

				//zamienic na for_each!!!
				jumpe_rate_sume_vector_[id + 1] = jumpe_rate_sume_vector_[id];
				for (i = 0; i < jump_rate_vector_[0].size(); i++){
					jumpe_rate_sume_vector_[id + 1] += jump_rate_vector_[id][i];
				}
			}

			random_for_atom = std::min(static_cast<double>(rand()) / RAND_MAX + 1.7E-308, 1.0) * jumpe_rate_sume_vector_.back();
			selected_atom_temp = std::lower_bound(jumpe_rate_sume_vector_.begin(), jumpe_rate_sume_vector_.end(), random_for_atom);
			selected_atom = selected_atom_temp - jumpe_rate_sume_vector_.begin() - 1;

			for (id = 1; id < jumpe_direction_sume_vector_.size(); id++) {
				jumpe_direction_sume_vector_[id] = jumpe_direction_sume_vector_[id-1] + jump_rate_vector_[selected_atom][id-1];
			}

			random_for_direction = std::min(static_cast<double>(rand()) / RAND_MAX + 1.7E-308, 1.0) * jumpe_direction_sume_vector_.back();
			selected_direction_temp = std::lower_bound(jumpe_direction_sume_vector_.begin(), jumpe_direction_sume_vector_.end(), random_for_direction);
			seleced_direction = selected_direction_temp - jumpe_direction_sume_vector_.begin() - 1;

			oxygen_array_[oxygen_positions_[selected_atom][2]][oxygen_positions_[selected_atom][1]][oxygen_positions_[selected_atom][0]] = 1;

			oxygen_positions_[selected_atom][2] += direction_vector[seleced_direction][2];
			oxygen_positions_[selected_atom][1] += direction_vector[seleced_direction][1];
			oxygen_positions_[selected_atom][0] += direction_vector[seleced_direction][0];

			// bardzo slaba optymalizacja, wymyslec cos innego
			///////////////////////////////////////////////////////////////////////////
			if (oxygen_positions_[selected_atom][2] == oxygen_array_size_[2] - 1) {
				oxygen_positions_[selected_atom][2] = 1;
			}
			if (oxygen_positions_[selected_atom][1] == oxygen_array_size_[1] - 1) {
				oxygen_positions_[selected_atom][1] = 1;
			}
			if (oxygen_positions_[selected_atom][0] == oxygen_array_size_[0] - 1) {
				oxygen_positions_[selected_atom][0] = 1;
			}

			if (oxygen_positions_[selected_atom][2] == 0) {
				oxygen_positions_[selected_atom][2] = static_cast<int>(oxygen_array_size_[2] - 2);
			}
			if (oxygen_positions_[selected_atom][1] == 0) {
				oxygen_positions_[selected_atom][1] = static_cast<int>(oxygen_array_size_[1] - 2);
			}
			if (oxygen_positions_[selected_atom][0] == 0) {
				oxygen_positions_[selected_atom][0] = static_cast<int>(oxygen_array_size_[0] - 2);
			}
			///////////////////////////////////////////////////////////////////////////

			oxygen_array_[oxygen_positions_[selected_atom][2]][oxygen_positions_[selected_atom][1]][oxygen_positions_[selected_atom][0]] = 0;
			
			random_for_time = std::min(static_cast<double>(rand()) / RAND_MAX + 1.7E-308, 1.0);
			time += (1.0 / jumpe_rate_sume_vector_.back())*log(1.0 / random_for_time);
		}
	}

	void SimulationLoop(
		long double time_start,
		long double time_end,
		const long double window,
		const long double window_epsilon,
		const double Amp,
		const double frequency,
		const double static_potential
	) {
		const double a = 5.559e-10; // at 1077K in [m]
		const double PI = 3.141592653589793238463;
		const double kT{ (800.0 * temperature_scale + 273.15) * 8.6173304e-5 };

		double random_for_atom, random_for_direction;
		double random_for_time;

		std::vector<double>::iterator selected_atom_temp, selected_direction_temp;
		size_t selected_atom, seleced_direction;

		size_t id, i;
		long double time{ time_start }, record_delta{ 0.0 }, d_t{ 0.0 };
		double v_shift{ 0.0 };

		if( remove(std::string(data_path + "/potentials.csv").c_str()) != 0 )
			std::cout<<"Error deleting file: potentials.csv"<<std::endl;
		else
			std::cout<< "File potentials.csv successfully deleted"<<std::endl;

		if( remove(std::string(data_path + "/simulation_frames.xyz").c_str()) != 0 )
			std::cout<<"Error deleting file: simulation_frames.xyz"<<std::endl;
		else
			std::cout<< "File simulation_frames.xyz successfully deleted"<<std::endl;

		if( remove(std::string(data_path + "/simulation_frames_inf.xyz").c_str()) != 0 )
			std::cout<<"Error deleting file: simulation_frames_inf.xyz"<<std::endl;
		else
			std::cout<< "File simulation_frames_inf.xyz successfully deleted"<<std::endl;

		std::ofstream f_out_oxygen_map(data_path + "/simulation_frames.xyz");
		std::ofstream f_out_oxygen_map_inf(data_path + "/simulation_frames_inf.xyz");
		std::ofstream potentials(data_path + "/potentials.csv"); 
		potentials << "time,v_shift\n";

		while(time < time_end){
			BourderyConditions(oxygen_array_, oxygen_array_size_);

			v_shift = Amp * sin(2 * PI * frequency * pow(10.0, -12) * time) + static_potential;

			for (id = 0; id < jump_rate_vector_.size(); id++) {
				jump_rate_vector_[id][0] = jump_rate(id, 0, 0, 1, oxygen_array_, oxygen_positions_, residence_time_array_) * exp(v_shift / kT);

				jump_rate_vector_[id][1] = jump_rate(id, 0, 0, -1, oxygen_array_, oxygen_positions_, residence_time_array_) * exp(-v_shift / kT);

				jump_rate_vector_[id][2] = jump_rate(id, 0, 1, 0, oxygen_array_, oxygen_positions_, residence_time_array_);
				jump_rate_vector_[id][3] = jump_rate(id, 0, -1, 0, oxygen_array_, oxygen_positions_, residence_time_array_);
				jump_rate_vector_[id][4] = jump_rate(id, 1, 0, 0, oxygen_array_, oxygen_positions_, residence_time_array_);
				jump_rate_vector_[id][5] = jump_rate(id, -1, 0, 0, oxygen_array_, oxygen_positions_, residence_time_array_);

				//zamienic na for_each!!!
				jumpe_rate_sume_vector_[id + 1] = jumpe_rate_sume_vector_[id];
				for (i = 0; i < jump_rate_vector_[0].size(); i++){
					jumpe_rate_sume_vector_[id + 1] += jump_rate_vector_[id][i];
				}
			}
			random_for_atom = std::min(static_cast<double>(rand()) / RAND_MAX + 1.7E-308, 1.0) * jumpe_rate_sume_vector_.back();
			selected_atom_temp = std::lower_bound(jumpe_rate_sume_vector_.begin(), jumpe_rate_sume_vector_.end(), random_for_atom);
			selected_atom = selected_atom_temp - jumpe_rate_sume_vector_.begin() - 1;

			for (id = 1; id < jumpe_direction_sume_vector_.size(); id++) {
				jumpe_direction_sume_vector_[id] = jumpe_direction_sume_vector_[id-1] + jump_rate_vector_[selected_atom][id-1];
			}

			random_for_direction = std::min(static_cast<double>(rand()) / RAND_MAX + 1.7E-308, 1.0) * jumpe_direction_sume_vector_.back();
			selected_direction_temp = std::lower_bound(jumpe_direction_sume_vector_.begin(), jumpe_direction_sume_vector_.end(), random_for_direction);
			seleced_direction = selected_direction_temp - jumpe_direction_sume_vector_.begin() - 1;

			oxygen_array_[oxygen_positions_[selected_atom][2]][oxygen_positions_[selected_atom][1]][oxygen_positions_[selected_atom][0]] = 1.0;

			oxygen_positions_[selected_atom][2] += direction_vector[seleced_direction][2];
			oxygen_positions_[selected_atom][1] += direction_vector[seleced_direction][1];
			oxygen_positions_[selected_atom][0] += direction_vector[seleced_direction][0];

			oxygen_positions_inf_[selected_atom][2] += direction_vector[seleced_direction][2];
			oxygen_positions_inf_[selected_atom][1] += direction_vector[seleced_direction][1];
			oxygen_positions_inf_[selected_atom][0] += direction_vector[seleced_direction][0];

			// bardzo slaba optymalizacja, wymyslec cos innego
			///////////////////////////////////////////////////////////////////////////
			if (oxygen_positions_[selected_atom][2] == oxygen_array_size_[2] - 1) {
				oxygen_positions_[selected_atom][2] = 1;
			}
			if (oxygen_positions_[selected_atom][1] == oxygen_array_size_[1] - 1) {
				oxygen_positions_[selected_atom][1] = 1;
			}
			if (oxygen_positions_[selected_atom][0] == oxygen_array_size_[0] - 1) {
				oxygen_positions_[selected_atom][0] = 1;
			}

			if (oxygen_positions_[selected_atom][2] == 0) {
				oxygen_positions_[selected_atom][2] = static_cast<int>(oxygen_array_size_[2] - 2);
			}
			if (oxygen_positions_[selected_atom][1] == 0) {
				oxygen_positions_[selected_atom][1] = static_cast<int>(oxygen_array_size_[1] - 2);
			}
			if (oxygen_positions_[selected_atom][0] == 0) {
				oxygen_positions_[selected_atom][0] = static_cast<int>(oxygen_array_size_[0] - 2);
			}
			///////////////////////////////////////////////////////////////////////////
			oxygen_array_[oxygen_positions_[selected_atom][2]][oxygen_positions_[selected_atom][1]][oxygen_positions_[selected_atom][0]] = 0.0;

			if (abs(record_delta - window) < window_epsilon || time == 0) {
				std::cout << data_path <<": " << time << "/" << time_end << "[ps]" << std::endl;

				f_out_oxygen_map << oxygen_positions_.size() << "\n\n";
				for (size_t x = 0; x < oxygen_positions_.size(); x++) {
					f_out_oxygen_map << "O\t" << oxygen_positions_[x][0] << "\t" \
											<< oxygen_positions_[x][1] << "\t" \
											<< oxygen_positions_[x][2] << "\n";
				}

				f_out_oxygen_map_inf << oxygen_positions_.size() << "\n\n";
				for (size_t x = 0; x < oxygen_positions_.size(); x++) {
					f_out_oxygen_map_inf << "O\t" << oxygen_positions_inf_[x][0] << "\t" \
											<< oxygen_positions_inf_[x][1] << "\t" \
											<< oxygen_positions_inf_[x][2] << "\n";
				}

				potentials << time << "," << v_shift << "\n";

				record_delta = 0.0;
			}
			
			random_for_time = std::min(static_cast<double>(rand()) / RAND_MAX + 1.7E-308, 1.0);
			d_t = (1.0 / jumpe_rate_sume_vector_.back())*log(1.0 / random_for_time);
			time += d_t;
			record_delta += d_t;
		}

		f_out_oxygen_map.close();
		f_out_oxygen_map_inf.close();
		potentials.close();
		std::cout << std::endl;
	}

	inline void BourderyConditions(double*** &array, const std::vector<size_t> &array_size) {
		size_t x, y, z;
		//// back -> front
		for (y = 0; y < array_size[1]; y++) {
			for (x = 0; x < array_size[0]; x++) {
				array[0][y][x] = array[array_size[2] - 2][y][x];
			}
		}

		// front -> back
		for (y = 0; y < array_size[1]; y++) {
			for (x = 0; x < array_size[0]; x++) {
				array[array_size[2]-1][y][x] = array[1][y][x];
			}
		}

		// right -> left
		for (z = 0; z < array_size[2]; z++) {
			for (y = 0; y < array_size[1]; y++) {
				array[z][y][0] = array[z][y][array_size[0] - 2];
			}
		}
		// left -> right
		for (z = 0; z < array_size[2]; z++) {
			for (y = 0; y < array_size[1]; y++) {
				array[z][y][array_size[0] - 1] = array[z][y][1];
			}
		}
		// down -> up
		for (z = 0; z < array_size[2]; z++) {
			for (x = 0; x < array_size[0]; x++) {
				array[z][0][x] = array[z][array_size[1] - 2][x];
			}
		}
		// up -> down
		for (z = 0; z < array_size[2]; z++) {
			for (x = 0; x < array_size[0]; x++) {
				array[z][array_size[1] - 1][x] = array[z][1][x];
			}
		}
	}
	
	inline double jump_rate(const size_t id, const int shift_z, const int shift_y, const int shift_x, double*** &oxygen_array, std::vector<std::vector<int>> &oxygen_positions, double*** &residence_time_array) {
		return oxygen_array[oxygen_positions[id][2] + shift_z][oxygen_positions[id][1] + shift_y][oxygen_positions[id][0] + shift_x]
			/ residence_time_array[oxygen_positions[id][2] - 1][oxygen_positions[id][1] - 1][oxygen_positions[id][0] - 1];
	}
};
