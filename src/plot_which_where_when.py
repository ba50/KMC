import os
from pathlib import Path

import h5py
from tqdm import tqdm
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from TimeHeatMap import TimeHeatMap


class DataProcess:
    options = {'dpi': 100, 'MSD': False, '3D': True, 'Len': False}
    jump = np.array([[1, 0, 0],
                     [-1, 0, 0],
                     [0, 1, 0],
                     [0, -1, 0],
                     [0, 0, 1],
                     [0, 0, -1]])

    def __init__(self, sim: Path, options: dict = None):
        if options:
            self.options = options

        self.data_path = sim / 'which_where_when.txt'
        self.pos_path = sim / 'positions.xyz'
        self.save_path = sim / 'paths'

        print('Save in:', self.save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)

        bi_base_positions = []
        y_base_positions = []
        o_base_positions = []
        with self.pos_path.open('r') as file_in:
            for line in file_in.readlines()[2:]:
                line = line.split('\t')
                if line[0] == 'Bi':
                    bi_base_positions.append([float(line[1]),
                                              float(line[2]),
                                              float(line[3])])
                if line[0] == 'Y':
                    y_base_positions.append([float(line[1]),
                                             float(line[2]),
                                             float(line[3])])
                if line[0] == 'O':
                    o_base_positions.append([float(line[1]),
                                             float(line[2]),
                                             float(line[3])])

        self.bi_base_positions = np.array(bi_base_positions)
        self.y_base_positions = np.array(y_base_positions)
        self.o_base_positions = np.array(o_base_positions)

        file_out = h5py.File((self.save_path/'o_paths.hdf5'), 'w')
        self.o_paths = file_out.create_dataset('o_path',
                                               (self.o_base_positions.shape[0],
                                                self.o_base_positions.shape[1],
                                                1),
                                               maxshape=(self.o_base_positions.shape[0],
                                                         self.o_base_positions.shape[1],
                                                         None),
                                               data=self.o_base_positions)

    def run(self, n_points: int):
        """

        :param n_points: Number of point to plot (Field, Time)
        """
        www = pd.read_csv(
            self.data_path,
            sep='\t',
            names=['which', 'where', 'delta_energy', 'when']
        )

        loc_index = list(range(0, www.shape[0], www.shape[0] // n_points))
        
        print("Calculating oxygen paths")
        # Histogram
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111)
        ax.set_xlabel('Direction')
        ax.set_ylabel('Count')
        ax.hist(www['where'], bins=11)
        plt.savefig(self.save_path / "Jumps.png", dpi=self.options['dpi'])
        
        self.plot_line(save_file=self.save_path / 'Field.png',
                       x=www['when'].iloc[loc_index],
                       y=www['delta_energy'].iloc[loc_index],
                       x_label='Time [ps]',
                       y_label='Field [eV]')
        
        self.plot_line(save_file=self.save_path / 'Time.png',
                       x=range(len(loc_index)),
                       y=www['when'].iloc[loc_index],
                       x_label='Step',
                       y_label='Time')
        
        self.plot_line(save_file=self.save_path / 'delta_Time.png',
                       x=range(len(loc_index)),
                       y=www['when'].iloc[loc_index].diff(),
                       x_label='Step',
                       y_label='Time')
        del www
        timed_heat_map = TimeHeatMap(load_data_path=self.data_path.parent, options=self.options, workers=3)
        timed_heat_map.process_data()

    def plot_line(self, save_file, x, y, x_label, y_label, x_size=8, y_size=6):
        _fig = plt.figure(figsize=(x_size, y_size))
        _ax = _fig.add_subplot(111)
        _ax.set_xlabel(x_label)
        _ax.set_ylabel(y_label)
        _ax.plot(x, y)
        plt.savefig(save_file, dpi=self.options['dpi'], bbox_inches='tight')


























if __name__ == '__main__':
    simulations = [Path(x) for x in [
        'D:/KMC_data/data_2019_09_08/30_7_7_random_0_a',
        'D:/KMC_data/data_2019_09_08/30_7_7_random_1_a',
        'D:/KMC_data/data_2019_09_08/30_7_7_random_2_a',
        'D:/KMC_data/data_2019_09_08/30_7_7_random_3_a',
        'D:/KMC_data/data_2019_09_08/30_7_7_random_4_a',
        'D:/KMC_data/data_2019_09_08/30_7_7_random_5_a',
        'D:/KMC_data/data_2019_09_08/30_7_7_random_6_a',
        'D:/KMC_data/data_2019_09_08/30_7_7_random_7_a',
        'D:/KMC_data/data_2019_09_08/30_7_7_random_0_b',
        'D:/KMC_data/data_2019_09_08/30_7_7_random_1_b',
        'D:/KMC_data/data_2019_09_08/30_7_7_random_2_b',
        'D:/KMC_data/data_2019_09_08/30_7_7_random_3_b',
        'D:/KMC_data/data_2019_09_08/30_7_7_random_4_b',
        'D:/KMC_data/data_2019_09_08/30_7_7_random_5_b',
        'D:/KMC_data/data_2019_09_08/30_7_7_random_6_b',
        'D:/KMC_data/data_2019_09_08/30_7_7_random_7_b',
        'D:/KMC_data/data_2019_09_08/30_7_7_random_0_c',
        'D:/KMC_data/data_2019_09_08/30_7_7_random_1_c',
        'D:/KMC_data/data_2019_09_08/30_7_7_random_2_c',
        'D:/KMC_data/data_2019_09_08/30_7_7_random_3_c',
        'D:/KMC_data/data_2019_09_08/30_7_7_random_4_c',
        'D:/KMC_data/data_2019_09_08/30_7_7_random_5_c',
        'D:/KMC_data/data_2019_09_08/30_7_7_random_6_c',
        'D:/KMC_data/data_2019_09_08/30_7_7_random_7_c',
        ]]

    config = {
        'dpi': 100,
        'MSD': False,
        'Len': False,
        '3D': False,
        'time_step': 100
    }

    for idx in simulations:
        DataProcess(idx, config).run(100)
