from pathlib import Path

import h5py
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from TimeHeatMap import TimeHeatMap
from utils.config import get_config


class DataProcess:
    options = {'dpi': 100, 'MSD': False, '3D': True, 'Len': False}
    jump = np.array([[1, 0, 0],
                     [-1, 0, 0],
                     [0, 1, 0],
                     [0, -1, 0],
                     [0, 0, 1],
                     [0, 0, -1]])

    def __init__(self, simulation_path: Path, workers: int = 1, options: dict = None):
        self.workers = workers
        if options:
            self.options = options

        self.simulation_path = simulation_path
        # self.field_path = simulation_path / 'field_plot.csv'
        # self.pos_path = simulation_path / 'positions.xyz'
        # self.save_path = simulation_path / 'paths'

        print('Save in:', self.simulation_path / 'paths')
        (self.simulation_path / 'paths').mkdir(parents=True, exist_ok=True)

        bi_base_positions = []
        y_base_positions = []
        o_base_positions = []
        with (self.simulation_path / 'positions.xyz').open('r') as file_in:
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

        file_out = h5py.File((self.simulation_path / 'paths' / 'o_paths.hdf5'), 'w')
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
        # when_which_where = pd.read_csv(self.simulation_path / 'when_which_where.csv', index_col=False)
        # field_path = pd.read_csv(self.simulation_path / 'field_plot.csv', index_col=False)
        #
        # if len(when_which_where) > 1:
        #
        #     print("Calculating oxygen paths")
        #     # Histogram
        #     fig = plt.figure(figsize=(8, 6))
        #     ax = fig.add_subplot(111)
        #     ax.set_xlabel('Direction')
        #     ax.set_ylabel('Count')
        #     ax.hist(when_which_where['selected_direction'], bins=11)
        #     plt.savefig(self.simulation_path / 'paths' / "Jumps.png", dpi=self.options['dpi'])
        #
        #     # Histogram
        #     fig = plt.figure(figsize=(8, 6))
        #     ax = fig.add_subplot(111)
        #     ax.set_xlabel('Atom')
        #     ax.set_ylabel('Count')
        #     ax.hist(when_which_where['selected_atom'])
        #     plt.savefig(self.simulation_path / 'paths' / "Atom.png", dpi=self.options['dpi'])
        #
        #     self.plot_line(save_file=self.simulation_path / 'paths' / 'Field.png',
        #                    x=field_path['time'],
        #                    y=field_path['delta_energy'],
        #                    x_label='Time [ps]',
        #                    y_label='Field [eV]')
        #
        #     loc_index = list(range(0, when_which_where.shape[0], int(when_which_where.shape[0] // n_points)))
        #
        #     self.plot_line(save_file=self.simulation_path / 'paths' / 'Time.png',
        #                    x=range(len(loc_index)),
        #                    y=when_which_where['time'].iloc[loc_index],
        #                    x_label='Step',
        #                    y_label='Time')
        #
        #     self.plot_line(save_file=self.simulation_path / 'paths' / 'delta_Time.png',
        #                    x=range(len(loc_index)),
        #                    y=when_which_where['time'].iloc[loc_index].diff(),
        #                    x_label='Step',
        #                    y_label='Time')
        #
        #     del when_which_where
        #     del field_path

        timed_heat_map = TimeHeatMap(load_data_path=self.simulation_path, options=self.options, workers=self.workers)
        timed_heat_map.process_data()

        plt.close('all')

    def plot_line(self, save_file, x, y, x_label, y_label, x_size=8, y_size=6):
        _fig = plt.figure(figsize=(x_size, y_size))
        _ax = _fig.add_subplot(111)
        _ax.set_xlabel(x_label)
        _ax.set_ylabel(y_label)
        _ax.plot(x, y)
        plt.savefig(save_file, dpi=self.options['dpi'], bbox_inches='tight')


if __name__ == '__main__':

    _workers = 3
    save_path = Path('D:/KMC_data/data_2019_11_19_v3')
    plot_steps = 100

    sim_path_list = [sim for sim in save_path.glob("*") if sim.is_dir()]

    plot_options = {
        'dpi': 100,
        'MSD': False,
        'Len': False,
        '3D': False,
    }

    for sim_path in sim_path_list:
        sim_config = get_config(sim_path/'input.kmc')
        if list(sim_path.glob('heat_map/*')):
            plot_options['time_step'] = sim_config['window']
            DataProcess(simulation_path=sim_path, workers=_workers, options=plot_options).run(n_points=plot_steps)
