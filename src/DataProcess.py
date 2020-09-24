import numpy as np
from pathlib import Path

import h5py
import pandas as pd
from tqdm import tqdm

# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from GenerateXYZ import GenerateXYZ
from TimeHeatMap import TimeHeatMap
from TimeOxygenMap import TimeOxygenMap


class DataProcess:
    options = {'dpi': 100, 'MSD': False, '3D': True, 'Len': False}
    jump = np.array([[1, 0, 0],
                     [-1, 0, 0],
                     [0, 1, 0],
                     [0, -1, 0],
                     [0, 0, 1],
                     [0, 0, -1]])

    def __init__(self, simulation_path: Path, workers: int = 1, options: dict = None):
        self.simulation_path = simulation_path
        self.workers = workers
        if options:
            self.options = options

        print('Save in:', self.simulation_path / 'paths')
        (self.simulation_path / 'paths').mkdir(parents=True)  # TODO: disable exist_ok

        self.bi_base_positions, self.y_base_positions, self.o_base_positions = \
            GenerateXYZ.read_file(self.simulation_path / 'positions.xyz')

        # TIMExATOMSxPOS
        file_out = h5py.File((self.simulation_path / 'paths' / 'o_paths.hdf5'), 'w')
        self.o_paths = file_out.create_dataset('o_paths',
                                               (1,
                                                self.o_base_positions.shape[0],
                                                self.o_base_positions.shape[1]),
                                               maxshape=(None,
                                                         self.o_base_positions.shape[0],
                                                         self.o_base_positions.shape[1]),
                                               data=self.o_base_positions,
                                               dtype=np.float32)

    def run(self, n_points: int):
        """

        :param n_points: Number of point to plot (Field, Time)
        """

        when_which_where = pd.read_csv(
            self.simulation_path / 'when_which_where.csv',
            index_col=False,
            memory_map=True,
            nrows=10**5
        )
        for step in tqdm(range(len(when_which_where))):
            event = when_which_where.iloc[step]
            self.o_paths.resize(self.o_paths.shape[0]+1, axis=0)
            self.o_paths[-1, :, :] = self.o_paths[self.o_paths.shape[0]-2, :, :]
            self.o_paths[-1, int(event['selected_atom']), :] =\
                self.o_paths[-1, int(event['selected_atom']), :] + self.jump[int(event['selected_direction'])]

        """
        timed_heat_map = TimeHeatMap(load_data_path=self.simulation_path, options=self.options, workers=self.workers)
        if timed_heat_map.load_data_path:
            timed_heat_map.process_data()

        # self.plot_data(n_points)

        # timed_oxygen_map = TimeOxygenMap(load_data_path=self.simulation_path)
        # if timed_oxygen_map.load_data_path:
        #     timed_oxygen_map.process_data()

        plt.close('all')
        """

    def plot_line(self, save_file, x, y, x_label, y_label, x_size=8, y_size=6):
        _fig = plt.figure(figsize=(x_size, y_size))
        _ax = _fig.add_subplot(111)
        _ax.set_xlabel(x_label)
        _ax.set_ylabel(y_label)
        _ax.plot(x, y)
        plt.savefig(save_file, dpi=self.options['dpi'], bbox_inches='tight')

    def plot_data(self, n_points):
        when_which_where = pd.read_csv(self.simulation_path / 'when_which_where.csv', index_col=False)
        field_path = pd.read_csv(self.simulation_path / 'field_plot.csv', index_col=False)

        if len(when_which_where) > 1:

            print("Calculating oxygen paths")
            # Histogram
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111)
            ax.set_xlabel('Direction')
            ax.set_ylabel('Count')
            ax.hist(when_which_where['selected_direction'], bins=11)
            plt.savefig(self.simulation_path / 'paths' / "Jumps.png", dpi=self.options['dpi'])

            # Histogram
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111)
            ax.set_xlabel('Atom')
            ax.set_ylabel('Count')
            ax.hist(when_which_where['selected_atom'])
            plt.savefig(self.simulation_path / 'paths' / "Atom.png", dpi=self.options['dpi'])

            self.plot_line(save_file=self.simulation_path / 'paths' / 'Field.png',
                           x=field_path['time'],
                           y=field_path['delta_energy'],
                           x_label='Time [ps]',
                           y_label='Field [eV]')

            loc_index = list(range(0, when_which_where.shape[0], int(when_which_where.shape[0] // n_points)))

            self.plot_line(save_file=self.simulation_path / 'paths' / 'Time.png',
                           x=range(len(loc_index)),
                           y=when_which_where['time'].iloc[loc_index],
                           x_label='Step',
                           y_label='Time')

            self.plot_line(save_file=self.simulation_path / 'paths' / 'delta_Time.png',
                           x=range(len(loc_index)),
                           y=when_which_where['time'].iloc[loc_index].diff(),
                           x_label='Step',
                           y_label='Time')

            del when_which_where
            del field_path
