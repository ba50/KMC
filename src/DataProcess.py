import numpy as np
from pathlib import Path

import h5py
import pandas as pd
from tqdm import tqdm

from GenerateXYZ import GenerateXYZ
from TimeHeatMap import TimeHeatMap
from TimeOxygenMap import TimeOxygenMap

# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.utils import plotting


class DataProcess:
    jump = np.array([[1, 0, 0],
                     [-1, 0, 0],
                     [0, 1, 0],
                     [0, -1, 0],
                     [0, 0, 1],
                     [0, 0, -1]])

    def __init__(self, simulation_path: Path):
        self.simulation_path = simulation_path

        self.bi_base_positions, self.y_base_positions, self.o_base_positions = \
            GenerateXYZ.read_file(self.simulation_path / 'positions.xyz')

    def run(self, args):
        """
        Args:
            args (Namespace): Args from main process.
        """
        when_which_where = None
        field_path = pd.read_csv(self.simulation_path / 'field_plot.csv', index_col=False)

        if args.plots or args.o_paths:
            when_which_where = pd.read_csv(
                self.simulation_path / 'when_which_where.csv',
                index_col=False,
                memory_map=True,
                nrows=args.read_len
            )

        if args.plots:
            assert len(when_which_where) != 0, "No data to process in when_which_where.csv"

            print("Calculating oxygen paths")
            # Histogram
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111)
            ax.set_xlabel('Direction')
            ax.set_ylabel('Count')
            ax.hist(when_which_where['selected_direction'], bins=11)
            plt.savefig(self.simulation_path / 'paths' / "Jumps.png", dpi=args.dpi)

            # Histogram
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111)
            ax.set_xlabel('Atom')
            ax.set_ylabel('Count')
            ax.hist(when_which_where['selected_atom'])
            plt.savefig(self.simulation_path / 'paths' / "Atom.png", dpi=args.dpi)

            loc_index = list(range(0, when_which_where.shape[0], int(when_which_where.shape[0] // args.time_points)))

            plotting.plot_line(save_file=self.simulation_path / 'paths' / 'Time.png',
                               x_list=[range(len(loc_index))],
                               y_list=[when_which_where['time'].iloc[loc_index]],
                               label_list=[None],
                               x_label='Step',
                               y_label='Time',
                               dpi=args.dpi)

            plotting.plot_line(save_file=self.simulation_path / 'paths' / 'delta_Time.png',
                               x_list=[range(len(loc_index))],
                               y_list=[when_which_where['time'].iloc[loc_index].diff()],
                               label_list=[None],
                               x_label='Step',
                               y_label='Time',
                               dpi=args.dpi)

            plotting.plot_line(save_file=self.simulation_path / 'paths' / 'Field.png',
                               x_list=[field_path['time']],
                               y_list=[field_path['delta_energy']],
                               label_list=[None],
                               x_label='Time [ps]',
                               y_label='Field [eV]',
                               dpi=args.dpi)

        if args.o_paths:
            print('Save in:', self.simulation_path / 'paths')
            (self.simulation_path / 'paths').mkdir(parents=True)

            # TIMExATOMSxPOS
            file_out = h5py.File((self.simulation_path / 'paths' / 'o_paths.hdf5'), 'w')
            o_paths = file_out.create_dataset('o_paths',
                                              (1,
                                               self.o_base_positions.shape[0],
                                               self.o_base_positions.shape[1]),
                                              maxshape=(None,
                                                        self.o_base_positions.shape[0],
                                                        self.o_base_positions.shape[1]),
                                              data=self.o_base_positions,
                                              dtype=np.float32)

            for step in tqdm(range(len(when_which_where))):
                event = when_which_where.iloc[step]
                o_paths.resize(o_paths.shape[0]+1, axis=0)
                o_paths[-1, :, :] = o_paths[o_paths.shape[0]-2, :, :]
                o_paths[-1, int(event['selected_atom']), :] =\
                    o_paths[-1, int(event['selected_atom']), :] + self.jump[int(event['selected_direction'])]

        if args.heat_map:
            timed_heat_map = TimeHeatMap(
                load_data_path=self.simulation_path,
                options={'dpi': args.dpi},
                workers=args.workers
            )
            if timed_heat_map.load_data_path:
                timed_heat_map.process_data()

        if args.ox_map:
            timed_oxygen_map = TimeOxygenMap(load_data_path=self.simulation_path)
            if timed_oxygen_map.load_data_path:
                timed_oxygen_map.process_data(time_end=field_path['time'].iloc[-1])

        plt.close('all')
