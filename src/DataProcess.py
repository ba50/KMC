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

jumps = np.array([[1, 0, 0],
                 [-1, 0, 0],
                 [0, 1, 0],
                 [0, -1, 0],
                 [0, 0, 1],
                 [0, 0, -1]])


def data_process(inputs):
    pos = inputs[0]
    simulation_path = inputs[1]
    args = inputs[2]

    bi_base_positions, y_base_positions, o_base_positions = \
        GenerateXYZ.read_file(simulation_path / 'positions.xyz')

    when_which_where = None
    field_path = pd.read_csv(simulation_path / 'field_plot.csv', index_col=False)

    if args.plots or args.o_paths:
        print('Save in:', simulation_path / 'paths')
        (simulation_path / 'paths').mkdir(parents=True, exist_ok=False)

        when_which_where = pd.read_csv(
            simulation_path / 'when_which_where.csv',
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
        plt.savefig(simulation_path / 'paths' / "Jumps.png", dpi=args.dpi)

        # Histogram
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111)
        ax.set_xlabel('Atom')
        ax.set_ylabel('Count')
        ax.hist(when_which_where['selected_atom'])
        plt.savefig(simulation_path / 'paths' / "Atom.png", dpi=args.dpi)

        loc_index = list(range(0, when_which_where.shape[0], int(when_which_where.shape[0] // args.time_points)))

        plotting.plot_line(save_file=simulation_path / 'paths' / 'Time.png',
                           x_list=[range(len(loc_index))],
                           y_list=[when_which_where['time'].iloc[loc_index]],
                           label_list=[None],
                           x_label='Step',
                           y_label='Time',
                           dpi=args.dpi)

        plotting.plot_line(save_file=simulation_path / 'paths' / 'delta_Time.png',
                           x_list=[range(len(loc_index))],
                           y_list=[when_which_where['time'].iloc[loc_index].diff()],
                           label_list=[None],
                           x_label='Step',
                           y_label='Time',
                           dpi=args.dpi)

        plotting.plot_line(save_file=simulation_path / 'paths' / 'Field.png',
                           x_list=[field_path['time']],
                           y_list=[field_path['delta_energy']],
                           label_list=[None],
                           x_label='Time [ps]',
                           y_label='Field [eV]',
                           dpi=args.dpi)

    if args.o_paths:
        # TIMExATOMSxPOS
        file_out = h5py.File((simulation_path / 'paths' / 'o_paths.hdf5'), 'w')

        when_which_where_o_paths = when_which_where
        if args.ions:
            ions = np.sort(np.random.choice(when_which_where['selected_atom'].unique(), size=(args.ions,)))
            when_which_where_o_paths = when_which_where.where(
                when_which_where['selected_atom'].map(lambda x: True if x in ions else False)
            ).dropna()
        else:
            ions = np.arange(o_base_positions.shape[0])

        o_paths = file_out.create_dataset('o_paths',
                                          (1,
                                           len(ions),
                                           o_base_positions.shape[1]),
                                          maxshape=(None,
                                                    len(ions),
                                                    o_base_positions.shape[1]),
                                          data=o_base_positions[ions, :],
                                          dtype=np.float32)

        for step in tqdm(range(len(when_which_where_o_paths)), position=pos):
            event = when_which_where_o_paths.iloc[step]
            selected_atom = int(event['selected_atom'])
            selected_direction = int(event['selected_direction'])

            o_paths.resize(o_paths.shape[0]+1, axis=0)
            o_paths[-1, :, :] = o_paths[o_paths.shape[0]-2, :, :]
            o_index = np.where(ions == selected_atom)
            o_paths[-1, o_index, :] = \
                o_paths[-1, o_index, :] + jumps[selected_direction]  # TODO: Error in processing. Mapping mismatch.

    if args.heat_map:
        timed_heat_map = TimeHeatMap(
            load_data_path=simulation_path,
            options={'dpi': args.dpi},
            workers=1
        )
        if timed_heat_map.load_data_path:
            timed_heat_map.process_data()

    if args.ox_map:
        timed_oxygen_map = TimeOxygenMap(load_data_path=simulation_path)
        if timed_oxygen_map.load_data_path:
            timed_oxygen_map.process_data(time_end=field_path['time'].iloc[-1])

    plt.close('all')
