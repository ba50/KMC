import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.GenerateXYZ import GenerateXYZ
from src.utils.config import get_config


def get_ions_dd_last_points(data_path):
    conf = get_config(data_path / 'input.kmc')
    positions_path = data_path / 'oxygen_map' / 'positions.xyz'
    field_data = pd.read_csv(data_path / 'field_plot.csv')

    num_atoms, raw_frames = GenerateXYZ.read_frames_dataframe(positions_path)

    time_steps = raw_frames['time_frames'].unique()
    x_positions = raw_frames['x'].unique()
    x_positions.sort()

    ions_dd = {'time': [], 'x': [], 'y': []}
    for time_step in time_steps:
        pos_frame = raw_frames[raw_frames['time_frames'] == time_step]
        for x_step in x_positions:
            ions_count = len(pos_frame.loc[x_step == pos_frame['x']])

            ions_dd['time'].append(time_step)
            ions_dd['x'].append(x_step)
            ions_dd['y'].append(ions_count/num_atoms)
    ions_dd = pd.DataFrame(ions_dd)

    last_points = []
    for time_step in time_steps:
        smooth_y = ions_dd[ions_dd['time'] == time_step]['y']
        smooth_y = smooth_y.rolling(8).sum()
        last_points.append(smooth_y.iloc[-2])

    field_data['last_points'] = last_points[:len(field_data['time'])]
    field_data['last_points'] = field_data['last_points'].rolling(8).sum()

    return field_data, conf['frequency']


def main(args):
    sim_path_list = args.data_path.glob('*')
    sim_path_list = [i for i in sim_path_list if i.is_dir()]

    output_df_list = []
    for sim_path in tqdm(sim_path_list):
        output_df, freq = get_ions_dd_last_points(sim_path)
        plt.figure()
        plt.plot(output_df['time'], output_df['last_points'])
        plt.xlabel('delta_energy [eV]')
        plt.ylabel('Ions density last point')
        plt.savefig(sim_path / f"ions_dd_last_points_{freq:.2e}.png")

        output_df_list.append((output_df, freq))

    plt.figure()
    for output_df, freq in output_df_list:
        plt.plot(output_df['delta_energy'], output_df['last_points'], label=f"{freq:.2e}")

    plt.xlabel('delta_energy [eV]')
    plt.ylabel('Ions density last point')
    plt.legend()
    plt.savefig(args.data_path / 'ions_dd_last_points.png')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="path to simulation data")
    parser.add_argument("--smooth", type=int, default=14, help="WIP: smoothing factor")

    main_args = parser.parse_args()

    main_args.data_path = Path(main_args.data_path)

    main(main_args)

