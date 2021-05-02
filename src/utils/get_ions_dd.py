import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.GenerateXYZ import GenerateXYZ
from src.utils.config import get_config


def get_ions_dd_last_points(data_path):
    conf = get_config(data_path / 'input.kmc')
    data_path = data_path / 'oxygen_map' / 'positions.xyz'

    num_atoms, raw_frames = GenerateXYZ.read_frames_dataframe(data_path)

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
        smooth_y = smooth_y.rolling(14).sum()
        last_points.append(smooth_y.iloc[-2])

    plt.plot(time_steps, last_points, label=f"{conf['energy_base']}")

def main(args):
    sim_path_list = args.data_path.glob('*')
    sim_path_list = [i for i in sim_path_list if i.is_dir()]
    for sim_path in sim_path_list:
        get_ions_dd_last_points(sim_path)

    plt.xlabel('time_step')
    plt.ylabel('Ions density last point')
    plt.legend()
    plt.savefig(args.data_path / 'ions_dd_last_points.png')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="path to simulation data")
    parser.add_argument("--smooth", type=int, default=14, help="smoothing factor")

    main_args = parser.parse_args()

    main_args.data_path = Path(main_args.data_path)

    main(main_args)

