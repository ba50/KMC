import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from src.GenerateXYZ import GenerateXYZ


def main(args):
    data_path = args.data_path / 'oxygen_map' / 'positions.xyz'

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

    for time_step in time_steps[[0, 100, 199]]:
        plt.plot(x_positions, ions_dd[ions_dd['time'] == time_step]['y'], label=f'{time_step} [ps]')

    plt.xlabel('x')
    plt.ylabel('Ions density')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="path to simulation data")

    main_args = parser.parse_args()

    main_args.data_path = Path(main_args.data_path)

    main(main_args)

