import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from src.GenerateXYZ import GenerateXYZ
from src.utils.config import get_config


def main(args):
    data_path = args.data_path / 'oxygen_map' / 'positions.xyz'

    config = get_config(args.data_path / 'input.kmc')

    num_atoms, raw_frames = GenerateXYZ.read_frames_dataframe(data_path)
    time_frames = raw_frames['time_frames'].unique()

    atom_0 = raw_frames[raw_frames['ids'] == 0]

    delta_x_0 = []
    for i in range(len(time_frames[:-1])):
         delta_x_0.append(atom_0[atom_0['time_frames'] == time_frames[i+1]]['x'].values - atom_0[atom_0['time_frames'] == time_frames[i]]['x'].values)

    print(delta_x_0)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="path to simulation data")

    main_args = parser.parse_args()

    main_args.data_path = Path(main_args.data_path)

    main(main_args)

