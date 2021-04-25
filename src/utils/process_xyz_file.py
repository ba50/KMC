import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from src.GenerateXYZ import GenerateXYZ


def filter_path(x, filter_threshold):
    x_diff = np.diff(x)
    test_max = np.where(x_diff >= filter_threshold)[0]
    test_min = np.where(x_diff <= -filter_threshold)[0]

    distance_test = np.concatenate([test_max, test_min])
    distance_test.sort()

    for index in distance_test:
        for i in range(index, len(x)-1):
            x[i + 1] -= x_diff[index]
    return x


def plot_3d(atom_0):
    import matplotlib.pyplot as plt

    plt.figure()
    ax = plt.axes(projection='3d')

    xline = atom_0['x']
    yline = atom_0['y']
    zline = atom_0['z']

    ax.plot3D(xline, yline, zline)
    plt.show()


def main(args):
    data_path = args.data_path / 'oxygen_map' / 'positions.xyz'

    num_atoms, raw_frames = GenerateXYZ.read_frames_dataframe(data_path)

    atom_0 = raw_frames[raw_frames['ids'] == 0]

    x = atom_0['x'].values
    y = atom_0['y'].values
    z = atom_0['z'].values

    x = filter_path(x, 25)
    y = filter_path(y, 15)
    z = filter_path(z, 15)

    raw_frames.loc[raw_frames['ids'] == 0, 'x'] = x
    raw_frames.loc[raw_frames['ids'] == 0, 'y'] = y
    raw_frames.loc[raw_frames['ids'] == 0, 'z'] = z

    atom_0 = raw_frames[raw_frames['ids'] == 0]
    plot_3d(atom_0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="path to simulation data")

    main_args = parser.parse_args()

    main_args.data_path = Path(main_args.data_path)

    main(main_args)
