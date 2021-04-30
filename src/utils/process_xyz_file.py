import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

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


def main(args):
    data_path = args.data_path / 'oxygen_map' / 'positions.xyz'
    save_path = args.data_path / 'oxygen_map' / 'positions_inf.xyz'

    num_atoms, raw_frames = GenerateXYZ.read_frames_dataframe(data_path)

    for atom_id in tqdm(range(num_atoms)):
        atom = raw_frames[raw_frames['ids'] == atom_id]

        x = atom['x'].values
        y = atom['y'].values
        z = atom['z'].values

        raw_frames.loc[raw_frames['ids'] == atom_id, 'x'] = filter_path(x, 25)
        raw_frames.loc[raw_frames['ids'] == atom_id, 'y'] = filter_path(y, 15)
        raw_frames.loc[raw_frames['ids'] == atom_id, 'z'] = filter_path(z, 15)

    GenerateXYZ.write_frames_from_dataframe(save_path, raw_frames, num_atoms)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="path to simulation data")

    main_args = parser.parse_args()

    main_args.data_path = Path(main_args.data_path)

    main(main_args)
