import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm
from KMC.GenerateModel import GenerateModel
from KMC.Config import Config


def filter_path(x, filter_threshold):
    x_diff = np.diff(x)
    test_max = np.where(x_diff >= filter_threshold)[0]
    test_min = np.where(x_diff <= -filter_threshold)[0]

    distance_test = np.concatenate([test_max, test_min])
    distance_test.sort()

    for index in distance_test:
        for i in range(index, len(x) - 1):
            x[i + 1] -= x_diff[index]
    return x


def main(args):

    sim_path_list = args.data_path.glob("*")
    sim_path_list = [i for i in sim_path_list if i.is_dir()]

    for sim_path in tqdm(sim_path_list):
        conf = Config.load(sim_path / "input.kmc")
        sim_frames_path = sim_path / "simulation_frames.xyz"
        save_path = sim_path / "simulation_frames_inf.xyz"

        num_atoms, simulation_frames = GenerateModel.read_frames_dataframe(sim_frames_path)

        for atom_id in tqdm(range(num_atoms)):
            atom = simulation_frames[simulation_frames["atom_ids"] == atom_id]

            x = atom["x"].values
            y = atom["y"].values
            z = atom["z"].values

            simulation_frames.loc[simulation_frames["atom_ids"] == atom_id, "x"] = filter_path(x, conf.size['x'])
            simulation_frames.loc[simulation_frames["atom_ids"] == atom_id, "y"] = filter_path(y, conf.size["y"])
            simulation_frames.loc[simulation_frames["atom_ids"] == atom_id, "z"] = filter_path(z, conf.size["z"])

        GenerateModel.write_frames_from_dataframe(save_path, simulation_frames, num_atoms)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path", type=str, required=True, help="path to simulation data"
    )

    main_args = parser.parse_args()

    main_args.data_path = Path(main_args.data_path)

    main(main_args)
