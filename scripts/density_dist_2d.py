import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from KMC.Config import Config
from KMC.GenerateModel import GenerateModel

# matplotlib.use("Agg")


def density_dist(args):
    sim_path_list = args.data_path.glob("*")
    sim_path_list = [i for i in sim_path_list if i.is_dir()]

    output_mean_last_points = []
    for sim_path in tqdm(sim_path_list, position=0):
        conf = Config.load(sim_path / "input.kmc")
        sim_frames_path = sim_path / "simulation_frames.xyz"
        field_data = pd.read_csv(sim_path / "field_data.csv")

        (sim_path / "ions_density_distribution").mkdir(exist_ok=True)

        if args.x_mean_plots:
            (sim_path / "ions_density_distribution" / "x_mean_plots").mkdir(
                exist_ok=True
            )

        num_atoms, simulation_frames = GenerateModel.read_frames_dataframe(
            sim_frames_path
        )

        ions_dd = []  # ions density distribution
        for time_index, time_chunk in tqdm(
            simulation_frames.groupby("time_index"), position=1
        ):
            ions_density = np.zeros(
                (
                    int(conf.size["x"]) * 2,
                    int(conf.size["y"]) * 2,
                    int(conf.size["z"]) * 2,
                )
            )

            model = time_chunk[["x", "y", "z"]]

            for _, pos_chunk in model.iterrows():
                x = int(pos_chunk["x"]) - 1
                y = int(pos_chunk["y"]) - 1
                z = int(pos_chunk["z"]) - 1

                ions_density[x][y][z] = 1
            ions_density = ions_density.sum(axis=2)
            ions_density /= num_atoms
            ions_dd.append(ions_density)

        ions_dd = np.array(ions_dd)
        mean_ions_dd = ions_dd.mean(axis=0)

        plt.imshow(mean_ions_dd.T)
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path", type=Path, required=True, help="path to simulation data"
    )
    parser.add_argument("--smooth", type=int, default=None, help="smoothing factor")
    parser.add_argument("--x-mean-plots", action="store_true")
    main_args = parser.parse_args()

    density_dist(main_args)
