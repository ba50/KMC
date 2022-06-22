import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from KMC.Config import Config
from KMC.GenerateModel import GenerateModel

matplotlib.use("Agg")


def density_dist(args):
    sim_path_list = args.data_path.glob("*")
    sim_path_list = [i for i in sim_path_list if i.is_dir()]

    output_mean_last_points = []
    for sim_path in tqdm(sim_path_list):
        conf = Config.load(sim_path / "input.kmc")
        sim_frames_path = sim_path / "simulation_frames.xyz"
        field_data = pd.read_csv(sim_path / "potentials.csv")

        (sim_path / "ions_density_distribution").mkdir(exist_ok=True)

        if args.x_mean_plots:
            (sim_path / "ions_density_distribution" / "x_mean_plots").mkdir(
                exist_ok=True
            )

        num_atoms, simulation_frames = GenerateModel.read_frames_dataframe(
            sim_frames_path
        )

        ions_dd = []  # ions density distribution
        for time_index, time_chunk in simulation_frames.groupby("time_index"):
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
            ions_density = ions_density.sum(axis=(1, 2))
            ions_density /= num_atoms
            ions_density = pd.DataFrame({"mean": ions_density})

            if args.smooth:
                ions_density = ions_density.rolling(args.smooth).mean()

            ions_dd.append(ions_density)

        last_points = []
        for time_index, chunk in enumerate(ions_dd):
            last_points.append(chunk.iloc[-2])

            if args.x_mean_plots and time_index < 25:
                plt.figure()
                plt.plot(chunk)
                plt.xlabel("x")
                plt.ylabel("Ions density")
                plt.savefig(
                    sim_path
                    / "ions_density_distribution"
                    / "x_mean_plots"
                    / f"{time_index:03d}.png"
                )
                plt.close()

        ions_dd = np.array(ions_dd)

        plt.figure()
        plt.plot(ions_dd.mean(axis=0))
        plt.xlabel("x [au]")
        plt.ylabel("Ions density [au]")
        plt.savefig(
            sim_path
            / "ions_density_distribution"
            / f"ions_dd_mean_freq_{conf.frequency:.2e}.png"
        )
        plt.close()

        field_data["last_points"] = last_points
        field_data.to_csv(
            sim_path / "ions_density_distribution" / "time_vs_ion_dd_last_points.csv",
            index=False,
        )

        if args.smooth:
            field_data["last_points"] = (
                field_data["last_points"].rolling(args.smooth).sum()
            )

        plt.figure()
        plt.plot(field_data["time"], field_data["last_points"])
        plt.xlabel("time [ps]")
        plt.ylabel("Ions density last point")
        plt.savefig(
            sim_path
            / "ions_density_distribution"
            / f"ions_dd_last_points_freq_{conf.frequency:.2e}.png"
        )
        plt.close()

        output_mean_last_points.append((field_data, conf.frequency))

    plt.figure()
    for output_df, freq in output_mean_last_points:
        plt.plot(
            output_df["v_shift"], output_df["last_points"], label=f"{freq:.2e}"
        )

    plt.xlabel("delta_energy [eV]")
    plt.ylabel("Ions density last point")
    plt.legend()
    plt.savefig(args.data_path / "ions_dd_last_points.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path", type=Path, required=True, help="path to simulation data"
    )
    parser.add_argument("--smooth", type=int, default=None, help="smoothing factor")
    parser.add_argument("--x-mean-plots", action="store_true")
    main_args = parser.parse_args()

    density_dist(main_args)
