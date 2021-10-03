import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

from KMC.Config import Config
from KMC.GenerateModel import GenerateModel


def ions_density_dist(args):
    sim_path_list = args.data_path.glob("*")
    sim_path_list = [i for i in sim_path_list if i.is_dir()]

    output_mean_last_points = []
    for sim_path in tqdm(sim_path_list):
        conf = Config.load(sim_path / "input.kmc")
        sim_frames_path = sim_path / "simulation_frames.xyz"
        field_data = pd.read_csv(sim_path / "field_data.csv")

        (sim_path / "ions_density_distribution").mkdir(exist_ok=True)
        (sim_path / "ions_density_distribution" / "x_mean_plots").mkdir(exist_ok=True)

        num_atoms, simulation_frames = GenerateModel.read_frames_dataframe(
            sim_frames_path
        )

        x_positions = simulation_frames["x"].unique()
        x_positions.sort()

        ions_dd = {"time": [], "x": [], "y": []}  # ions density distribution
        for time_index, chunk in simulation_frames.groupby("time_index"):
            for x_step in x_positions:
                ions_count = len(chunk.loc[x_step == chunk["x"]])

                ions_dd["time"].append(field_data["time"][time_index])
                ions_dd["x"].append(x_step)
                ions_dd["y"].append(ions_count / num_atoms)

        ions_dd = pd.DataFrame(ions_dd)
        ions_dd.to_csv(
            sim_path / "ions_density_distribution" / "ions_density_distribution.csv",
            index=False,
        )

        last_points = []
        for time, chunk in ions_dd.groupby("time"):
            chunk["y"] = chunk["y"].rolling(args.smooth).sum()
            last_points.append(chunk["y"].iloc[-2])

            if args.x_mean_plots:
                plt.figure()
                plt.plot(chunk["x"], chunk["y"])
                plt.xlabel("x")
                plt.ylabel("Ions density")
                plt.savefig(
                    sim_path
                    / "ions_density_distribution"
                    / "x_mean_plots"
                    / f"time_{time:.2e}.png"
                )
                plt.close()

        last_points = last_points[1:]
        field_data = field_data[:len(last_points)]

        field_data["last_points"] = last_points
        field_data.to_csv(
            sim_path / "ions_density_distribution" / "time_vs_ion_dd_last_points.csv",
            index=False,
        )

        field_data["last_points"] = field_data["last_points"].rolling(args.smooth).sum()

        plt.figure()
        plt.plot(field_data["time"], field_data["last_points"])
        plt.xlabel("time [ps]")
        plt.ylabel("Ions density last point")
        plt.savefig(
            sim_path
            / "ions_density_distribution"
            / f"ions_dd_last_points_{conf.frequency:.2e}.png"
        )
        plt.close()

        output_mean_last_points.append((field_data, conf.frequency))

    plt.figure()
    for output_df, freq in output_mean_last_points:
        plt.plot(
            output_df["delta_energy"], output_df["last_points"], label=f"{freq:.2e}"
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
    parser.add_argument("--smooth", type=int, default=8, help="smoothing factor")
    parser.add_argument("--x-mean-plots", action="store_true")
    main_args = parser.parse_args()

    ions_density_dist(main_args)
