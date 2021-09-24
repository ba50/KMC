import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import numpy as np

from KMC.Config import Config
from KMC.GenerateModel import GenerateModel


def get_ions_density_dist(num_atoms: int, raw_frames: pd.DataFrame, time_steps: list):
    x_positions = raw_frames["x"].unique()
    x_positions.sort()
    data = {"time": [], "x": [], "y": []}  # ions density distribution
    for time_step in time_steps:
        pos_frame = raw_frames[raw_frames["time_frames"] == time_step]
        for x_step in x_positions:
            ions_count = len(pos_frame.loc[x_step == pos_frame["x"]])

            data["time"].append(time_step)
            data["x"].append(x_step)
            data["y"].append(ions_count / num_atoms)
    return pd.DataFrame(data)


def get_mean_last_point_in_time(field_data: pd.DataFrame, ions_dd: pd.DataFrame, time_steps: list, smooth: int):
    last_points = []
    for time_step in time_steps:
        smooth_y = ions_dd[ions_dd["time"] == time_step]["y"]
        smooth_y = smooth_y.rolling(smooth).sum()
        last_points.append(np.mean(smooth_y.iloc[-2]))

    field_data["last_points"] = last_points[:len(field_data["time"])]
    field_data["last_points"] = field_data["last_points"].rolling(smooth).sum()
    field_data = field_data.dropna()

    return field_data


def ions_density_dist(args):
    sim_path_list = args.data_path.glob("*_5_*")
    sim_path_list = [i for i in sim_path_list if i.is_dir()]

    output_mean_last_points = []
    for sim_path in tqdm(sim_path_list):
        (sim_path / "ions_density_distribution").mkdir(exist_ok=True)

        conf = Config.load(sim_path / "input.kmc")

        field_data = pd.read_csv(sim_path / "field_plot.csv")

        positions_path = sim_path / "oxygen_map" / "positions.xyz"
        num_atoms, raw_frames = GenerateModel.read_frames_dataframe(positions_path)

        time_steps = raw_frames["time_frames"].unique()

        ions_dd = get_ions_density_dist(num_atoms, raw_frames, time_steps)
        ions_dd.to_csv(sim_path / "ions_density_distribution.csv", index=False)

        for time, chunk in ions_dd.groupby("time"):
            plt.figure()
            plt.plot(chunk["x"], chunk["y"])
            plt.xlabel("x")
            plt.ylabel("Ions density")
            plt.savefig(sim_path / "ions_density_distribution" / f"time_{time:.2e}.png")
            plt.close()

        mean_last_points = get_mean_last_point_in_time(field_data, ions_dd, time_steps, args.smooth)

        plt.figure()
        plt.plot(mean_last_points["time"], mean_last_points["last_points"])
        plt.xlabel("time [ps]")
        plt.ylabel("Ions density last point")
        plt.savefig(sim_path / f"ions_dd_last_points_{conf.frequency:.2e}.png")
        plt.close()

        output_mean_last_points.append((mean_last_points, conf.frequency))

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
        "--data-path", type=str, required=True, help="path to simulation data"
    )
    parser.add_argument("--smooth", type=int, default=1, help="smoothing factor")

    main_args = parser.parse_args()

    main_args.data_path = Path(main_args.data_path)

    ions_density_dist(main_args)