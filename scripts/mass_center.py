import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

from KMC.Config import Config
from KMC.GenerateModel import GenerateModel
from scripts.fit_function import FindPhi
from KMC.filters import high_pass


def mass_center(args):
    sim_path_list = args.data_path.glob("*")
    sim_path_list = [i for i in sim_path_list if i.is_dir()]

    for sim_path in tqdm(sim_path_list):
        conf = Config.load(sim_path / "input.kmc")
        sim_frames_path = sim_path / "simulation_frames_inf.xyz"
        field_data = pd.read_csv(sim_path / "field_data.csv")
        (sim_path / "mass_center").mkdir(parents=True, exist_ok=True)

        _, simulation_frames = GenerateModel.read_frames_dataframe(sim_frames_path)

        mass_center_df = {"time": [], "x": []}
        for time_index, chunk in simulation_frames.groupby("time_index"):
            mass_center_df["time"].append(field_data["time"][time_index])
            mean_position = chunk[["x"]].mean()

            mass_center_df["x"].append(mean_position["x"])

        mass_center_df = pd.DataFrame(mass_center_df)

        plt.figure()
        plt.plot(mass_center_df["time"], mass_center_df["x"])
        plt.xlabel("time [ps]")
        plt.ylabel("Ions mass center")
        plt.savefig(
            sim_path
            / "mass_center"
            / f"ions_mass_center_x_original_freq_{conf.frequency:.2e}.png"
        )
        plt.close()

        if args.high_pass:
            mass_center_df["x"] = high_pass(mass_center_df["x"], 1, 200)

        if args.one_period:
            mass_center_df = FindPhi.reduce_periods(mass_center_df, conf.frequency)

        if args.smooth:
            mass_center_df[["x", "y", "z"]] = (
                mass_center_df[["x", "y", "z"]].rolling(args.smooth).mean()
            )

            mass_center_df = mass_center_df.dropna()

        mass_center_df.to_csv(
            sim_path
            / "mass_center"
            / f"ions_mass_center_freq_{conf.frequency:.2e}.csv",
            index=False,
        )

        plt.figure()
        plt.plot(mass_center_df["time"], mass_center_df["x"])
        plt.xlabel("time [ps]")
        plt.ylabel("Ions mass center")
        plt.savefig(
            sim_path
            / "mass_center"
            / f"ions_mass_center_x_freq_{conf.frequency:.2e}.png"
        )
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path", type=Path, required=True, help="path to simulation data"
    )
    parser.add_argument("--smooth", type=int, default=None, help="smoothing factor")
    parser.add_argument(
        "--one-period", action="store_true", help="Stack data points to one period"
    )
    parser.add_argument(
        "--high-pass", action="store_true", help="Apply high pass filter"
    )
    main_args = parser.parse_args()

    mass_center(main_args)
