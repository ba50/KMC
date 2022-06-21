import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pandas as pd
from tqdm import tqdm

from KMC.Config import Config
from KMC.filters import high_pass
from KMC.GenerateModel import GenerateModel
from KMC.plotting import plot_line
from scripts.fit_function import FindPhi

#matplotlib.use("Agg")

# elementary charge
e = 1.602176634e-19

# cell size
a = 5.559e-10  # at 1077K in [m]


def charge_center(args):
    sim_path_list = args.data_path.glob(args.search)
    sim_path_list = [i for i in sim_path_list if i.is_dir()]

    for sim_path in tqdm(sim_path_list):
        config = Config.load(sim_path / "input.kmc")
        sim_frames_path = sim_path / "simulation_frames_inf.xyz"
        field_data = pd.read_csv(sim_path / "potentials.csv")
        charge_center_path = sim_path / "charge_center"

        charge_center_path.mkdir(parents=True, exist_ok=True)

        _, simulation_frames = GenerateModel.read_frames_dataframe(
            sim_frames_path
        )

        charge_center_df = {"t": [], "x": []}
        for time_index, chunk in simulation_frames.groupby("time_index"):
            charge_center_df["t"].append(field_data["time"][time_index])
            if time_index == 0:
                mean_position_zero = chunk[["x"]].mean() * a
            mean_position = chunk[["x"]].mean() * a - mean_position_zero

            charge_center_df["x"].append(mean_position["x"] * 1e9)

        charge_center_df = pd.DataFrame(charge_center_df)
        charge_center_df["dE"] = field_data["v_shift"]
        charge_center_df["dx"] = charge_center_df["x"].diff()
        charge_center_df["dt"] = charge_center_df["t"].diff()
        charge_center_df["vel"] = charge_center_df["dx"] / charge_center_df["dt"]
        charge_center_df["u"] = config.amplitude

        plot_line(
            sim_path
            / "charge_center"
            / f"ions_charge_center_x_original_freq_{config.frequency:.2e}.png",
            [charge_center_df["t"]],
            [charge_center_df["x"]],
            [None],
            "Czas [ps]",
            "Położenie środka ładunku [nm]"
        )

        plot_line(
            sim_path
            / "charge_center"
            / f"ions_charge_center_vel_original_freq_{config.frequency:.2e}.png",
            [charge_center_df["t"]],
            [charge_center_df["vel"]],
            [None],
            "Czas [ps]",
            "Prędkość środka ładunku [nm/ps]",
        )

        if args.high_pass:
            charge_center_df["vel"] = high_pass(
                y=charge_center_df["vel"],
                high_cut=config.frequency,
                fs=args.fs,
            )

        if args.one_period:
            charge_center_df = FindPhi.reduce_periods(
                charge_center_df, 1e12 / config.frequency
            )

        if args.smooth:
            charge_center_df["vel"] = (
                charge_center_df["vel"].rolling(args.smooth).mean()
            )
            charge_center_df = charge_center_df.dropna()

        charge_center_df.to_csv(
            sim_path
            / "charge_center"
            / f"ions_charge_center_freq_{config.frequency:.2e}.csv",
            index=False,
        )

        plot_line(
            sim_path
            / "charge_center"
            / f"ions_charge_center_vel_freq_{config.frequency:.2e}.png",
            [charge_center_df["t"]],
            [charge_center_df["vel"]],
            [None],
            "Czas [ps]",
            "Prędkość Srodka ładunku [nm/ps]",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path", type=Path, required=True, help="Path to simulation data"
    )
    parser.add_argument("--smooth", type=int, default=None, help="Smoothing factor")
    parser.add_argument(
        "--one-period", action="store_true", help="Stack data points to one period"
    )
    parser.add_argument(
        "--high-pass", action="store_true", help="Apply high pass filter"
    )
    parser.add_argument("--fs", type=int, default="21", help="Sampling rate")
    parser.add_argument("--search", type=str, default="*", help="Simulation search")
    main_args = parser.parse_args()

    charge_center(main_args)
