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

matplotlib.use("Agg")

# elementary charge
e = 1.602176634e-19


def mass_center(args):
    sim_path_list = args.data_path.glob(args.search)
    sim_path_list = [i for i in sim_path_list if i.is_dir()]

    for sim_path in tqdm(sim_path_list):
        config = Config.load(sim_path / "input.kmc")
        sim_frames_path = sim_path / "simulation_frames_inf.xyz"
        field_data = pd.read_csv(sim_path / "potentials.csv")
        mass_center_path = sim_path / "mass_center"

        mass_center_path.mkdir(parents=True, exist_ok=True)

        n_atoms, simulation_frames = GenerateModel.read_frames_dataframe(
            sim_frames_path
        )

        mass_center_df = {"t": [], "x": []}
        for time_index, chunk in simulation_frames.groupby("time_index"):
            mass_center_df["t"].append(field_data["time"][time_index])
            mean_position = chunk[["x"]].mean()

            mass_center_df["x"].append(mean_position["x"])

        mass_center_df = pd.DataFrame(mass_center_df)
        mass_center_df["dE"] = field_data["v_shift"]
        mass_center_df["dx"] = mass_center_df["x"].diff()
        mass_center_df["dt"] = mass_center_df["t"].diff()
        mass_center_df["vel"] = mass_center_df["dx"] / mass_center_df["dt"]
        mass_center_df["u"] = config.amplitude

        _fig, _ax1 = plt.subplots(figsize=(8, 6))
        _ax2 = _ax1.twinx()
        _ax1.plot(
            mass_center_df["t"],
            mass_center_df["x"],
            color="b",
            linestyle="--",
        )
        _ax2.plot(
            field_data["time"],
            field_data["v_shift"],
            linestyle="-",
            color="g",
        )

        _ax1.set_xlabel("Czas [ps]")
        _ax1.xaxis.set_major_formatter(mtick.FormatStrFormatter("%.1e"))
        _ax1.set_ylabel("Położenie środka masy [au]", color="b")
        _ax2.set_ylabel("Pole [eV]", color="g")

        plt.savefig(
            sim_path
            / "mass_center"
            / f"ions_mass_center_x_original_freq_{config.frequency:.2e}.png",
            dpi=250,
            bbox_inches="tight",
        )
        plt.close(_fig)

        plot_line(
            sim_path
            / "mass_center"
            / f"ions_mass_center_vel_original_freq_{config.frequency:.2e}.png",
            [mass_center_df["t"]],
            [mass_center_df["vel"]],
            [None],
            "Time [ps]",
            "Ions mass center velocity [au]",
        )

        if args.high_pass:
            mass_center_df["vel"] = high_pass(
                y=mass_center_df["vel"],
                high_cut=config.frequency,
                fs=args.fs,
            )

        if args.one_period:
            mass_center_df = FindPhi.reduce_periods(
                mass_center_df, 1e12 / config.frequency
            )

        if args.smooth:
            mass_center_df["vel"] = mass_center_df["vel"].rolling(args.smooth).mean()
            mass_center_df = mass_center_df.dropna()

        mass_center_df.to_csv(
            sim_path
            / "mass_center"
            / f"ions_mass_center_freq_{config.frequency:.2e}.csv",
            index=False,
        )

        plot_line(
            sim_path
            / "mass_center"
            / f"ions_mass_center_vel_freq_{config.frequency:.2e}.png",
            [mass_center_df["t"]],
            [mass_center_df["vel"]],
            [None],
            "Time [ps]",
            "Ions mass center velocity [au]",
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

    mass_center(main_args)
