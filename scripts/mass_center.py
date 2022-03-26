import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

from KMC.Config import Config
from KMC.filters import high_pass
from KMC.GenerateModel import GenerateModel
from scripts.fit_function import FindPhi


def mass_center(args):
    sim_path_list = args.data_path.glob(args.search)
    sim_path_list = [i for i in sim_path_list if i.is_dir()]

    for sim_path in tqdm(sim_path_list):
        conf = Config.load(sim_path / "input.kmc")
        sim_frames_path = sim_path / "simulation_frames_inf.xyz"
        field_data = pd.read_csv(sim_path / "field_data.csv")
        mass_center_path = sim_path / "mass_center"

        if not mass_center_path.exists():
            mass_center_path.mkdir(parents=True)

            _, simulation_frames = GenerateModel.read_frames_dataframe(sim_frames_path)

            mass_center_df = {"time": [], "x": []}
            for time_index, chunk in simulation_frames.groupby("time_index"):
                mass_center_df["time"].append(field_data["time"][time_index])
                mean_position = chunk[["x"]].mean()

                mass_center_df["x"].append(mean_position["x"])

            mass_center_df = pd.DataFrame(mass_center_df)
            mass_center_df['v'] = mass_center_df["x"].diff()/mass_center_df["time"].diff()
            mass_center_df['dE'] = field_data["delta_energy"]

            plt.figure()
            plt.plot(mass_center_df["time"], mass_center_df["v"])
            plt.xlabel("time [ps]")
            plt.ylabel("Ions mass center")
            plt.savefig(
                sim_path
                / "mass_center"
                / f"ions_mass_center_x_original_freq_{conf.frequency:.2e}.png"
            )
            plt.close()

            if args.high_pass:
                mass_center_df["x"] = high_pass(
                    y=mass_center_df["x"],
                    high_cut=conf.frequency * 1e-7,
                    fs=args.fs,
                    order=1,
                )

            if args.one_period:
                mass_center_df = FindPhi.reduce_periods(
                    mass_center_df, 1e12 / conf.frequency
                )

            if args.smooth:
                mass_center_df["x"] = mass_center_df["x"].rolling(args.smooth).mean()

                mass_center_df = mass_center_df.dropna()

            mass_center_df.to_csv(
                sim_path
                / "mass_center"
                / f"ions_mass_center_freq_{conf.frequency:.2e}.csv",
                index=False,
            )

            plt.figure()
            plt.plot(mass_center_df["time"], mass_center_df["v"])
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
    parser.add_argument("--fs", type=int, default=170, help="Sampling rate")
    parser.add_argument("--search", type=str, default="*", help="file search")
    main_args = parser.parse_args()

    mass_center(main_args)
