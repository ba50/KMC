import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

from KMC.Config import Config
from KMC.GenerateModel import GenerateModel


def ions_mass_center(args):
    sim_path_list = args.data_path.glob("*")
    sim_path_list = [i for i in sim_path_list if i.is_dir()]

    for sim_path in tqdm(sim_path_list):
        conf = Config.load(sim_path / "input.kmc")
        sim_frames_path = sim_path / "simulation_frames_inf.xyz"
        field_data = pd.read_csv(sim_path / "field_data.csv")
        (sim_path / "mass_center").mkdir(parents=True, exist_ok=True)

        _, simulation_frames = GenerateModel.read_frames_dataframe(sim_frames_path)

        mass_center = {"time": [], "x": [], "y": [], "z": []}
        for time_index, chunk in simulation_frames.groupby("time_index"):
            mass_center["time"].append(field_data["time"][time_index])
            mean_position = chunk[["x", "y", "z"]].mean()

            mass_center["x"].append(mean_position["x"])
            mass_center["y"].append(mean_position["y"])
            mass_center["z"].append(mean_position["z"])

        mass_center = pd.DataFrame(mass_center)
        mass_center.to_csv(
            sim_path / "mass_center" / "ions_mass_center.csv", index=False
        )

        mass_center[["x", "y", "z"]] = (
            mass_center[["x", "y", "z"]].rolling(args.smooth).mean().dropna()
        )

        plt.figure()
        plt.plot(mass_center["time"], mass_center["x"])
        plt.xlabel("time [ps]")
        plt.ylabel("Ions mass center")
        plt.savefig(
            sim_path / "mass_center" / f"ions_mass_center_x_{conf.frequency:.2e}.png"
        )
        plt.close()

        plt.figure()
        plt.plot(mass_center["time"], mass_center["y"])
        plt.xlabel("time [ps]")
        plt.ylabel("Ions mass center")
        plt.savefig(
            sim_path / "mass_center" / f"ions_mass_center_y_{conf.frequency:.2e}.png"
        )
        plt.close()

        plt.figure()
        plt.plot(mass_center["time"], mass_center["z"])
        plt.xlabel("time [ps]")
        plt.ylabel("Ions mass center")
        plt.savefig(
            sim_path / "mass_center" / f"ions_mass_center_z_{conf.frequency:.2e}.png"
        )
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path", type=Path, required=True, help="path to simulation data"
    )
    parser.add_argument("--smooth", type=int, default=8, help="smoothing factor")
    main_args = parser.parse_args()

    ions_mass_center(main_args)
