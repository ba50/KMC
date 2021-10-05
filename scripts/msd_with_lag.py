import argparse
from multiprocessing import Pool
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from KMC.Config import Config
from KMC.GenerateModel import GenerateModel
from tqdm import tqdm
import pandas as pd


def worker(inputs):
    lag, sim_path = inputs
    conf = Config.load(sim_path / "input.kmc")
    data_path = sim_path / "simulation_frames_inf.xyz"
    field_data = pd.read_csv(sim_path / "field_data.csv")
    (sim_path / "msd").mkdir(parents=True, exist_ok=True)

    n_atoms, simulation_frames = GenerateModel.read_frames_dataframe(data_path)

    msd_list = [chunk[['x', 'y', 'z']].diff(periods=lag).reset_index(drop=True) for _, chunk in simulation_frames.groupby("atom_ids")]

    mean_msd = pd.DataFrame({"time": field_data["time"][1:], "x": .0, "y": .0, "z": .0})
    for atom_ids in range(n_atoms):
        msd_list[atom_ids]["time"] = field_data["time"]
        msd_list[atom_ids] = msd_list[atom_ids].dropna()
        mean_msd[["x", "y", "z"]] = mean_msd[["x", "y", "z"]].add(msd_list[atom_ids][["x", "y", "z"]])
    mean_msd[["x", "y", "z"]] /= n_atoms
    mean_msd.to_csv(
        sim_path / "msd" / f"mean_msd_lag_{lag}_freq_{conf.frequency:.2e}.csv",
        index=False,
    )

    plt.figure()
    plt.plot(mean_msd['time'], mean_msd[['x', "y", "z"]], label=["x", "y", "z"])

    plt.xlabel("time [ps]")
    plt.ylabel("MSD [au]")
    plt.legend()
    plt.savefig(
        sim_path / "msd" / f"msd_with_lag_{lag}_freq_{conf.frequency:.2e}.png"
    )
    plt.close()


def msd_with_lag(args):
    sim_path_list = [(args.lag, sim) for sim in args.data_path.glob("*") if sim.is_dir()]

    with Pool(args.workers) as p:
        p.map(worker, sim_path_list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path", type=str, required=True, help="path to simulation data"
    )
    parser.add_argument(
        "--lag", type=int, default=1, help="MSD lag"
    )
    parser.add_argument("--workers", type=int, help="number of workers", default=1)

    main_args = parser.parse_args()

    main_args.data_path = Path(main_args.data_path)

    msd_with_lag(main_args)
