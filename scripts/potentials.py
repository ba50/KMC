import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pandas as pd
from tqdm import tqdm

from KMC.Config import Config
from KMC.static import *
from scripts.fit_function import FindPhi

matplotlib.use("Agg")


def potentials(args):
    sim_path_list = args.data_path.glob(args.search)
    sim_path_list = [i for i in sim_path_list if i.is_dir()]

    for sim_path in tqdm(sim_path_list):
        config = Config.load(sim_path / "input.kmc")
        pot_df = pd.read_csv(sim_path / "potentials.csv")
        potentials_path = sim_path / "potentials"

        potentials_path.mkdir(parents=True, exist_ok=True)

        pot_df["i"] = (pot_df["v_elec"].diff() / pot_df["time"].diff()) * (
            eps_0 * eps_r * config.size["y"] * config.size["z"] * a / config.size["x"]
        )

        if args.cut_time:
            pot_df = pot_df[pot_df["time"] > args.cut_time]

        labels_font = {"fontname": "Times New Roman", "size": 24}
        fig, ax1 = plt.subplots(figsize=(8, 6))
        ax2 = ax1.twinx()
        ax1.plot(
            pot_df["time"], pot_df["v_total"], color="b", label="v_total", marker="."
        )
        ax1.plot(
            pot_df["time"],
            pot_df["v_elec"],
            color="r",
            label="v_elec",
            linestyle=":",
            marker=".",
        )
        ax2.plot(
            pot_df["time"], pot_df["i"], color="g", label="I", linestyle=":", marker="v"
        )

        ax1.set_xlabel("Czas [ps]", **labels_font)
        ax1.xaxis.set_major_formatter(mtick.FormatStrFormatter("%.1e"))
        ax1.set_ylabel("Potencjał [V]", **labels_font)
        ax2.set_ylabel("Prąd [A]", **labels_font)

        ax1.legend(loc="upper left")
        ax2.legend(loc="upper right")

        plt.savefig(
            potentials_path / f"potentials_original_{sim_path.name}.png",
            dpi=250,
            bbox_inches="tight",
        )
        plt.close(fig)

        if args.one_period:
            pot_df = FindPhi.reduce_periods(pot_df, 1e12 / config.frequency)

        if args.smooth:
            pot_df["i"] = pot_df["i"].rolling(args.smooth).mean()
            pot_df = pot_df.dropna()

        pot_df.to_csv(
            potentials_path / f"potentials_{config.frequency:.2e}.csv", index=False
        )

        fig, ax1 = plt.subplots(figsize=(8, 6))
        ax2 = ax1.twinx()
        ax1.plot(
            pot_df["time"], pot_df["v_total"], color="b", label="v_total", marker="."
        )
        ax2.plot(
            pot_df["time"], pot_df["i"], color="g", label="I", linestyle=":", marker="v"
        )

        ax1.set_xlabel("Czas [ps]", **labels_font)
        ax1.xaxis.set_major_formatter(mtick.FormatStrFormatter("%.1e"))
        ax1.set_ylabel("Potencjał [V]", **labels_font)
        ax2.set_ylabel("Prąc [A]", **labels_font)

        ax1.legend(loc="upper left")
        ax2.legend(loc="upper right")

        plt.savefig(
            potentials_path / f"potentials_{config.frequency:.2e}.png",
            dpi=250,
            bbox_inches="tight",
        )
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path", type=Path, required=True, help="Path to simulation data"
    )
    parser.add_argument("--smooth", type=int, default=None, help="Smoothing factor")
    parser.add_argument(
        "--one-period", action="store_true", help="Stack data points to one period"
    )
    parser.add_argument("--search", type=str, default="*", help="Simulation search")
    parser.add_argument(
        "--cut-time", type=float, default=None, help="Cut simulation time"
    )
    main_args = parser.parse_args()

    potentials(main_args)
