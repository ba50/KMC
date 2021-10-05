import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def freq_plot(args):
    delta_phi_data = pd.read_csv(args.delta_phi)

    plot_data = {
        "frequency": [],
        "phi_rad_mean": [],
        "phi_rad_sem": [],
        "phi_deg_mean": [],
        "phi_deg_sem": [],
    }

    for freq, chunk in delta_phi_data.groupby("frequency"):
        plot_data["frequency"].append(freq)

        plot_data["phi_rad_mean"].append(chunk["phi_rad"].mean())
        plot_data["phi_rad_sem"].append(chunk["phi_rad"].std() / np.sqrt(len(chunk)))

        plot_data["phi_deg_mean"].append(chunk["phi_deg"].mean())
        plot_data["phi_deg_sem"].append(chunk["phi_deg"].std() / np.sqrt(len(chunk)))

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    ax.errorbar(
        plot_data["frequency"],
        plot_data["phi_rad_mean"],
        yerr=plot_data["phi_rad_sem"],
        fmt="--o",
    )
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("delta phi [rad]")

    plt.xscale("log")
    plt.savefig(
        args.delta_phi.parent
        / f"delta_phi_rad_vs_freq_{args.delta_phi.parent.name}_{args.suffix}.png",
        dpi=1000,
        bbox_inches="tight",
    )
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--delta-phi", type=Path, required=True, help="path to delta phi csv"
    )
    parser.add_argument(
        "--suffix", type=str, required=True
    )
    main_args = parser.parse_args()

    freq_plot(main_args)
