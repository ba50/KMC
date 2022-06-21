import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")


def nyquist(args):
    delta_phi_data = pd.read_csv(args.delta_phi)

    plot_data = {
        "frequency": [],
        "phi_rad_mean": [],
        "phi_rad_sem": [],
    }

    for freq, chunk in delta_phi_data.groupby("frequency"):
        plot_data["frequency"].append(freq)

        plot_data["phi_rad_mean"].append(chunk["phi_rad"].mean())
        plot_data["phi_rad_sem"].append(chunk["phi_rad"].std() / np.sqrt(len(chunk)))

    plot_data = pd.DataFrame(plot_data)

    labels_font = {"fontname": "Times New Roman", "size": 24}
    ticks_font = {"fontname": "Times New Roman", "size": 16}

    # Delta phi
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    ax.errorbar(
        plot_data["frequency"],
        abs(plot_data["phi_rad_mean"]) * 180 / np.pi,
        yerr=plot_data["phi_rad_sem"] * 180 / np.pi,
        fmt=":.",
    )
    ax.set_xlabel("Częstotliwość [Hz]", **labels_font)
    ax.set_ylabel("|ϕ| [stopnie]", **labels_font)
    plt.xticks(**ticks_font)
    plt.yticks(**ticks_font)

    plt.xscale("log")
    plt.savefig(
        args.delta_phi.parent
        / f"delta_phi_rad_vs_freq_{args.delta_phi.parent.name}_{args.suffix}.png",
        dpi=250,
        bbox_inches="tight",
    )
    plt.close(fig)

    nq_plot = [[], []]

    for version, chunk in delta_phi_data.groupby("version"):
        nq_plot[0].append((chunk["u0"] / chunk["i0"]) * np.cos(chunk["phi_rad"]))
        nq_plot[1].append((chunk["u0"] / chunk["i0"]) * np.sin(chunk["phi_rad"]))

    nq_plot = np.array(nq_plot)

    plot_data["Re"] = nq_plot[0].mean(axis=0)
    plot_data["Im"] = nq_plot[1].mean(axis=0)
    plot_data["Re_sem"] = nq_plot[0].std(axis=0) / np.sqrt(nq_plot.shape[1])
    plot_data["Im_sem"] = nq_plot[1].std(axis=0) / np.sqrt(nq_plot.shape[1])
    plot_data["|Z|"] = np.sqrt(
        np.power(plot_data["Re"], 2) + np.power(plot_data["Im"], 2)
    )

    # Nyqiust plot
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    ax.errorbar(
        plot_data["Re"],
        -plot_data["Im"],
        xerr=plot_data["Re_sem"],
        yerr=plot_data["Im_sem"],
        fmt=":.",
    )
    ax.set_xlabel("Z' [Ω]", **labels_font)
    ax.set_ylabel("-Z'' [Ω]", **labels_font)
    plt.xticks(**ticks_font)
    plt.yticks(**ticks_font)

    """
    for _, row in plot_data.iterrows():
        ax.text(row["Re"], -row["Im"], f"{row['frequency']:.2e}")
    """

    plt.savefig(
        args.delta_phi.parent
        / f"nyquist_plot_{args.delta_phi.parent.name}_{args.suffix}.png",
        dpi=250,
        bbox_inches="tight",
    )
    plt.close(fig)

    plot_data.to_csv(
        args.delta_phi.parent
        / f"nyquist_data_{args.delta_phi.parent.name}_{args.suffix}.csv"
    )

    # |Z| plot
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    ax.errorbar(
        plot_data["frequency"],
        np.log10(plot_data["|Z|"]),
        fmt=":.",
    )
    ax.set_xlabel("Częstotliwość [Hz]", **labels_font)
    ax.set_ylabel("Log|Z| [Ω]", **labels_font)
    plt.xticks(**ticks_font)
    plt.yticks(**ticks_font)

    plt.xscale("log")
    plt.savefig(
        args.delta_phi.parent
        / f"abs_z_freq_plot_{args.delta_phi.parent.name}_{args.suffix}.png",
        dpi=250,
        bbox_inches="tight",
    )
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--delta-phi", type=Path, required=True, help="path to delta phi csv"
    )
    parser.add_argument("--suffix", type=str, required=True)
    main_args = parser.parse_args()

    nyquist(main_args)
