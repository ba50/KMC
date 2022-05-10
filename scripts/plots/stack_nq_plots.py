import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

from tqdm import tqdm


def main(args):
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    for index, sim_name in enumerate(tqdm(args.sim_list)):
        nq_plot = pd.read_csv(args.data_path / sim_name / ("nyquist_data_" + sim_name + "_mass_center.csv"))

        ax.errorbar(
            nq_plot["Re"],
            -nq_plot["Im"],
            xerr=nq_plot["Re_sem"],
            yerr=nq_plot["Im_sem"],
            fmt=":",
            color=colors[index % len(colors)],
            label=sim_name
        )
    fig.legend()
    ax.set_xlabel("Z' [Ω]")
    ax.set_ylabel("-Z'' [Ω]")

    plt.savefig(
        args.data_path
        / f"nyquist_plot_{'.'.join(args.sim_list)}.png",
        dpi=250,
        bbox_inches="tight",
    )
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-path", type=Path, required=True, help="Path to simulation data"
    )
    parser.add_argument(
        "--sim-list", nargs="+", required=True, help="list of simulations paths"
    )
    main_args = parser.parse_args()

    main(main_args)
