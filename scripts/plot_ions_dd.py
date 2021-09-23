import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from src.utils.config import get_config
from tqdm import tqdm


def main(args):
    sim_path_list = args.data_path.glob("*")
    sim_path_list = [i for i in sim_path_list if i.is_dir()]

    output_df_list = []
    for sim_path in tqdm(sim_path_list):
        conf = get_config(sim_path / "input.kmc")
        output_df = pd.read_csv(sim_path / "ions_density.csv")

        plt.figure()
        plt.plot(output_df["time"], output_df["last_points"])
        plt.xlabel("time [ps]")
        plt.ylabel("Ions density last point")
        plt.savefig(sim_path / f"ions_dd_last_points_{conf['frequency']:.2e}.png")

        output_df_list.append((output_df, conf["frequency"]))

    plt.figure()
    for output_df, freq in output_df_list:
        plt.plot(
            output_df["delta_energy"], output_df["last_points"], label=f"{freq:.2e}"
        )

    plt.xlabel("delta_energy [eV]")
    plt.ylabel("Ions density last point")
    plt.legend()
    plt.savefig(args.data_path / "ions_dd_last_points.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path", type=str, required=True, help="path to simulation data"
    )

    main_args = parser.parse_args()

    main_args.data_path = Path(main_args.data_path)

    main(main_args)
