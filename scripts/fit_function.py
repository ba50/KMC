import argparse
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

from KMC.FindPhi import FindPhi


def fit_function(args):
    find_phi = FindPhi(args.one_period)

    sim_path_list = [
        sim for sim in args.data_path.glob("*") if sim.is_dir()
    ]
    with Pool(args.workers) as p:
        data_out = p.map(find_phi.run, sim_path_list)

    mass_center_df = []
    for chunk in data_out:
        mass_center_df.append(chunk["mass_center"])

    mass_center_df = pd.DataFrame(mass_center_df)

    data_out = {"mass_center": mass_center_df}

    for df_type in data_out:
        data_out[df_type] = data_out[df_type].sort_values(["frequency", "version"])
        data_out[df_type].to_csv(
            args.data_path / f"delta_phi_{df_type}_x_{args.data_path.name}.csv",
            index=False,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path", type=Path, required=True, help="path to data from simulation"
    )
    parser.add_argument("--workers", type=int, help="number of workers", default=1)
    parser.add_argument(
        "--one-period", action="store_true", help="Stack data points to one period"
    )
    main_args = parser.parse_args()

    fit_function(main_args)
