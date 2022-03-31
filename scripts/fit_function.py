import argparse
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

from KMC.FindPhi import FindPhi


def fit_function(args):
    find_phi = FindPhi("mass_center")

    sim_path_list = [sim for sim in args.data_path.glob(args.search) if sim.is_dir()]
    assert len(sim_path_list) != 0, f"No data at: {args.data_path}"
    print(f"Read {len(sim_path_list)} folders.")
    with Pool(args.workers) as p:
        data_out = p.map(find_phi.run, sim_path_list)

    delta_phi = pd.DataFrame(data_out)

    delta_phi = delta_phi.sort_values(["frequency", "version"])
    delta_phi.to_csv(
        args.data_path / f"delta_phi_mass_center_vel_{args.data_path.name}.csv",
        index=False,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path", type=Path, required=True, help="path to data from simulation"
    )
    parser.add_argument("--workers", type=int, help="number of workers", default=1)
    parser.add_argument("--search", type=str, default="*", help="file search")
    main_args = parser.parse_args()

    fit_function(main_args)
