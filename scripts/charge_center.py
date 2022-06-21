import argparse
from multiprocessing import Pool
from pathlib import Path

import matplotlib

from KMC.ChargeCenter import ChargeCenter

matplotlib.use("Agg")


def main(args):
    charge_center = ChargeCenter(args.high_pass, args.fs, args.one_period, args.smooth)
    sim_path_list = [sim for sim in args.data_path.glob(args.search) if sim.is_dir()]
    assert len(sim_path_list) != 0, f"No data at: {args.data_path}"
    print(f"Read {len(sim_path_list)} folders.")
    with Pool(args.workers) as p:
        p.map(charge_center.run, sim_path_list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path", type=Path, required=True, help="Path to simulation data"
    )
    parser.add_argument("--workers", type=int, help="number of workers", default=1)
    parser.add_argument("--smooth", type=int, default=None, help="Smoothing factor")
    parser.add_argument(
        "--one-period", action="store_true", help="Stack data points to one period"
    )
    parser.add_argument(
        "--high-pass", action="store_true", help="Apply high pass filter"
    )
    parser.add_argument("--fs", type=int, default="21", help="Sampling rate")
    parser.add_argument("--search", type=str, default="*", help="Simulation search")
    main_args = parser.parse_args()

    main(main_args)
