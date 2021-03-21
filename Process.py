import argparse
from pathlib import Path
from multiprocessing import Pool

from src.DataProcess import data_process


def main(args):
    sim_path_list = [sim for sim in args.data_path.glob("*") if sim.is_dir()]
    sim_path_list = [((index % args.workers), sim, args) for index, sim in enumerate(sim_path_list)]

    with Pool(args.workers) as p:
        p.map(data_process, sim_path_list)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="path to simulation data")
    parser.add_argument("--workers", type=int, help="number of workers", default=1)
    parser.add_argument("--time_points", type=int, help="Number of plot points", default=10**2)
    parser.add_argument("--read_len", type=int, help="Len of read file in steps", default=None)
    parser.add_argument("--ions", type=int, help="Number of ions in o_paths", default=None)
    parser.add_argument("--dpi", type=int, help="Plotting dpi", default=100)
    parser.add_argument("--plots", action='store_true', help="Process plots")
    parser.add_argument("--o_paths", action='store_true', help="Process oxygen paths")
    parser.add_argument("--ox_map", action='store_true', help="Process oxygen map")
    main_args = parser.parse_args()

    main_args.data_path = Path(main_args.data_path)

    main(main_args)
