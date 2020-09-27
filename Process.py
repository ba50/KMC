import argparse
from pathlib import Path

from src.utils.config import get_config
from src.DataProcess import DataProcess


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="path to simulation data")
    parser.add_argument("--workers", type=int, help="number of workers", default=1)
    parser.add_argument("--time_points", type=int, help="Number of plot points", default=10**2)
    parser.add_argument("--read_len", type=int, help="Len of read file in steps", default=10**5)
    parser.add_argument("--dpi", type=int, help="Plotting dpi", default=100)
    parser.add_argument("--plots", action='store_true', help="Process plots")
    parser.add_argument("--o_paths", action='store_true', help="Process oxygen paths")
    parser.add_argument("--heat_map", action='store_true', help="Process heat map")
    parser.add_argument("--ox_map", action='store_true', help="Process oxygen map")
    main_args = parser.parse_args()

    main_args.data_path = Path(main_args.data_path)

    sim_path_list = [sim for sim in main_args.data_path.glob("*") if sim.is_dir()]

    for sim_path in sim_path_list:
        sim_config = get_config(sim_path/'input.kmc')
        if list(sim_path.glob('heat_map/*')):
            DataProcess(simulation_path=sim_path).run(main_args)
