import argparse
from pathlib import Path


from src.utils.config import get_config
from src.DataProcess import DataProcess


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True, help="path to simulation data")
    parser.add_argument("--workers", type=int, help="number of workers", default=1)
    parser.add_argument("--steps", type=int, help="number of plot steps", default=100)
    args = parser.parse_args()

    args.data_path = Path(args.data_path)

    sim_path_list = [sim for sim in args.data_path.glob("*") if sim.is_dir()]

    plot_options = {
        'dpi': 100,
        'MSD': False,
        'Len': False,
        '3D': False,
    }

    for sim_path in sim_path_list:
        sim_config = get_config(sim_path/'input.kmc')
        if list(sim_path.glob('heat_map/*')):
            plot_options['time_step'] = sim_config['window']
            DataProcess(simulation_path=sim_path, workers=args.workers, options=plot_options).run(n_points=args.steps)
