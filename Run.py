import argparse
from pathlib import Path

import pandas as pd

from src.GenerateWorkers import GenerateWorkers


def main(args):
    commends = pd.DataFrame({'data_path': [i for i in args.data_path.glob('*') if i.is_dir()]})
    commends['program'] = args.program_path
    assert len(commends) != 0, "No simulations to run"

    swarm = GenerateWorkers(commends, args.workers)
    swarm.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--program_path", required=True, help="path to program bin")
    parser.add_argument("--data_path", required=True, help="path to data")
    parser.add_argument("--workers", type=int, help="number of workers", default=1)
    main_args = parser.parse_args()

    main_args.program_path = Path(main_args.program_path)
    main_args.data_path = Path(main_args.data_path)
    main(main_args)
