import argparse
from pathlib import Path

import numpy as np

from GenerateXYZ import GenerateXYZ


def main(args):
    bi, y, o = GenerateXYZ.read_file(args.model_path)
    
    cells = 45,11,11
    model = GenerateXYZ(cells)

    bi = bi[np.where(bi[:, 2] < cells[2]/2)]
    y = y[np.where(y[:, 2] < cells[2]/2)]
    o = o[np.where(o[:, 2] < cells[2]/2)]

    model.generate_from_array(bi, y, o)
    model.save_positions(args.save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, help="path to model in xyz")
    parser.add_argument("--save_path", help="path to save model", default=None)
    args = parser.parse_args()
    
    args.model_path = Path(args.model_path)
    if args.save_path is None:
        args.save_path = args.model_path.parents[0]/'positions_cross_section.xyz'
    else:
        args.save_path = Path(args.save_path)
    main(args)
