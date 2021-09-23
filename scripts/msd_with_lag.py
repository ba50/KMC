import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from src.GenerateXYZ import GenerateXYZ


def main(args):
    data_path = args.data_path / "oxygen_map" / "positions_inf.xyz"

    num_atoms, raw_frames = GenerateXYZ.read_frames_dataframe(data_path)
    time_frames = raw_frames["time_frames"].unique()

    atom_0 = raw_frames[raw_frames["ids"] == 0]
    atom_1 = raw_frames[raw_frames["ids"] == 1]
    atom_2 = raw_frames[raw_frames["ids"] == 2]

    delta_x_0 = []
    delta_x_1 = []
    delta_x_2 = []
    for i in range(len(time_frames[:-1])):
        delta_x_0.append(
            atom_0[atom_0["time_frames"] == time_frames[i + 1]]["x"].values
            - atom_0[atom_0["time_frames"] == time_frames[i]]["x"].values
        )
        delta_x_1.append(
            atom_1[atom_1["time_frames"] == time_frames[i + 1]]["x"].values
            - atom_1[atom_1["time_frames"] == time_frames[i]]["x"].values
        )
        delta_x_2.append(
            atom_2[atom_2["time_frames"] == time_frames[i + 1]]["x"].values
            - atom_2[atom_2["time_frames"] == time_frames[i]]["x"].values
        )

    delta_x_0 = np.array(delta_x_0)
    delta_x_1 = np.array(delta_x_1)
    delta_x_2 = np.array(delta_x_2)

    plt.plot(delta_x_0)
    plt.plot(delta_x_1)
    plt.plot(delta_x_2)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path", type=str, required=True, help="path to simulation data"
    )

    main_args = parser.parse_args()

    main_args.data_path = Path(main_args.data_path)

    main(main_args)
