import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from KMC.Config import Config
from KMC.GenerateModel import GenerateModel


def get_sim_version(path: Path):
    sim_paths = list(path.parent.glob("%s*" % path.name))
    return len(sim_paths)


def main(args):
    save_path = Path(str(args.save_path) + "_v" + str(get_sim_version(args.save_path)))
    save_path.mkdir(parents=True)

    freq_list = []
    for i in range(args.low_freq, args.high_freq):
        test = np.logspace(i, i + 1, num=args.num_per_decade, endpoint=False)
        freq_list.extend(test)
    freq_list.append(pow(10, args.high_freq))

    simulations = pd.DataFrame({"frequency": freq_list})

    simulations["cell_type"] = args.cell_type
    simulations["thermalization_time"] = args.thermalization_time
    simulations["window"] = args.window
    simulations["window_epsilon"] = args.window_epsilon
    simulations["contact_switch_left"] = args.contact_switch_left
    simulations["contact_switch_right"] = args.contact_switch_right
    simulations["contact_left"] = args.contact_left
    simulations["contact_right"] = args.contact_right
    simulations["amplitude"] = args.amplitude
    simulations["energy_base"] = args.energy_base

    simulations["periods"] = simulations["frequency"].map(
        lambda freq: np.clip(freq / freq_list[0] * args.base_periods, 0, 2.5)
    )

    start_stop = {
        "time_start": [],
        "time_end": [],
        "periods": [],
        "frequency": [],
        "split": [],
    }
    for _, row in simulations.iterrows():
        total_time = row["periods"] / row["frequency"] * 10 ** 12
        last_step = 0
        steps = np.ceil(total_time / args.split).astype(int)
        for split_step, next_step in enumerate(
            range(steps, int(total_time) + steps, steps)
        ):
            start_stop["time_start"].append(last_step)
            start_stop["time_end"].append(next_step)
            start_stop["periods"].append(row["periods"])
            start_stop["frequency"].append(row["frequency"])
            start_stop["split"].append(split_step)
            last_step = next_step
    start_stop = pd.DataFrame(start_stop)
    simulations = simulations.merge(start_stop, on=["frequency", "periods"])

    simulations["window"] = simulations[["periods", "frequency"]].apply(
        lambda x: (x[0] / (x[1] * 10.0 ** -12)) / args.window_points, axis=1
    )

    freq_list = simulations["frequency"]
    simulations = simulations.loc[
        np.repeat(simulations.index.values, len(args.version))
    ].reset_index()
    simulations["version"] = np.array(
        [[x for x in args.version] for _ in range(len(freq_list))]
    ).flatten()
    freq_list = set(simulations["frequency"])

    freq_list = simulations["frequency"]

    simulations = simulations.loc[
        np.repeat(simulations.index.values, len(args.temperature_scale))
    ].reset_index()
    simulations["temperature_scale"] = np.array(
        [[x for x in args.temperature_scale] for _ in range(len(freq_list))]
    ).flatten()

    simulations["size_x"] = args.model_size[0]
    simulations["size_y"] = args.model_size[1]
    simulations["size_z"] = args.model_size[2]

    select_columns = [
        "size_x",
        "size_y",
        "size_z",
        "cell_type",
        "index",
        "version",
        "split",
        "temperature_scale",
    ]
    simulations["sim_name"] = simulations[select_columns].apply(
        lambda x: "_".join(
            [
                str(x[0]),
                str(x[1]),
                str(x[2]),
                x[3],
                str(x[4]),
                x[5],
                str(x[6]),
                str(x[7]),
            ]
        ),
        axis=1,
    )

    simulations["path_to_data"] = simulations["sim_name"].map(lambda x: save_path / x)

    simulations = simulations.drop(columns=["level_0", "index"])

    simulations.to_csv(save_path / "simulations.csv", index=False)

    for _, row in simulations.iterrows():
        sim_path = save_path / row["sim_name"]
        sim_path.mkdir(parents=True, exist_ok=True)
        Config(row).save(sim_path)

        structure = GenerateModel((row["size_x"], row["size_y"], row["size_z"]))
        if args.cell_type == "random":
            structure.generate_random()
        elif args.cell_type == "sphere":
            structure.generate_sphere(11)
        elif args.cell_type == "plane":
            structure.generate_plane(3)
        else:
            print("wrong cell type")
        structure.save_positions(sim_path / "model.xyz")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-path", required=True, help="path to save models")

    parser.add_argument("--split", type=int, help="number of subparts", default=1)
    parser.add_argument(
        "--base-periods", type=float, help="base sin period", default=0.5
    )
    parser.add_argument(
        "--window-points", type=int, help="points in window", default=200
    )
    parser.add_argument("--low-freq", type=int, help="low freq, pow of 10", default=5)
    parser.add_argument("--high-freq", type=int, help="high freq, pow of 10", default=8)
    parser.add_argument(
        "--num-per-decade", type=int, help="number of point per decade", default=5
    )

    parser.add_argument(
        "--cell-type", choices=["random", "sphere", "plane"], default="random"
    )
    parser.add_argument("--model-size", type=int, nargs="+", default=[5, 3, 3])
    parser.add_argument("--thermalization-time", type=int, default=200)
    parser.add_argument("--window", type=int, default=100)
    parser.add_argument("--window-epsilon", type=float, default=8.0)
    parser.add_argument(
        "--contact-switch-left", type=int, default=0, help="0-off, 2-on"
    )
    parser.add_argument(
        "--contact-switch-right", type=int, default=0, help="0-off, 2-on"
    )
    parser.add_argument("--contact-left", type=float, default=1.0)
    parser.add_argument("--contact-right", type=float, default=1.0)
    parser.add_argument("--amplitude", type=float, default=0.01)
    parser.add_argument("--energy-base", type=float, default=0.0)
    parser.add_argument("--temperature-scale", type=float, nargs="+", default=[1.0])
    parser.add_argument("--version", type=str, nargs="+", default=["a"])
    main_args = parser.parse_args()

    main_args.save_path = Path(main_args.save_path)
    main(main_args)
