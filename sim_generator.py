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
    for index, i in enumerate(range(args.low_freq, args.high_freq)):
        test = np.logspace(i, i + 1, num=args.num_per_decade[index], endpoint=False)
        freq_list.extend(test)
    freq_list.append(pow(10, args.high_freq))

    simulations = pd.DataFrame({"frequency": freq_list})

    simulations["cell_type"] = args.cell_type
    simulations["thermalization_time"] = args.thermalization_time
    simulations["window_epsilon"] = args.window_epsilon
    simulations["contact_switch_left"] = args.contact_switch_left
    simulations["contact_switch_right"] = args.contact_switch_right
    simulations["contact_left"] = args.contact_left
    simulations["contact_right"] = args.contact_right
    simulations["energy_base"] = args.energy_base
    simulations["periods"] = simulations["frequency"].map(
        lambda x: x / freq_list[0] * args.base_periods
    )

    amp_tmp = []
    freq_tmp = []
    for freq in freq_list:
        for amp in args.amplitudes:
            amp_tmp.append(amp)
            freq_tmp.append(freq)

    amp_df = pd.DataFrame({"frequency": freq_tmp, "amplitude": amp_tmp})

    simulations = simulations.merge(amp_df)

    simulations["time_start"] = 0

    time_end = []
    for _, row in simulations.iterrows():
        total_time = row["periods"] / row["frequency"] * 10 ** 12
        time_end.append(total_time)
    simulations["time_end"] = time_end

    simulations["window"] = simulations[["periods", "frequency"]].apply(
        lambda x: (x[0] / (x[1] * 10.0 ** -12)) / args.window_points, axis=1
    )

    freq_list = simulations["frequency"]
    simulations = simulations.loc[
        np.repeat(simulations.index.values, args.versions)
    ].reset_index()
    simulations["version"] = np.array(
        [[x for x in range(args.versions)] for _ in range(len(freq_list))]
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
                str(x[5]),
                str(x[6]),
            ]
        ),
        axis=1,
    )

    simulations["path_to_data"] = simulations["sim_name"].map(lambda x: save_path / x)

    simulations = simulations.drop(columns=["level_0", "index"])

    simulations.to_csv(save_path / "simulations.csv", index=False)

    structure = GenerateModel(
        (simulations["size_x"][0], simulations["size_y"][0], simulations["size_z"][0])
    )
    if args.cell_type == "random":
        structure.generate_random()
    elif args.cell_type == "sphere":
        structure.generate_sphere(11)
    elif args.cell_type == "plane":
        structure.generate_plane(3)
    else:
        print("wrong cell type")

    for _, row in simulations.iterrows():
        sim_path = save_path / row["sim_name"]
        sim_path.mkdir(parents=True, exist_ok=True)
        Config(row).save(sim_path)

        structure.save_positions(sim_path / "model.xyz")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-path", required=True, help="path to save models")

    parser.add_argument(
        "--base-periods", type=float, help="base sin period", default=0.5
    )
    parser.add_argument("--low-freq", type=int, help="low freq, pow of 10", default=5)
    parser.add_argument("--high-freq", type=int, help="high freq, pow of 10", default=8)
    parser.add_argument(
        "--num-per-decade",
        type=int,
        nargs="+",
        help="list of number of point per decade",
        required=True,
    )
    parser.add_argument(
        "--cell-type", choices=["random", "sphere", "plane"], default="random"
    )
    parser.add_argument("--model-size", type=int, nargs="+", default=[5, 3, 3])
    parser.add_argument("--thermalization-time", type=int, default=200)
    parser.add_argument(
        "--window-points", type=int, help="points in window", default=512
    )
    parser.add_argument("--window-epsilon", type=float, default=4.0)
    parser.add_argument(
        "--contact-switch-left", type=int, default=0, help="0-off, 2-on"
    )
    parser.add_argument(
        "--contact-switch-right", type=int, default=0, help="0-off, 2-on"
    )
    parser.add_argument("--contact-left", type=float, default=1.0)
    parser.add_argument("--contact-right", type=float, default=1.0)
    parser.add_argument("--amplitudes", type=float, nargs="+", default=[0.0])
    parser.add_argument("--energy-base", type=float, default=0.0)
    parser.add_argument("--temperature-scale", type=float, nargs="+", default=[1.0])
    parser.add_argument("--versions", type=int, default=1)
    main_args = parser.parse_args()

    main_args.save_path = Path(main_args.save_path)
    main(main_args)
