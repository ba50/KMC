import argparse
from pathlib import Path

from scripts.fit_function import fit_sin
from scripts.freq_plot import freq_plot
from scripts.ions_density_dist import ions_density_dist


def main(args):
    for data_path in args.data_paths:

        get_ions_dd_args = argparse.Namespace(data_path=data_path)
        fit_sin_args = argparse.Namespace(data_path=data_path, workers=args.workers)
        freq_plot_args = argparse.Namespace(
            delta_phi=data_path / ("delta_phi_" + data_path.name + ".csv")
        )

        if args.ions_dd:
            ions_density_dist(get_ions_dd_args)

        if args.fit_sin:
            fit_sin(fit_sin_args)

        if args.freq_plot:
            freq_plot(freq_plot_args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-paths",
        type=Path,
        required=True,
        nargs="+",
        help="list of paths to simulation data",
    )
    parser.add_argument("--smooth", type=int, default=14, help="smoothing factor")
    parser.add_argument("--workers", type=int, help="number of workers", default=4)
    parser.add_argument("--ions-dd", action="store_true")
    parser.add_argument("--fit-sin", action="store_true")
    parser.add_argument("--freq-plot", action="store_true")

    main_args = parser.parse_args()

    main(main_args)
