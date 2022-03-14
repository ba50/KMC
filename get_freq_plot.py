import argparse
from pathlib import Path

from scripts.fit_function import fit_function
from scripts.freq_plot import freq_plot
from scripts.mass_center import mass_center


def main(args):
    for data_path in args.data_paths:

        get_mass_center_args = argparse.Namespace(
            data_path=data_path,
            high_pass=False,
            one_period=True,
            smooth=None,
            fs=170,
            search=args.search,
        )
        fit_sin_args = argparse.Namespace(
            data_path=data_path,
            one_period=True,
            workers=args.workers,
            search=args.search,
        )
        freq_plot_args = argparse.Namespace(
            delta_phi=data_path
            / ("delta_phi_mass_center_x_classic_" + data_path.name + ".csv"),
            suffix="mass_center",
        )

        if args.mass_center:
            mass_center(get_mass_center_args)

        if args.fit_sin:
            fit_function(fit_sin_args)

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
    parser.add_argument("--smooth", type=int, default=None, help="smoothing factor")
    parser.add_argument("--workers", type=int, help="number of workers", default=1)
    parser.add_argument("--mass-center", action="store_true")
    parser.add_argument("--fit-sin", action="store_true")
    parser.add_argument("--freq-plot", action="store_true")
    parser.add_argument("--search", type=str, default="*", help="file search")

    main_args = parser.parse_args()

    main(main_args)
