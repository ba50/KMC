import argparse
from pathlib import Path

from scripts.charge_center import charge_center
from scripts.fit_function import fit_function
from scripts.nyquist import nyquist


def main(args):
    for data_path in args.data_paths:
        get_charge_center_args = argparse.Namespace(
            data_path=data_path,
            high_pass=False,
            one_period=True,
            smooth=None,
            workers=args.workers,
            fs=21,
            search=args.search,
        )
        fit_sin_args = argparse.Namespace(
            data_path=data_path,
            data_type=args.data_type,
            one_period=True,
            workers=args.workers,
            search=args.search,
        )
        nyquist_args = argparse.Namespace(
            delta_phi=data_path
            / (f"delta_phi_{args.data_type}_" + data_path.name + ".csv"),
            suffix="mass_center",
        )

        if args.charge_center:
            charge_center(get_charge_center_args)

        if args.fit_sin:
            fit_function(fit_sin_args)

        if args.nyquist_plot:
            nyquist(nyquist_args)


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
    parser.add_argument("--charge-center", action="store_true")
    parser.add_argument("--fit-sin", action="store_true")
    parser.add_argument("--nyquist-plot", action="store_true")
    parser.add_argument("--search", type=str, default="*", help="file search")
    parser.add_argument(
        "--data-type", choices=["charge_center", "potentials"], default="charge_center"
    )

    main_args = parser.parse_args()

    main(main_args)
