import argparse
from pathlib import Path

from scripts.get_ions_dd import get_ions_dd
from scripts.fit_sin import fit_sin
from scripts.freq_plot import freq_plot


def main(args):
    for data_path in args.data_paths:

        get_ions_dd_args = argparse.Namespace(data_path=data_path)
        get_ions_dd(get_ions_dd_args)

        fit_sin_args = argparse.Namespace(data_path=data_path, workers=args.workers)
        fit_sin(fit_sin_args)

        freq_plot_args = argparse.Namespace(
            delta_phi=data_path / ("delta_phi_" + data_path.name + ".csv")
        )
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
    parser.add_argument("--workers", type=int, help="number of workers", default=1)

    main_args = parser.parse_args()

    main(main_args)
