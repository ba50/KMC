import argparse
from multiprocessing import Pool
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
from scipy import optimize

from KMC.Config import Config


def remove_line_function(fitting_function, signal):
    params, _ = optimize.curve_fit(
        fitting_function.sin_with_line,
        signal["time"],
        signal["y"],
        p0=[0, 0, 0, 0],
    )
    params = {
        "sine_amp": params[0],
        "sine_phi": params[1],
        "line_a": params[2],
        "line_b": params[3],
    }

    for index in range(len(signal)):
        signal["y"].iloc[index] -= fitting_function.line(
            signal["time"].iloc[index], params["line_a"], params["line_b"]
        )
    return signal


def fit_curve_signal(fitting_function, sim_signal):
    params, _ = optimize.curve_fit(
        fitting_function.sin,
        sim_signal["time"],
        sim_signal["y"],
        p0=[0, 0],
    )
    params = {"sine_amp": params[0], "sine_phi": abs(params[1])}

    fit_y = []
    fit_signal = pd.DataFrame(
        {
            "time": np.linspace(
                sim_signal["time"].iloc[0],
                sim_signal["time"].iloc[-1],
                len(sim_signal["time"]),
            )
        }
    )
    for step in fit_signal["time"]:
        fit_y.append(fitting_function.sin(step, params["sine_amp"], params["sine_phi"]))
    fit_signal["y"] = np.array(fit_y)
    return params, fit_signal


class Function:
    def __init__(self, sine_frequency):
        self.sine_frequency = sine_frequency

    def sin(self, x, amp, phi):
        return amp * np.sin(2 * np.pi * self.sine_frequency * x + phi)

    def sin_with_line(self, x, amp, phi, a, b):
        return amp * np.sin(2 * np.pi * self.sine_frequency * x + phi) + a * x + b

    @staticmethod
    def line(x, a, b):
        return a * x + b

    @staticmethod
    def mse(y_true, y_pred):
        return ((y_true - y_pred) ** 2).mean(axis=0)

    @staticmethod
    def mape(y_true, y_pred):
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    @staticmethod
    def exp_decay(x, amp: float = 50, tau: float = 5):
        return amp * np.exp(-x / tau)


def generate_phi(sim_path):
    config = Config.load(sim_path / "input.kmc")
    field_data = pd.read_csv(sim_path / "field_data.csv")

    outputs = {}
    for df_type in ["msd", "mass_center"]:
        input_path = list((sim_path / df_type).glob("*.csv"))
        assert len(input_path) == 1, f"in {sim_path}: {input_path}"
        data = pd.read_csv(input_path[0], sep=",")

        fitting_function = Function(config.frequency * 10 ** -12)
        signal = pd.DataFrame({"time": data["time"], "y": data["x"]})
        signal = remove_line_function(fitting_function, signal)
        params, fit_signal = fit_curve_signal(fitting_function, signal)

        _fig, _ax1 = plt.subplots()
        _ax2 = _ax1.twinx()
        _ax1.scatter(signal["time"], signal["y"], marker=".", color="b")
        _ax1.plot(
            fit_signal["time"],
            fit_signal["y"],
            color="r",
            linestyle="--",
            label="Fitted func",
        )
        _ax2.plot(
            field_data["time"],
            field_data["delta_energy"],
            linestyle="-",
            color="g",
            label="Field",
        )

        _ax1.set_xlabel("Time [ps]")
        _ax1.xaxis.set_major_formatter(mtick.FormatStrFormatter("%.1e"))
        _ax1.set_ylabel("Data", color="b")
        _ax2.set_ylabel("Field [eV]", color="g")

        _ax1.legend(loc="upper left")
        _ax2.legend(loc="upper right")

        plt.savefig(
            sim_path / df_type / f"fit_sin_{df_type}_x_freq_{config.frequency:.2e}.png"
        )
        plt.close(_fig)
        direction_dict = {
            "phi_rad": params["sine_phi"],
            "phi_deg": params["sine_phi"] * 180 / np.pi,
            "path": sim_path,
            "version": (lambda split: split[5])(sim_path.name.split("_")),
            "temperature_scale": config.temperature_scale,
            "frequency": config.frequency,
        }
        outputs[df_type] = direction_dict

    return outputs


def fit_function(args):
    sim_path_list = [sim for sim in args.data_path.glob("*") if sim.is_dir()]
    with Pool(args.workers) as p:
        data_out = p.map(generate_phi, sim_path_list)

    msd_df = []
    mass_center_df = []
    for chunk in data_out:
        msd_df.append(chunk["msd"])
        mass_center_df.append(chunk["mass_center"])

    msd_df = pd.DataFrame(msd_df)
    mass_center_df = pd.DataFrame(mass_center_df)

    data_out = {"msd": msd_df, "mass_center": mass_center_df}

    for df_type in data_out:
        data_out[df_type] = data_out[df_type].sort_values(["frequency", "version"])
        data_out[df_type].to_csv(
            args.data_path / f"delta_phi_{df_type}_x_{args.data_path.name}.csv",
            index=False,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path", type=Path, required=True, help="path to data from simulation"
    )
    parser.add_argument("--workers", type=int, help="number of workers", default=1)
    main_args = parser.parse_args()

    fit_function(main_args)
