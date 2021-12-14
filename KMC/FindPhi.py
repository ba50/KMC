from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
from scipy import optimize
import pandas as pd

from KMC.Config import Config


class Functions:
    def __init__(self, freq):
        self.freq = freq

    def sin_with_cubic_spline(self, x, amp, phi, a, b, c, d):
        return self.sin(x, amp, phi) + Functions.cubic_spline(x, a, b, c, d)

    def sin_with_line(self, x, amp, phi, a, b):
        return self.sin(x, amp, phi) + Functions.line(x, a, b)

    def sin(self, x, amp, phi):
        return amp * np.sin(2 * np.pi * self.freq * x + phi)

    @staticmethod
    def cubic_spline(x, a, b, c, d):
        return a * x ** 3 + b * x ** 2 + c * x + d

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


class FindPhi:
    def __init__(self, one_period):
        self.one_period = one_period

    def run(self, sim_path: Path):
        config = Config.load(sim_path / "input.kmc")
        field_data = pd.read_csv(sim_path / "field_data.csv")

        if self.one_period:
            field_data = self.reduce_periods(field_data, config.frequency)

        outputs = {}
        for df_type in ["mass_center"]:
            input_path = list((sim_path / df_type).glob("*.csv"))
            assert len(input_path) == 1, f"in {sim_path}: {input_path}"
            data = pd.read_csv(input_path[0], sep=",")

            fitting_function = Functions(config.frequency * 10 ** -12)
            signal = pd.DataFrame({"time": data["time"], "y": data["x"]})
            signal = FindPhi.remove_line_function(fitting_function, signal)
            params, fit_signal = FindPhi.fit_curve_signal(fitting_function, signal)

            FindPhi._save_plots(
                sim_path, df_type, config.frequency, signal, fit_signal, field_data
            )

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

    @staticmethod
    def _save_plots(sim_path, df_type, frequency, signal, fit_signal, field_data):
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
            sim_path / df_type / f"fit_sin_{df_type}_x_freq_{frequency:.2e}.png"
        )
        plt.close(_fig)

    @staticmethod
    def reduce_periods(df, frequency: int):
        time_limit = 1e12 / frequency
        new_time = df["time"].copy()
        for index in range(len(new_time)):
            while new_time.iloc[index] > time_limit:
                new_time.iloc[index] -= time_limit

        df["time"] = new_time.values

        df = df.sort_values(by="time")

        return df

    @staticmethod
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

        y = []
        for index in range(len(signal)):
            y.append(
                signal["y"].iloc[index]
                - fitting_function.line(
                    signal["time"].iloc[index], params["line_a"], params["line_b"]
                )
            )

        signal["y"] = y

        return signal

    @staticmethod
    def fit_curve_signal(fitting_function, sim_signal):
        params, _ = optimize.curve_fit(
            fitting_function.sin, sim_signal["time"], sim_signal["y"], p0=[0, 0],
        )
        params = {"sine_amp": params[0], "sine_phi": params[1]}

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
            fit_y.append(
                fitting_function.sin(step, params["sine_amp"], params["sine_phi"])
            )
        fit_signal["y"] = np.array(fit_y)
        return params, fit_signal
