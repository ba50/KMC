from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
from scipy import optimize

from KMC.Config import Config

# elementary charge
e = 1.602176634e-19


class Functions:
    def __init__(self, freq):
        self.freq = freq

    def sin_with_cubic_spline(self, x, amp, phi, a, b, c, d):
        return self.sin(x, amp, phi) + Functions.cubic_spline(x, a, b, c, d)

    def sin_with_line(self, x, amp, phi, a, b):
        return self.sin(x, amp, phi) + Functions.line(x, a, b)

    def sin_with_const(self, x, amp, phi, a):
        return self.sin(x, amp, phi) + a

    def sin(self, x, amp, phi):
        return amp * np.sin(2 * np.pi * self.freq * x + phi)

    @staticmethod
    def arcsin(x):
        return np.arcsin(x)

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
    def __init__(self, df_type):
        self.df_type = df_type

    def run(self, sim_path: Path):
        config = Config.load(sim_path / "input.kmc")

        input_path = list((sim_path / self.df_type).glob("*.csv"))
        assert len(input_path) == 1, f"No mass center in {sim_path}!"
        data = pd.read_csv(input_path[0], sep=",")
        data.dropna(inplace=True)

        fitting_function = Functions(config.frequency * 10 ** -12)

        signal = pd.DataFrame({"t": data["t"], "y": data["vel"]})

        params, fit_signal = FindPhi.fit_curve_signal(
            fitting_function.sin, signal, sim_path
        )

        if params is None:
            return {
                "amp": None,
                "phi_rad": None,
                "path": sim_path,
                "version": (lambda split: split[5])(sim_path.name.split("_")),
                "temperature_scale": config.temperature_scale,
                "frequency": config.frequency,
                "u0": np.mean(data["u"]),
                "i0": None,
                "params": None,
            }

        FindPhi._save_plots(
            sim_path, self.df_type, config.frequency, signal, fit_signal, data
        )

        return {
            "amp": params[0],
            "phi_rad": params[1],
            "path": sim_path,
            "version": (lambda split: split[5])(sim_path.name.split("_")),
            "temperature_scale": config.temperature_scale,
            "frequency": config.frequency,
            "u0": np.mean(data["u"]),
            "i0": abs(2 * e * params[0] / 1e-12),
            "params": params,
        }

    @staticmethod
    def _save_plots(sim_path, df_type, frequency, signal, fit_signal, field_data):
        _fig, _ax1 = plt.subplots(figsize=(8, 6))
        _ax2 = _ax1.twinx()
        _ax1.scatter(signal["t"], signal["y"], marker=".", color="b")
        _ax1.plot(
            fit_signal["t"],
            fit_signal["y"],
            color="r",
            linestyle="--",
            label="Fitted func",
        )
        _ax2.plot(
            field_data["t"],
            field_data["dE"],
            linestyle="-",
            color="g",
            label="Field",
        )

        _ax1.set_xlabel("Time [ps]")
        _ax1.xaxis.set_major_formatter(mtick.FormatStrFormatter("%.1e"))
        _ax1.set_ylabel("Ions mass center velocity [au]", color="b")
        _ax2.set_ylabel("Field [eV]", color="g")

        _ax1.legend(loc="upper left")
        _ax2.legend(loc="upper right")

        plt.savefig(
            sim_path / df_type / f"fit_sin_{df_type}_x_freq_{frequency:.2e}.png",
            dpi=250,
            bbox_inches="tight",
        )
        plt.close(_fig)

    @staticmethod
    def reduce_periods(df, period: float):
        df["t"] %= period
        df = df.sort_values(by="t").reset_index(drop=True)

        return df

    @staticmethod
    def fit_curve_signal(fitting_function, sim_signal, sim_path):

        try:
            params, _ = optimize.curve_fit(
                fitting_function,
                sim_signal["t"],
                sim_signal["y"],
                bounds=[[0, -np.pi], [np.inf, 0]],
            )
        except Exception as e:
            print(e)
            print(sim_path)
            return None, None

        fit_signal = pd.DataFrame(
            {
                "t": np.linspace(
                    sim_signal["t"].iloc[0],
                    sim_signal["t"].iloc[-1],
                    len(sim_signal["t"]),
                )
            }
        )
        fit_signal["y"] = fitting_function(fit_signal["t"], *params)

        return params, fit_signal
