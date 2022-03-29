from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
from scipy import optimize

from KMC.Config import Config


def get_guess(yy):
    guess_amp = np.std(yy) * 2.0 ** 0.5
    guess_offset = np.mean(yy)
    return np.array([guess_amp, 0.0, guess_offset])


def fit_sin(tt, yy):
    '''Fit sin to the input time sequence, and return fitting parameters "amp", "omega", "phase", "offset", "freq", "period" and "fitfunc"'''
    tt = np.array(tt)
    yy = np.array(yy)
    ff = np.fft.fftfreq(len(tt), (tt[1] - tt[0]))  # assume uniform spacing
    Fyy = abs(np.fft.fft(yy))
    guess_freq = abs(
        ff[np.argmax(Fyy[1:]) + 1]
    )  # excluding the zero frequency "peak", which is related to offset
    guess_amp = np.std(yy) * 2.0 ** 0.5
    guess_offset = np.mean(yy)
    guess = np.array([guess_amp, 2.0 * np.pi * guess_freq, 0.0, guess_offset])

    def sinfunc(t, A, w, p, c):
        return A * np.sin(w * t + p) + c

    popt, pcov = optimize.curve_fit(
        sinfunc,
        tt,
        yy,
        p0=guess,
    )
    A, w, p, c = popt

    f = w / (2.0 * np.pi)
    fitfunc = lambda t: A * np.sin(w * t + p) + c
    return {
        "amp": A,
        "omega": w,
        "phase": p,
        "offset": c,
        "freq": f,
        "period": 1.0 / f,
        "fitfunc": fitfunc,
        "maxcov": np.max(pcov),
        "rawres": (guess, popt, pcov),
    }


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
    def __init__(self, one_period, df_type):
        self.one_period = one_period
        self.df_type = df_type

    def run(self, sim_path: Path):
        config = Config.load(sim_path / "input.kmc")
        field_data = pd.read_csv(sim_path / "field_data.csv")

        if self.one_period:
            field_data = self.reduce_periods(field_data, 1e12 / config.frequency)

        input_path = list((sim_path / self.df_type).glob("*.csv"))
        assert len(input_path) == 1, f"No mass center in {sim_path}!"
        data = pd.read_csv(input_path[0], sep=",")
        data.dropna(inplace=True)

        fitting_function = Functions(config.frequency * 10 ** -12)

        signal = pd.DataFrame({"time": data["time"], "y": data["x"]})

        signal["y"] -= signal["y"].mean()
        signal["y"] /= np.abs(signal["y"]).max()

        params, fit_signal = FindPhi.fit_curve_signal(
            fitting_function.sin, signal, sim_path
        )

        if params is None:
            return {
                "phi_rad": None,
                "path": sim_path,
                "version": (lambda split: split[5])(sim_path.name.split("_")),
                "temperature_scale": config.temperature_scale,
                "frequency": config.frequency,
                "u0": np.max(data["dE"]),
                "i0": np.mean(data["i"]),
                "params": None,
            }

        FindPhi._save_plots(
            sim_path, self.df_type, config.frequency, signal, fit_signal, field_data
        )

        return {
            "phi_rad": params[1],
            "path": sim_path,
            "version": (lambda split: split[5])(sim_path.name.split("_")),
            "temperature_scale": config.temperature_scale,
            "frequency": config.frequency,
            "u0": np.max(data["dE"]),
            "i0": np.mean(data["i"]),
            "params": params,
        }

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
    def reduce_periods(df, period: float):
        df["time"] %= period
        df = df.sort_values(by="time").reset_index(drop=True)

        return df

    @staticmethod
    def fit_curve_signal(fitting_function, sim_signal, sim_path):

        try:
            params, _ = optimize.curve_fit(
                fitting_function,
                sim_signal["time"],
                sim_signal["y"],
            )
        except Exception as e:
            print(e)
            print(sim_path)
            return None, None

        fit_signal = pd.DataFrame(
            {
                "time": np.linspace(
                    sim_signal["time"].iloc[0],
                    sim_signal["time"].iloc[-1],
                    len(sim_signal["time"]),
                )
            }
        )
        fit_signal["y"] = fitting_function(fit_signal["time"], *params)

        return params, fit_signal
