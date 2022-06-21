import pandas as pd

from KMC.Config import Config
from KMC.filters import high_pass
from KMC.GenerateModel import GenerateModel
from KMC.plotting import plot_line
from KMC.static import *
from scripts.fit_function import FindPhi


class ChargeCenter:
    def __init__(self, high_pass, fs, one_period, smooth):
        self.high_pass = high_pass
        self.fs = fs
        self.one_period = one_period
        self.smooth = smooth

    def run(self, sim_path):
        config = Config.load(sim_path / "input.kmc")
        sim_frames_path = sim_path / "simulation_frames_inf.xyz"
        field_data = pd.read_csv(sim_path / "potentials.csv")
        charge_center_path = sim_path / "charge_center"

        charge_center_path.mkdir(parents=True, exist_ok=True)

        n_atoms, simulation_frames = GenerateModel.read_frames_dataframe(
            sim_frames_path
        )

        charge_center_df = {"time": [], "x": []}
        for time_index, chunk in simulation_frames.groupby("time_index"):
            charge_center_df["time"].append(field_data["time"][time_index])
            if time_index == 0:
                mean_position_zero = chunk[["x"]].mean() * a
            mean_position = chunk[["x"]].mean() * a - mean_position_zero

            charge_center_df["x"].append(mean_position["x"] * 1e9)

        charge_center_df = pd.DataFrame(charge_center_df)
        charge_center_df["dE"] = field_data["v_shift"]
        charge_center_df["dx"] = charge_center_df["x"].diff()
        charge_center_df["dt"] = charge_center_df["time"].diff()
        charge_center_df["vel"] = charge_center_df["dx"] / charge_center_df["dt"]
        charge_center_df["i"] = (
            -2 * n_atoms * e * (charge_center_df["vel"] * 1e-9 / 1e-12) * pow(a, 2)
        )

        plot_line(
            sim_path
            / "charge_center"
            / f"ions_charge_center_x_original_freq_{config.frequency:.2e}.png",
            [charge_center_df["time"]],
            [charge_center_df["x"]],
            [None],
            "Czas [ps]",
            "Położenie środka ładunku [nm]",
        )

        plot_line(
            sim_path
            / "charge_center"
            / f"ions_charge_center_vel_original_freq_{config.frequency:.2e}.png",
            [charge_center_df["time"]],
            [charge_center_df["vel"]],
            [None],
            "Czas [ps]",
            "Prędkość środka ładunku [nm/ps]",
        )

        if self.high_pass:
            charge_center_df["vel"] = high_pass(
                y=charge_center_df["vel"],
                high_cut=config.frequency,
                fs=self.fs,
            )

        if self.one_period:
            charge_center_df = FindPhi.reduce_periods(
                charge_center_df, 1e12 / config.frequency
            )

        if self.smooth:
            charge_center_df["vel"] = (
                charge_center_df["vel"].rolling(args.smooth).mean()
            )
            charge_center_df = charge_center_df.dropna()

        charge_center_df.to_csv(
            sim_path
            / "charge_center"
            / f"ions_charge_center_freq_{config.frequency:.2e}.csv",
            index=False,
        )

        plot_line(
            sim_path
            / "charge_center"
            / f"ions_charge_center_vel_freq_{config.frequency:.2e}.png",
            [charge_center_df["time"]],
            [charge_center_df["vel"]],
            [None],
            "Czas [ps]",
            "Prędkość Srodka ładunku [nm/ps]",
        )

        plot_line(
            sim_path
            / "charge_center"
            / f"ions_charge_center_i_freq_{config.frequency:.2e}.png",
            [charge_center_df["time"]],
            [charge_center_df["i"]],
            [None],
            "Czas [ps]",
            "Prąd [A]",
        )
