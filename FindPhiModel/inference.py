import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from FindPhiModel.Models import SimpleFC
from KMC.Config import Config
from KMC.FindPhi import FindPhi


def main(args):
    model = SimpleFC(256, 1)

    print("Load model from:", args.model_path)
    model.load_state_dict(torch.load(args.model_path))

    sim_path_list = [sim for sim in args.data_path.glob(args.search) if sim.is_dir()]
    print(f"Read {len(sim_path_list)} folders.")

    data_out = []
    for sim_path in sim_path_list:
        config = Config.load(sim_path / "input.kmc")

        input_path = list((sim_path / "mass_center").glob("*.csv"))
        assert len(input_path) == 1, f"No mass center in {sim_path}!"
        data = pd.read_csv(input_path[0], sep=",")

        time = data["time"].values.astype(np.float32)
        x = data["x"].values.astype(np.float32)

        time = torch.from_numpy(time)
        x = torch.from_numpy(x)

        time = torch.unsqueeze(time, 0)
        x = torch.unsqueeze(x, 0)

        phi_pred = model(time, x)
        phi_pred = phi_pred.cpu().detach().numpy()[0][0]

        data_out.append(
            {
                "phi_rad": phi_pred,
                "path": sim_path,
                "version": (lambda split: split[5])(sim_path.name.split("_")),
                "temperature_scale": config.temperature_scale,
                "frequency": config.frequency,
                "params": phi_pred,
            }
        )

    mass_center_df = pd.DataFrame(data_out)

    mass_center_df = mass_center_df.sort_values(["frequency", "version"])
    mass_center_df.to_csv(
        args.data_path / f"delta_phi_mass_center_x_dnn_{args.data_path.name}.csv",
        index=False,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True, help="")
    parser.add_argument("--data-path", type=Path, required=True, help="")
    parser.add_argument("--search", type=str, default="*", help="file search")
    parser.add_argument(
        "--one-period", action="store_true", help="Stack data points to one period"
    )

    main_args = parser.parse_args()
    main(main_args)
