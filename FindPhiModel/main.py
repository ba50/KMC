import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from KMC.FindPhi import Functions
from tensorboardX import SummaryWriter
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from FindPhiModel.GenerateData import GenerateData


class FindPhi(torch.nn.Module):
    def __init__(self, output_size):
        super(FindPhi, self).__init__()
        self.fc_1_x_branch = nn.Linear(256, 1024)
        self.fc_2_x_branch = nn.Linear(1024, 2048)
        self.fc_3_x_branch = nn.Linear(2048, 1024)
        self.fc_4_x_branch = nn.Linear(1024, 512)
        self.fc_5_x_branch = nn.Linear(512, 512)
        self.fc_6_x_branch = nn.Linear(512, 256)
        self.fc_out_x_branch = nn.Linear(256, output_size)

        self.fc_1_y_branch = nn.Linear(256, 1024)
        self.fc_2_y_branch = nn.Linear(1024, 2048)
        self.fc_3_y_branch = nn.Linear(2048, 1024)
        self.fc_4_y_branch = nn.Linear(1024, 512)
        self.fc_5_y_branch = nn.Linear(512, 512)
        self.fc_6_y_branch = nn.Linear(512, 256)
        self.fc_out_y_branch = nn.Linear(256, output_size)

        self.activation = nn.Tanh()

    def forward(self, x_in, y_in):
        x = self.fc_1_x_branch(x_in)
        x = self.activation(x)

        x = self.fc_2_x_branch(x)
        x = self.activation(x)

        x = self.fc_3_x_branch(x)
        x = self.activation(x)

        x = self.fc_4_x_branch(x)
        x = self.activation(x)

        x = self.fc_5_x_branch(x)
        x = self.activation(x)

        x = self.fc_6_x_branch(x)
        x = self.activation(x)

        x = self.fc_out_x_branch(x)

        y = self.fc_1_y_branch(y_in)
        y = self.activation(y)

        y = self.fc_2_y_branch(y)
        y = self.activation(y)

        y = self.fc_3_y_branch(y)
        y = self.activation(y)

        y = self.fc_4_y_branch(y)
        y = self.activation(y)

        y = self.fc_5_y_branch(y)
        y = self.activation(y)

        y = self.fc_6_y_branch(y)
        y = self.activation(y)

        y = self.fc_out_y_branch(y)

        out = x + y
        return out


def main(args):
    print("Save at: ", args.log_dir)

    args.freq *= 1e-12  # cast to [ps]
    func = Functions(args.freq).sin_with_cubic_spline

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Train on:", device)

    train_ds = GenerateData(func, 256, args.train_ds_len)
    valid_ds = GenerateData(func, 256, 256)

    train_dataloader = DataLoader(
        train_ds, num_workers=args.num_workers, batch_size=args.batch_size
    )
    valid_dataloader = DataLoader(valid_ds, batch_size=1)

    model = FindPhi(6).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss(reduction="mean")

    writer = SummaryWriter(args.log_dir)

    valid_plots_dir = Path(args.log_dir, "valid_plots")
    valid_plots_dir.mkdir()

    save_path = Path(args.log_dir, "saved_models")
    save_path.mkdir()

    best_loss = float("inf")

    # training loop
    for epoch in range(1, args.max_epochs + 1):
        print(f"\nepoch {epoch} / {args.max_epochs}")
        model.train()
        # training loop
        train_loss_list = list()
        for batch in tqdm(train_dataloader):
            x, y, params = batch

            x = x.to(device)
            y = y.to(device)
            params = params.to(device)

            optimizer.zero_grad()

            params_pred = model(x, y)

            loss = criterion(input=params_pred, target=params)

            loss.backward()
            optimizer.step()

            train_loss_list.append(loss.detach().cpu().numpy())

        # validation
        model.eval()

        valid_loss_list = list()
        for batch in valid_dataloader:
            x, y, params = batch

            x = x.to(device)
            y = y.to(device)
            params = params.to(device)

            params_pred = model(x, y)

            loss = criterion(input=params_pred, target=params)

            valid_loss_list.append(loss.detach().cpu().numpy())

        average_valid_loss = np.average(valid_loss_list)

        writer.add_scalar("train/loss", np.average(train_loss_list), epoch)
        writer.add_scalar(
            "train/lr", optimizer.state_dict()["param_groups"][0]["lr"], epoch
        )
        writer.add_scalar("valid/loss", average_valid_loss, epoch)
        writer.flush()

        if best_loss > average_valid_loss and average_valid_loss < args.valid_min:
            print(f"\nSave best model at:", average_valid_loss)
            best_loss = np.average(valid_loss_list)
            torch.save(model.state_dict(), save_path / f"model_{epoch}.pth")

            x = x.cpu().detach().numpy()[0]
            y = y.cpu().detach().numpy()[0]

            params = [
                params[0, i].cpu().detach().numpy() for i in range(params.shape[1])
            ]
            parmas_pred = [
                params_pred[0, i].cpu().detach().numpy()
                for i in range(params_pred.shape[1])
            ]

            y_pred = func(x, *parmas_pred)

            plt.figure()
            plt.plot(x, y)
            plt.plot(x, y_pred)
            plt.savefig(valid_plots_dir / f"{epoch}.png")
            plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", type=Path, required=True, help="")
    parser.add_argument("--num-workers", type=int, required=True, help="")
    parser.add_argument("--batch-size", type=int, required=True, help="")
    parser.add_argument("--train-ds-len", type=int, required=True, help="")
    parser.add_argument("--lr", type=float, required=True, help="")
    parser.add_argument("--max-epochs", type=int, required=True, help="")
    parser.add_argument("--freq", type=float, required=True, help="")
    parser.add_argument("--valid-min", type=int, required=True, help="")

    main_args = parser.parse_args()
    main(main_args)
