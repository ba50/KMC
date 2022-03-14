import argparse
import uuid
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from FindPhiModel.GenerateData import GenerateData
from FindPhiModel.Models import SimpleFC
from KMC.FindPhi import Functions


def main(args):
    model_id = str(uuid.uuid4())[:4]
    args.log_dir = args.log_dir / model_id
    print("Save at: ", args.log_dir)

    args.freq *= 1e-12  # cast to [ps]
    func = Functions(args.freq).sin

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Train on:", device)

    train_ds = GenerateData(func, args.n_points, args.train_ds_len)
    valid_ds = GenerateData(func, args.n_points, 4096)

    train_dataloader = DataLoader(
        train_ds, num_workers=args.num_workers, batch_size=args.batch_size
    )
    valid_dataloader = DataLoader(valid_ds, batch_size=1)

    model = SimpleFC(args.n_points, 1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss(reduction="mean")

    if args.load_path:
        print("Load model from:", args.load_path)
        model.load_state_dict(torch.load(args.load_path))

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
        writer.add_scalar("valid/loss", average_valid_loss, epoch)
        writer.flush()

        if best_loss > average_valid_loss and average_valid_loss < args.valid_min:
            print(f"\nSave best model at:", average_valid_loss)
            best_loss = np.average(valid_loss_list)
            torch.save(model.state_dict(), save_path / f"{model_id}_{epoch}.pth")

            x = x.cpu().detach().numpy()[0]
            y = y.cpu().detach().numpy()[0]

            phi_pred = params_pred[0, 0].cpu().detach().numpy()

            y_pred = func(x, 1.0, phi_pred)

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
    parser.add_argument("--load-path", type=Path, default=None, help="")
    parser.add_argument(
        "--n-points", type=int, default=256, help="Number of data points"
    )

    main_args = parser.parse_args()
    main(main_args)
