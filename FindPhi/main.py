from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
import numpy as np

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


def sin(x, amp, freq, phi):
    return amp * np.sin(2 * np.pi * freq * x + phi).astype(np.float32)


class GenerateData(Dataset):
    def __init__(self, freq, dataset_length):
        self.freq = freq
        self.dataset_length = dataset_length

    def __getitem__(self, item):

        amp = np.array(np.random.uniform(1, 100)).astype(np.float32)
        phi = np.array(np.random.uniform(0, 2*np.pi)).astype(np.float32)

        x = np.random.uniform(0, 5e5, 256).astype(np.float32)
        x.sort()

        y = sin(x, amp, self.freq, phi)
        y += np.random.uniform(-0.25*amp, 0.25*amp, 256).astype(np.float32)  # add noise

        return x, y, amp, phi

    def __len__(self):
        return self.dataset_length


class FindPhi(torch.nn.Module):
    def __init__(self, freq):
        super(FindPhi, self).__init__()
        self.fc_1_x_branch = nn.Linear(256, 1024)
        self.fc_2_x_branch = nn.Linear(1024, 2048)
        self.fc_3_x_branch = nn.Linear(2048, 512)
        self.fc_4_x_branch = nn.Linear(512, 32)
        self.fc_out_x_branch = nn.Linear(32, 2)

        self.fc_1_y_branch = nn.Linear(256, 1024)
        self.fc_2_y_branch = nn.Linear(1024, 2048)
        self.fc_3_y_branch = nn.Linear(2048, 512)
        self.fc_4_y_branch = nn.Linear(512, 32)
        self.fc_out_y_branch = nn.Linear(32, 2)

        self.elu = nn.ELU(inplace=True)

    def forward(self, x_in, y_in):
        x = self.fc_1_x_branch(x_in)
        x = self.elu(x)

        x = self.fc_2_x_branch(x)
        x = self.elu(x)

        x = self.fc_3_x_branch(x)
        x = self.elu(x)

        x = self.fc_4_x_branch(x)
        x = self.elu(x)

        x = self.fc_out_x_branch(x)

        y = self.fc_1_y_branch(y_in)
        y = self.elu(y)

        y = self.fc_2_y_branch(y)
        y = self.elu(y)

        y = self.fc_3_y_branch(y)
        y = self.elu(y)

        y = self.fc_4_y_branch(y)
        y = self.elu(y)

        y = self.fc_out_y_branch(y)

        out = x + y
        return out[:, 0], out[:, 1]


def main():
    num_workers = 0
    batch_size = 512

    lr = 1e-5
    max_epochs = 10000
    freq = 1e7

    freq *= 1e-12  # cast to [ps]

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Train on:", device)

    train_ds = GenerateData(freq, 2**16)
    valid_ds = GenerateData(freq, 256)

    train_dataloader = DataLoader(train_ds, num_workers=num_workers, batch_size=batch_size)
    valid_dataloader = DataLoader(valid_ds, batch_size=1)

    model = FindPhi(freq).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss(reduction="mean")

    writer = SummaryWriter()

    valid_plots_dir = Path(writer.get_logdir(),  "valid_plots")
    valid_plots_dir.mkdir()

    save_path = Path(writer.get_logdir(),  "saved_models")
    save_path.mkdir()

    best_loss = float("inf")

    # training loop
    for epoch in range(1, max_epochs+1):
        print(f"epoch {epoch} / {max_epochs}")
        model.train()
        # training loop
        train_loss_list = list()
        for batch in train_dataloader:
            x, y, amp, phi = batch

            x = x.to(device)
            y = y.to(device)
            amp = amp.to(device)
            phi = phi.to(device)

            optimizer.zero_grad()

            amp_pred, phi_pred = model(x, y)

            amp_loss = criterion(input=amp_pred, target=amp)
            phi_loss = criterion(input=phi_pred, target=phi)

            loss = amp_loss + phi_loss

            loss.backward()
            optimizer.step()

            train_loss_list.append(loss.detach().cpu().numpy())

        # validation
        model.eval()

        valid_loss_list = list()
        for batch in valid_dataloader:
            x, y, amp, phi = batch

            x = x.to(device)
            y = y.to(device)
            amp = amp.to(device)
            phi = phi.to(device)

            amp_pred, phi_pred = model(x, y)

            amp_loss = criterion(input=amp_pred, target=amp)
            phi_loss = criterion(input=phi_pred, target=phi)

            loss = amp_loss + phi_loss

            valid_loss_list.append(loss.detach().cpu().numpy())

        average_valid_loss = np.average(valid_loss_list)

        writer.add_scalar("train/loss", np.average(train_loss_list), epoch)
        writer.add_scalar("valid/loss", average_valid_loss, epoch)
        writer.flush()

        if best_loss > average_valid_loss:
            print(f"Save best model at:", average_valid_loss)
            best_loss = np.average(valid_loss_list)
            torch.save(model.state_dict(), save_path / f"model_{epoch}.pth")

            x = x.cpu().detach().numpy()[0]
            y = y.cpu().detach().numpy()[0]

            amp = amp.cpu().detach().numpy()
            phi = phi.cpu().detach().numpy()

            amp_pred = amp_pred.cpu().detach().numpy()
            phi_pred = phi_pred.cpu().detach().numpy()

            y_pred = sin(x, amp_pred, freq, phi_pred)

            plt.figure()
            plt.plot(x, y)
            plt.plot(x, y_pred)
            plt.savefig(valid_plots_dir / f"{epoch}.png")
            plt.close()


if __name__ == "__main__":
    main()
