import numpy as np
import torch
from torch.utils.data import Dataset


class GenerateData(Dataset):
    def __init__(self, func, n_points, dataset_length):
        self.func = func
        self.n_points = n_points
        self.dataset_length = dataset_length

    def __getitem__(self, item):

        amp = 1.0
        phi = np.array(np.random.uniform(-1.4, -1.6)).astype(np.float32)

        time = np.array(5e6).astype(np.float32)
        x = np.random.uniform(0, time, self.n_points).astype(np.float32)
        x.sort()

        y = self.func(x, amp, phi)
        y += np.random.uniform(-0.05 * amp, 0.05 * amp, self.n_points).astype(
            np.float32
        )  # add noise

        params = np.expand_dims(phi, -1).astype(np.float32)

        return x, y, params

    def __len__(self):
        return self.dataset_length
