import numpy as np
import torch
from torch.utils.data import Dataset


class GenerateData(Dataset):
    def __init__(self, func, n_points, dataset_length):
        self.func = func
        self.n_points = n_points
        self.dataset_length = dataset_length

    def __getitem__(self, item):

        amp = np.array(np.random.uniform(1, 100)).astype(np.float32)
        phi = np.array(np.random.uniform(0, 2 * np.pi)).astype(np.float32)

        a = np.random.normal(loc=-7.588976e-17, scale=2.356172e-15)
        b = np.random.normal(loc=-1.488656e-10, scale=1.820300e-09)
        c = np.random.normal(loc=1.411618e-04, scale=3.683428e-04)
        d = np.random.normal(loc=4.617189e01, scale=1.224945e01)

        time = np.array(np.random.uniform(1e5, 50e5)).astype(np.float32)
        x = np.random.uniform(0, time, self.n_points).astype(np.float32)
        x.sort()

        y = self.func(x, amp, phi, a, b, c, d)
        y += np.random.uniform(-0.25 * amp, 0.25 * amp, self.n_points).astype(
            np.float32
        )  # add noise

        params = np.stack([amp, phi, a, b, c, d]).astype(np.float32)

        return x, y, params

    def __len__(self):
        return self.dataset_length
