import time
from pathlib import Path
from multiprocessing import Pool

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt


class MSD:
    def __init__(self, inputs):
        pos = inputs[0]
        data_path = inputs[1]
        self.file_name = data_path.stem
        file_in = h5py.File(data_path/'paths'/'o_paths.hdf5', 'r')
        o_paths = file_in.get('o_paths')

        when_which_where = pd.read_csv(
            data_path / 'when_which_where.csv',
            index_col=False,
            memory_map=True,
            nrows=10**5
        )

        self.time = when_which_where['time']

        self.data = []
        for index in tqdm(range(o_paths.shape[1]), position=pos):
            self.data.append(self.msd_fft(o_paths[:, index, :]))
        self.time = np.array(self.time)
        self.data = np.array(self.data)
        self.data = self.data.mean(axis=0)
        self.data = self.data[:-1]

    def autocorrFFT(self, x):
        N = len(x)
        F = np.fft.fft(x, n=2*N)  # 2*N because of zero-padding
        PSD = F * F.conjugate()
        res = np.fft.ifft(PSD)
        res = (res[:N]).real  # now we have the autocorrelation in convention B
        n = N*np.ones(N)-np.arange(0, N)  # divide res(m) by (N-m)
        return res/n  # this is the autocorrelation in convention

    def msd_fft(self, r):
        N = len(r)
        D = np.square(r).sum(axis=1)
        D = np.append(D, 0)
        S2 = sum([self.autocorrFFT(r[:, i]) for i in range(r.shape[1])])
        Q = 2*D.sum()
        S1 = np.zeros(N)
        for m in range(N):
            Q = Q-D[m-1]-D[N-m]
            S1[m] = Q/(N-m)
        return S1-2*S2


if __name__ == "__main__":
    with Pool(1) as p:
        msd_list = p.map(MSD, [
            (0, Path('F:\\KMC_data\\data_2020_09_23_random_mix_amp_v0\\11_7_7_random_0_a_0_1.0_low')),
        ])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    for msd in msd_list:
        ax.plot(msd.time, msd.data, label=msd.file_name)
    ax.legend()
    ax.set_xlabel('Time /ps')
    ax.set_ylabel('MSD /au')
    plt.show()
