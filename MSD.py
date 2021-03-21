import argparse
from pathlib import Path
from multiprocessing import Pool

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.utils import plotting

import matplotlib.pyplot as plt


class MSD:
    def __init__(self, inputs):
        pos = inputs[0]
        data_path = inputs[1]
        ion_count = inputs[2]
        time_points = inputs[3]

        self.file_name = data_path.name
        file_in = h5py.File(data_path/'paths'/'o_paths.hdf5', 'r')
        o_paths = file_in.get('o_paths')

        when_which_where = pd.read_csv(
            data_path / 'when_which_where.csv',
            index_col=False,
            memory_map=True,
            nrows=o_paths.shape[0]
        )

        loc_index = list(range(0, when_which_where.shape[0], int(when_which_where.shape[0] // time_points)))
        self.time = when_which_where['time'].iloc[loc_index]

        self.data = []
        if ion_count is None:
            ion_count = o_paths.shape[1]
        for index in tqdm(range(ion_count), position=pos):
            self.data.append(self.msd_fft(o_paths[loc_index, index, :]))

        # self.time = np.array(self.time) TODO: Fix time
        self.time = np.array(range(len(self.data[0])))
        self.data = np.array(self.data)
        self.data = self.data.mean(axis=0)

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


def main(args):
    sim_path_list = [sim for sim in args.data_path.glob("*") if sim.is_dir()]
    sim_path_list = [(index % args.workers, sim, args.ions, 10**3) for index, sim in enumerate(sim_path_list)]

    with Pool(args.workers) as p:
        msd_list = p.map(MSD, sim_path_list)

    x_list = []
    y_list = []
    label_list = []
    for msd in msd_list:
        x_list.append(msd.time)
        y_list.append(msd.data)
        label_list.append(msd.file_name)

    plotting.plot_line(save_file=args.data_path / 'MSD.png',
                       x_list=x_list,
                       y_list=y_list,
                       label_list=label_list,
                       x_label='Time [ps]',
                       y_label='MSD [au]',
                       dpi=250)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="path to simulation data")
    parser.add_argument("--workers", type=int, help="number of workers", default=1)
    parser.add_argument("--ions", type=int, help="number of ions.", default=None)
    main_args = parser.parse_args()

    main_args.data_path = Path(main_args.data_path)

    main(main_args)
