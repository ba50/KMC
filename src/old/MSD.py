import numpy as np
import matplotlib.pyplot as plt
import os.path as path
from multiprocessing import Pool
import time


class MSD:
    def __init__(self, data):
        path_to_data = data[0]
        self.file_name = data[1]
        param_file = path.join(path_to_data, 'param_'+self.file_name+'.dat')
        time_file = path.join(path_to_data, 'time_vector_'+self.file_name+'.npy')
        data_file = path.join(path_to_data, 'update_vector_'+self.file_name+'.npy')
        shape = np.genfromtxt(param_file).astype(np.int)
        self.oxygen_path = np.load(data_file)

        self.time = np.load(time_file)
        self.start = time.time()

        self.oxygen_path = self.oxygen_path.reshape(shape[0], shape[1], 3)

        if len(data) < 2:
            atom_numbers = data[2]
        else:
            atom_numbers = self.oxygen_path.shape[1]

        self.data = []
        for index in range(atom_numbers):
            self.data.append(self.msd_fft(self.oxygen_path[:, index]))
        self.data = np.array(self.data)
        self.data = self.data.mean(axis=0)
        print('End: ', self.file_name, '{0:.2f} [s]'.format(time.time()-self.start))

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
    p = Pool(3)
    msd_list = p.map(MSD, [
        ('./data', '7_random_2'),
        ('./data', '7_sphere_2'),
        ('./data', '7_plane_2')
    ])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    for msd in msd_list:
        ax.plot(msd.time, msd.data, label=msd.file_name)
    ax.legend()
    ax.set_xlabel('Time /ps')
    ax.set_ylabel('MSD /au')
    plt.show()
