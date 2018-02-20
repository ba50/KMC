import numpy as np
import matplotlib.pyplot as plt


class MSD:
    def __init__(self):
        shape = np.genfromtxt('param_11_random_00.dat').astype(np.int)
        self.oxygen_path = np.memmap('update_vector.bin', dtype='float32', mode='r+', shape=(shape[0], shape[1]*3-2))

        self.oxygen_path = self.oxygen_path[:1000]

        self.time = self.oxygen_path[:-2, 0]
        self.data = np.zeros(self.oxygen_path.shape)
        for index in range(1, self.oxygen_path.shape[1]):
            self.data[:, index] = self.msd_fft(self.oxygen_path[:,index:index+3])
        
    def autocorrFFT(self, x):
        N=len(x)
        F = np.fft.fft(x, n=2*N)  #2*N because of zero-padding
        PSD = F * F.conjugate()
        res = np.fft.ifft(PSD)
        res= (res[:N]).real   #now we have the autocorrelation in convention B
        n=N*np.ones(N)-np.arange(0,N) #divide res(m) by (N-m)
        return res/n #this is the autocorrelation in convention

    def msd_fft(self, r):
        N=len(r)
        D=np.square(r).sum(axis=1) 
        D=np.append(D,0) 
        S2=sum([self.autocorrFFT(r[:, i]) for i in range(r.shape[1])])
        Q=2*D.sum()
        S1=np.zeros(N)
        for m in range(N):
          Q=Q-D[m-1]-D[N-m]
          S1[m]=Q/(N-m)
        return S1-2*S2


if __name__ == "__main__":
    msd = MSD()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(msd.time, msd.data[1:-1].mean(axis=1))
    ax.legend()
    ax.set_xlabel('Time /ps')
    ax.set_ylabel('MSD /au')
    plt.show()
