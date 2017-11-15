import sys
import os
import numpy as np
from OxygenPath import OxygenPath
import matplotlib.pyplot as plt
import glob
from multiprocessing import Pool


class MSD:
    def __init__(self, start_position_path, when_which_where_path, data_path='.', atom_number=None, steps=None):
        self.oxygen_path = OxygenPath(start_position_path, when_which_where_path, data_path, atom_number, steps)
        
        self.data = np.zeros((self.oxygen_path.steps, self.oxygen_path.atom_number+1))
        self.time = self.oxygen_path.when_which_where[0:-2, 0]
        for index in range(self.oxygen_path.atom_number):
            self.data[:, index+1] = self.msd_fft(self.oxygen_path.paths[index, :, :])
        

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

def generate_MSD(simulation):
    data_path = os.path.join('E:', 'data_KMC')
    cell_types = ['random', 'sphere', 'plane']
    cell_sizes = ['7', '9', '11']
    atom_number = None
    steps = None

    sim_data = os.path.splitext(os.path.basename(simulation))[0].replace('when_which_where_', '')
    for cell_type in cell_types:
        if cell_type in sim_data.split('_'):
            for cell_size in cell_sizes:
                if cell_size in sim_data.split('_'):
                    return MSD(os.path.join(data_path, cell_size+'_'+cell_type+'.xyz'), simulation, data_path, atom_number, steps)

    
if __name__ == "__main__":
    simulations = glob.glob(sys.argv[1])
    print("Loading data: ", simulations)
               
    with Pool(4) as p:
        msds = p.map(generate_MSD, simulations)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    for msd in msds:
        ax.plot(msd.time, msd.data[1:-1].mean(axis=1), label=msd.oxygen_path.label)
    ax.legend()
    ax.set_xlabel('Time /ps')
    ax.set_ylabel('MSD /au')
    plt.show()
