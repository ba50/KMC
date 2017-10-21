import numpy as np
import matplotlib.pyplot as plt


class Histogram:
    def __init__(self, when_which_where_path, steps):
        self.when_which_where = np.memmap(when_which_where_path, dtype='float32', mode='r', shape=(steps, 3))

    def atoms(self):
        plt.figure(1)
        plt.hist(self.when_which_where[:, 1], bins=range(0, int(np.max(self.when_which_where)+10)))

    def directions(self):
        plt.figure(2)
        plt.hist(self.when_which_where[:, 2], bins=range(0, int(np.max(self.when_which_where[:, 2])+2)))


hist = Histogram('when_which_where_7_random_02.bin', 1000000)
hist.atoms()
plt.show()

