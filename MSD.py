import sys
import os
import numpy as np
from OxygenPath import OxygenPath
import matplotlib.pyplot as plt
import glob


class MSD:
    def __init__(self, start_position_path, when_which_where_path, steps):
        oxygen_path = OxygenPath(start_position_path)
        positions = np.copy(oxygen_path.start_positions)

        when_which_where = np.memmap(when_which_where_path, dtype='float32', mode='r', shape=(steps, 3))
        self.msd = np.zeros((steps, 2)).astype(np.float32)
        for index in np.ndindex(when_which_where.shape[0]):
            self.msd[index, 0] = when_which_where[index, 0]

            diffs = positions - oxygen_path.start_positions
            sqdist = np.square(diffs).sum(axis=1)
            self.msd[index, 1] = sqdist.mean()

            positions[int(when_which_where[index, 1]), 0] += oxygen_path.direction_vector[int(when_which_where[index, 2])][0]
            positions[int(when_which_where[index, 1]), 1] += oxygen_path.direction_vector[int(when_which_where[index, 2])][1]
            positions[int(when_which_where[index, 1]), 2] += oxygen_path.direction_vector[int(when_which_where[index, 2])][2]


if __name__ == "__main__":
    simulations = glob.glob(sys.argv[1])
    print(simulations)
    steps = int(sys.argv[2])
    labels = [os.path.splitext(os.path.basename(path_to_file))[0].replace('when_which_where_', '')
              for path_to_file in simulations]
    msds = []
    for index, label in enumerate(labels):
        if 'random' in label.split('_'):
            if '7' in label.split('_'):
                msds.append(MSD(os.path.join('build', '7_random.xyz'), simulations[index], steps))
            if '9' in label.split('_'):
                msds.append(MSD(os.path.join('build', '9_random.xyz'), simulations[index], steps))
            if '11' in label.split('_'):
                msds.append(MSD(os.path.join('build', '11_random.xyz'), simulations[index], steps))
        elif 'sphere' in label.split('_'):
            if '7' in label.split('_'):
                msds.append(MSD(os.path.join('build', '7_sphere.xyz'), simulations[index], steps))
            if '9' in label.split('_'):
                msds.append(MSD(os.path.join('build', '9_sphere.xyz'), simulations[index], steps))
            if '11' in label.split('_'):
                msds.append(MSD(os.path.join('build', '11_sphere.xyz'), simulations[index], steps))
        elif 'plane' in label.split('_'):
            if '7' in label.split('_'):
                msds.append(MSD(os.path.join('build', '7_plane.xyz'), simulations[index], steps))
            if '9' in label.split('_'):
                msds.append(MSD(os.path.join('build', '9_plane.xyz'), simulations[index], steps))
            if '11' in label.split('_'):
                msds.append(MSD(os.path.join('build', '11_plane.xyz'), simulations[index], steps))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    for index, msd in enumerate(msds):
        ax.plot(msd.msd[:, 0], msd.msd[:, 1], label=labels[index])
    ax.legend()
    ax.set_xlabel('Time /ps')
    ax.set_ylabel('MSD /au')
    plt.show()
