from os import path

import numpy as np
import matplotlib.pyplot as plt

def get_heat_map(_oxygen_path, _oxygen_index):
    diff = np.diff(_oxygen_path[:, _oxygen_index], axis=0)

    max_position = int(max(_oxygen_path[:, 0, 0]))
    min_position = int(min(_oxygen_path[:, 0, 0]))

    _heat_map = np.zeros((len((range(min_position, max_position+1))), 1))

    jumps = np.where(diff[:, 0]!=0)[0]
    for time_index in jumps:
        _heat_map[int(_oxygen_path[time_index, 0, 0]), 0] += _oxygen_path[time_index, 0, 0]

    return np.rot90(_heat_map)


data_path = '7_7_7_random'
param_file = path.join(data_path, 'params.dat')
time_file = path.join(data_path, 'time_vector.npy')
data_file = path.join(data_path, 'update_vector.npy')


shape = np.genfromtxt(param_file).astype(np.int)
oxygen_path = np.load(data_file, mmap_mode='r')
time = np.load(time_file)
oxygen_path = oxygen_path.reshape(shape[0], shape[1], 3)


diff = np.diff(oxygen_path, axis=0)
max_position = int(oxygen_path[:, :, 0].max())
min_position = int(oxygen_path[:, :, 0].min())


heat_map = np.rot90(np.zeros((len((range(min_position, max_position+1))), 1)))


jumps = [np.where(diff[:, i, 0]!=0)[0] for i in range(oxygen_path.shape[1])]

for jump in jumps:
    for time_index in jump:
        heat_map[int(oxygen_path[time_index, 0, 0]), 0] += _oxygen_path[time_index, 0, 0]


exit()
heat_map = np.sum(heat_maps,axis=0)
plt.imshow(heat_map)
plt.show()

