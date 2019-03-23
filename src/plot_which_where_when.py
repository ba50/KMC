import os

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


jump = np.array([[0, 0, 1],
                 [0, 0, -1],
                 [0, 1, 0],
                 [0, -1, 0],
                 [1, 0, 0],
                 [-1, 0, 0]])

path_to_folder = os.path.join('C:/Users/barja/source/repos/KMC/KMC/KMC_data/z_pola/plus/15_7_7_random')
path_to_data = os.path.join(path_to_folder, 'which_where_when.txt')
path_to_positions = os.path.join(path_to_folder, 'positions.xyz')
save_path = os.path.join(path_to_folder, 'paths.png')

with open(path_to_data) as file_in:
    www = file_in.readlines()

www = [line[:-1].split('\t') for line in www]
www = [[int(line[0]), int(line[1]), float(line[2])] for line in www]

bi_base_positions = []
y_base_positions = []
o_base_positions = []
with open(path_to_positions) as file_in:
    for line in file_in.readlines()[2:]:
        line = line.split('\t')
        if line[0] == 'Bi':
            bi_base_positions.append([float(line[1]),
                                      float(line[2]),
                                      float(line[3])])
        if line[0] == 'Y':
            y_base_positions.append([float(line[1]),
                                     float(line[2]),
                                     float(line[3])])
        if line[0] == 'O':
            o_base_positions.append([float(line[1]),
                                     float(line[2]),
                                     float(line[3])])


bi_base_positions = np.array(bi_base_positions)
y_base_positions = np.array(y_base_positions)
o_base_positions = np.array(o_base_positions)

o_paths = []
for i in o_base_positions:
    o_paths.append([i])

for which, where, when in www:
    o_paths[which].append(o_paths[which][-1]+jump[where])

o_paths = [np.array(i) for i in o_paths]

n = 2
fig = plt.figure(figsize=(40*n, 30*n))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(o_base_positions[:, 0], o_base_positions[:, 1], o_base_positions[:, 2], marker='x', s=3, c='r')
for i in o_paths:
    ax.plot(i[:, 0], i[:, 1], i[:, 2], linewidth=1)

# ax.view_init(30, 0)
ax.dist = 5

plt.axis('off')
plt.savefig(save_path, dpi=100)
