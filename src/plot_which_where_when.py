import os
from pathlib import Path

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import tqdm

from src.TimeHeatMap import TimeHeatMap


def plot_line(save_file, x, y,  x_label, y_label, x_size=8, y_size=6, dpi=100):
    _fig = plt.figure(figsize=(x_size, y_size))
    _ax = _fig.add_subplot(111)
    _ax.set_xlabel(x_label)
    _ax.set_ylabel(y_label)
    _ax.plot(x, y)
    plt.savefig(save_file, dpi=dpi)
    plt.close(_fig)


jump = np.array([[1, 0, 0],
                 [-1, 0, 0],
                 [0, 1, 0],
                 [0, -1, 0],
                 [0, 0, 1],
                 [0, 0, -1]])


path_to_folder = Path('D:/KMC_data/sin/30_7_7_random')
path_to_data = path_to_folder / 'which_where_when.txt'
path_to_positions = path_to_folder / 'positions.xyz'
save_path = path_to_folder / 'paths'
print('Save in:', save_path)
if not os.path.exists(save_path):
    os.mkdir(save_path)

with open(path_to_data) as file_in:
    www = file_in.readlines()

# which where (delta Energy) when
www = [line.split('\t') for line in www]
www = [[int(line[0]), int(line[1]), float(line[2]), float(line[3])] for line in www]

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

hist_jumps = []
field = []
time = []
print("Calculating oxygen paths")
for which, where, delta_energy, when in tqdm.tqdm(www):
    o_paths[which].append(o_paths[which][-1]+jump[where])
    """
    for index in range(len(o_base_positions)):
        if not index == which:
            o_paths[index].append(o_paths[index][-1])
    """

    hist_jumps.append(where)
    field.append([when, delta_energy])
    time.append(when)

hist_jumps = np.array(hist_jumps)
field = np.array(field)
time = np.array(time)
o_paths = [np.array(i) for i in o_paths]

# Jmol simulation
"""
print("Calculating Jmol simulation frames")
with open(os.path.join(save_path, "simulation.xyz"), 'w') as file_out:
    atom_number = len(bi_base_positions) + len(y_base_positions) + len(o_base_positions)
    for step in tqdm.tqdm(range(1*10**3)):
        file_out.write("{}\n\n".format(atom_number))
        for pos in bi_base_positions:
            file_out.write("Bi")
            for r in pos:
                file_out.write("\t{}".format(r))
            file_out.write("\n")
        for pos in y_base_positions:
            file_out.write("Y")
            for r in pos:
                file_out.write("\t{}".format(r))
            file_out.write("\n")
        for pos in o_paths:
            file_out.write("O")
            for r in pos[step]:
                file_out.write("\t{}".format(r))
            file_out.write("\n")
        file_out.write("\n")
"""

# Histogram
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111)
ax.set_xlabel('Direction')
ax.set_ylabel('Count')
ax.hist(hist_jumps, bins=11)
plt.savefig(os.path.join(save_path, "Jumps.png"), dpi=100)

# Field Time (delta Time)
plot_line(save_file=save_path / 'Field.png', x=field[:, 0], y=field[:, 1], x_label='Time [ps]', y_label='Field [eV]')
plot_line(save_file=save_path / 'Time.png', x=range(time.shape[0]), y=time, x_label='Step', y_label='Time')
plot_line(save_file=save_path / 'delta_Time.png',
          x=range(time.shape[0]-1),
          y=np.diff(time),
          x_label='Step',
          y_label='Time')

# Paths
n = 1
fig = plt.figure(figsize=(40*n, 30*n))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(o_base_positions[:, 0], o_base_positions[:, 1], o_base_positions[:, 2], marker='x', s=20, c='k')
for i in o_paths:
    ax.plot(i[:, 0], i[:, 1], i[:, 2], linewidth=0.5)

plt.axis('off')

print("Generate frames")
for i in tqdm.tqdm(range(0, 360, 36)):
    ax.view_init(30, i)
    ax.dist = 5
    plt.savefig(os.path.join(save_path, "{}.png".format(i)), dpi=100)

timed_heat_map = TimeHeatMap(load_data_path=path_to_folder / 'heat_map')
timed_heat_map.process_data(['jumps'])
