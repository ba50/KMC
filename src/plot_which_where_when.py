import os

import pandas as pd
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import tqdm


jump = np.array([[1, 0, 0],
                 [-1, 0, 0],
                 [0, 1, 0],
                 [0, -1, 0],
                 [0, 0, 1],
                 [0, 0, -1]])

path_to_folder = os.path.join('D:/KMC_data/3_3_3_random')
path_to_data = os.path.join(path_to_folder, 'which_where_when.txt')
path_to_positions = os.path.join(path_to_folder, 'positions.xyz')
save_path = os.path.join(path_to_folder, 'paths')
print('Save in:', save_path)
if not os.path.exists(save_path):
    os.mkdir(save_path)

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

hist_jumps = []
time = []
jumps = np.concatenate((o_base_positions, np.zeros((o_base_positions.shape[0], 6))), axis=1)
jumps = pd.DataFrame(data=jumps, columns=['x', 'y', 'z', 'right', 'left', 'up', 'down', 'back', 'front'])
print("Calculating oxygen paths")
for which, where, when in tqdm.tqdm(www):
    o_paths[which].append(o_paths[which][-1]+jump[where])
    for index in range(len(o_base_positions)):
        if not index == which:
            o_paths[index].append(o_paths[index][-1])

    hist_jumps.append(where)
    time.append(when)

    if where == 0:
        jumps.loc[which]['right'] += 1
    elif where == 1:
        jumps.loc[which]['left'] += 1
    elif where == 2:
        jumps.loc[which]['up'] += 1
    elif where == 3:
        jumps.loc[which]['down'] += 1
    elif where == 4:
        jumps.loc[which]['back'] += 1
    elif where == 5:
        jumps.loc[which]['front'] += 1

jumps.to_csv(os.path.join(save_path, "total_jumps.csv"), sep=';', index=False, decimal=',')
hist_jumps = np.array(hist_jumps)
time = np.array(time)
o_paths = [np.array(i) for i in o_paths]

with open(os.path.join(save_path, "simulation.xyz"), 'w') as file_out:
    atom_number = len(bi_base_positions) + len(y_base_positions) + len(o_base_positions)
    print("Calculating simulation frames")
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

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111)
ax.set_xlabel('Direction')
ax.set_ylabel('Count')
ax.hist(hist_jumps, bins=11)
plt.savefig(os.path.join(save_path, "Jumps.png"), dpi=100)

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111)
ax.set_xlabel('Step')
ax.set_ylabel('Time')
ax.plot(time)
plt.savefig(os.path.join(save_path, "Time.png"), dpi=100)

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111)
ax.set_xlabel('Step')
ax.set_ylabel('delta Time')
ax.plot(np.diff(time))
plt.savefig(os.path.join(save_path, "delta Time.png"), dpi=100)

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
