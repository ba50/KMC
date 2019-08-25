import os
from pathlib import Path

import numpy as np

import tqdm

from src.TimeHeatMap import TimeHeatMap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


jump = np.array([[1, 0, 0],
                 [-1, 0, 0],
                 [0, 1, 0],
                 [0, -1, 0],
                 [0, 0, 1],
                 [0, 0, -1]])


class DataProcess:
    options = {'dpi': 100, 'MSD': False, '3D': True, 'Len': False}

    def __init__(self, simulation, options: dict = None):
        self.simulation = simulation
        if options:
            self.options = options

        path_to_folder = Path(simulation)
        path_to_data = path_to_folder / 'which_where_when.txt'
        path_to_positions = path_to_folder / 'positions.xyz'
        save_path = path_to_folder / 'paths'
        self.save_path = save_path
        print('Save in:', save_path)
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        with open(path_to_data) as file_in:
            www = file_in.readlines()

        # which where (delta Energy) when
        www = [line.split('\t') for line in www]
        www = [[int(line[0]), int(line[1]), float(line[2]), float(line[3])] for line in www]
        if options['Len']:
            www = www[:options['Len']]

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

        self.bi_base_positions = bi_base_positions
        self.y_base_positions = y_base_positions
        self.o_base_positions = o_base_positions

        o_paths = []
        for i in o_base_positions:
            o_paths.append([i])

        step = 0
        hist_jumps = []
        field = []
        time = []
        msd = np.zeros((100, 2))
        print("Calculating oxygen paths")
        if options['MSD']:
            msd = np.zeros((len(www), 2))
            for which, where, delta_energy, when in tqdm.tqdm(www):
                o_paths[which].append(o_paths[which][-1]+jump[where])
                step_msd = 0
                for index in range(o_base_positions.shape[0]):
                    step_msd += pow(o_base_positions[index][0] - o_paths[index][-1][0], 2) +\
                                pow(o_base_positions[index][1] - o_paths[index][-1][1], 2) +\
                                pow(o_base_positions[index][2] - o_paths[index][-1][2], 2)
                step_msd /= o_base_positions.shape[0]
                msd[step, 0] = when
                msd[step, 1] = step_msd

                for index in range(len(o_base_positions)):
                    if not index == which:
                        o_paths[index].append(o_paths[index][-1])

                hist_jumps.append(where)
                field.append([when, delta_energy])
                time.append(when)
                step += 1
        else:
            for which, where, delta_energy, when in tqdm.tqdm(www):
                o_paths[which].append(o_paths[which][-1] + jump[where])

                hist_jumps.append(where)
                field.append([when, delta_energy])
                time.append(when)
                step += 1

        hist_jumps = np.array(hist_jumps)
        field = np.array(field)
        time = np.array(time)
        o_paths = [np.array(i) for i in o_paths]

        # Histogram
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111)
        ax.set_xlabel('Direction')
        ax.set_ylabel('Count')
        ax.hist(hist_jumps, bins=11)
        plt.savefig(os.path.join(save_path, "Jumps.png"), dpi=options['dpi'])

        # Field Time (delta Time)
        if options['MSD']:
            self.plot_line(save_file=save_path / 'MSD.png',
                           x=msd[:, 0], y=msd[:, 1],
                           x_label='Time [ps]',
                           y_label='MSD')

        self.plot_line(save_file=save_path / 'Field.png',
                       x=field[:, 0],
                       y=field[:, 1],
                       x_label='Time [ps]',
                       y_label='Field [eV]')

        self.plot_line(save_file=save_path / 'Time.png',
                       x=range(time.shape[0]),
                       y=time,
                       x_label='Step',
                       y_label='Time')

        self.plot_line(save_file=save_path / 'delta_Time.png',
                       x=range(time.shape[0]-1),
                       y=np.diff(time),
                       x_label='Step',
                       y_label='Time')

        # Paths
        if self.options['3D']:
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
                plt.savefig(os.path.join(save_path, "{}.png".format(i)), dpi=self.options['dpi'])

        timed_heat_map = TimeHeatMap(load_data_path=path_to_folder, options=self.options)
        timed_heat_map.process_data()

    def plot_line(self, save_file, x, y, x_label, y_label, x_size=8, y_size=6):
        _fig = plt.figure(figsize=(x_size, y_size))
        _ax = _fig.add_subplot(111)
        _ax.set_xlabel(x_label)
        _ax.set_ylabel(y_label)
        _ax.plot(x, y)
        plt.savefig(save_file, dpi=self.options['dpi'], bbox_inches='tight')

    def jmol_simulation(self, o_paths):
        # Jmol simulation
        print("Calculating Jmol simulation frames")
        with open(os.path.join(self.save_path, "simulation.xyz"), 'w') as file_out:
            atom_number = len(self.bi_base_positions) + len(self.y_base_positions) + len(self.o_base_positions)
            for step in tqdm.tqdm(range(1*10**3)):
                file_out.write("{}\n\n".format(atom_number))
                for pos in self.bi_base_positions:
                    file_out.write("Bi")
                    for r in pos:
                        file_out.write("\t{}".format(r))
                    file_out.write("\n")
                for pos in self.y_base_positions:
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


if __name__ == '__main__':
    """
    simulations = ['D:/KMC_data/data_2019_07_24/amplitude/30_7_7_random_01',
                   'D:/KMC_data/data_2019_07_24/amplitude/30_7_7_random_04',
                   'D:/KMC_data/data_2019_07_24/freq/30_7_7_random_01',
                   'D:/KMC_data/data_2019_07_24/freq/30_7_7_random_0025']
    """
    simulations = ['D:/KMC_data/data_2019_08_21/30_9_9_random_1_a',
                   'D:/KMC_data/data_2019_08_21/30_9_9_random_2_a',
                   'D:/KMC_data/data_2019_08_21/30_9_9_random_3_a',
                   'D:/KMC_data/data_2019_08_21/30_9_9_random_4_a',
                   'D:/KMC_data/data_2019_08_21/30_9_9_random_1_b',
                   'D:/KMC_data/data_2019_08_21/30_9_9_random_2_b',
                   'D:/KMC_data/data_2019_08_21/30_9_9_random_3_b',
                   'D:/KMC_data/data_2019_08_21/30_9_9_random_4_b'
                   'D:/KMC_data/data_2019_08_21/30_9_9_random_1_c',
                   'D:/KMC_data/data_2019_08_21/30_9_9_random_2_c',
                   'D:/KMC_data/data_2019_08_21/30_9_9_random_3_c',
                   'D:/KMC_data/data_2019_08_21/30_9_9_random_4_c']

    for idx in simulations:
        DataProcess(idx, {'dpi': 100, 'MSD': False, 'Len': False, '3D': False})
