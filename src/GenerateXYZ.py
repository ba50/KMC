import random
from pathlib import Path

import numpy as np
import pandas as pd


class GenerateXYZ:
    def __init__(self, cells: tuple):

        self.cell_size = 1.0

        self.kations = np.zeros(2 * np.array(cells) + 1).astype(np.int)
        self.anions = np.zeros(2 * np.array(cells)).astype(np.int)

        self.positions = {'Bi': [], 'Y': [], 'O': []}

        self.Bi = 0
        self.Y = 0
        self.O = 0

    def save_positions(self, save_path: Path):
        if len(self.positions['Bi']) > 0:
            with save_path.open('w') as file_out:
                file_out.write("{}\n\n".format(self.Bi + self.Y + self.O))
                for atom_type in self.positions:
                    for atom in self.positions[atom_type]:
                        file_out.write("{}".format(atom_type))
                        for r in atom:
                            file_out.write("\t{}".format(r))
                        file_out.write("\n")
        else:
            print('No structure')

    def generate_sphere(self, radius):
        center = np.floor(np.array(self.kations.shape) * self.cell_size / 2)

        to_change = True
        for index, kation in np.ndenumerate(self.kations):
            position = index[0] * self.cell_size, index[1] * self.cell_size, index[2] * self.cell_size
            if to_change:
                test = np.sqrt((position[0] - center[0]) ** 2 +
                               (position[1] - center[1]) ** 2 +
                               (position[2] - center[2]) ** 2)
                if test > radius:
                    self.positions['Bi'].append(position)
                    self.Bi += 1
                else:
                    self.positions['Y'].append(position)
                    self.Y += 1
                to_change = False
            else:
                to_change = True

        for index, anion in np.ndenumerate(self.anions):
            position = (index[0] * self.cell_size + self.cell_size * 0.5,
                        index[1] * self.cell_size + self.cell_size * 0.5,
                        index[2] * self.cell_size + self.cell_size * 0.5)

            if random.uniform(0, 1) > 0.75:
                self.positions['O'].append(position)
                self.O += 1

    def generate_random(self):
        to_change = True
        for index in np.ndindex(self.kations.shape):
            if to_change:
                if random.uniform(0, 1) > .25:
                    self.kations[index] = 1
                else:
                    self.kations[index] = 2
                to_change = False
            else:
                to_change = True

        self.kations[:, :, 0] = self.kations[:, :, -1]
        self.kations[:, 0, :] = self.kations[:, -1, :]
        self.kations[0, :, :] = self.kations[-1, :, :]

        for index, atom in np.ndenumerate(self.kations):
            position = index[0] * self.cell_size, index[1] * self.cell_size, index[2] * self.cell_size
            if atom == 1:
                self.positions['Bi'].append(position)
                self.Bi += 1
            if atom == 2:
                self.positions['Y'].append(position)
                self.Y += 1

        for index in np.ndindex(self.anions.shape):
            if random.uniform(0, 1) > 0.75:
                self.anions[index] = 1

        for index, atom in np.ndenumerate(self.anions):
            position = (index[0] * self.cell_size + self.cell_size * 0.5,
                        index[1] * self.cell_size + self.cell_size * 0.5,
                        index[2] * self.cell_size + self.cell_size * 0.5)
            if atom == 1:
                self.positions['O'].append(position)
                self.O += 1

    def generate_plane(self, thickness):
        center = np.floor(np.array(self.kations.shape) * self.cell_size / 3)

        to_change = True
        for index, kation in np.ndenumerate(self.kations):
            position = index[0] * self.cell_size, index[1] * self.cell_size, index[2] * self.cell_size
            if to_change:
                if not position[0] % 10 == 0:
                    self.positions['Bi'].append(position)
                    self.Bi += 1
                else:
                    self.positions['Y'].append(position)
                    self.Y += 1
                to_change = False
            else:
                to_change = True

        for index, anion in np.ndenumerate(self.anions):
            position = (index[0] * self.cell_size + self.cell_size * 0.5,
                        index[1] * self.cell_size + self.cell_size * 0.5,
                        index[2] * self.cell_size + self.cell_size * 0.5)

            if random.uniform(0, 1) > 0.75:
                self.positions['O'].append(position)
                self.O += 1

    def generate_from_array(self, bi, y, o):
        for pos in bi:
            self.positions['Bi'].append(pos)
            self.Bi += 1
        for pos in y:
            self.positions['Y'].append(pos)
            self.Y += 1
        for pos in o:
            self.positions['O'].append(pos)
            self.O += 1

    @staticmethod
    def read_file(simulation_path):
        bi_base_positions = []
        y_base_positions = []
        o_base_positions = []
        with simulation_path.open('r') as file_in:
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

        return np.array(bi_base_positions), np.array(y_base_positions), np.array(o_base_positions)

    @staticmethod
    def read_frames_dataframe(data_path: Path, num_frames=None) -> (int, pd.DataFrame):
        with data_path.open('r') as f_in:
            n_atoms = f_in.readline()
            n_atoms = int(n_atoms)

        data_in = pd.read_table(data_path, names=['atom', 'x', 'y', 'z'])
        index_list = data_in[data_in['x'].isnull()].index

        frame = data_in[1:n_atoms+1].reset_index(drop=True)
        frame_index = [{'frame': 0} for _ in range(n_atoms)]
        frame_index = pd.DataFrame(frame_index)
        frames = pd.concat([frame, frame_index], axis=1)

        if num_frames:
            index_list = index_list[:num_frames]

        for index in index_list[1:]:
            frame = data_in[index+1:n_atoms+index+1].reset_index(drop=True)
            frame_index = [{'frame': index} for _ in range(n_atoms)]
            frame_index = pd.DataFrame(frame_index)
            frame = pd.concat([frame, frame_index], axis=1)
            frames = frames.append(frame)

        return n_atoms, frames

    @staticmethod
    def write_frames_from_dataframe(save_path: Path, data: pd.DataFrame, num_atoms):
        index_list = data['frame'].unique()
        with save_path.open('w') as f_out:
            for index in index_list:
                frame = data[data['frame'] == index]
                f_out.write(f"{num_atoms}\n\n")
                for _, row in frame.iterrows():
                    f_out.write(f"{row.atom}\t{int(row.x)}\t{int(row.y)}\t{int(row.z)}\n")
