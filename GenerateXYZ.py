import sys
import os
import random
import numpy as np
import click


class GenerateXYZ:
    def __init__(self, cells, path_out):

        self.cell_size = 1.0
        self.path_out = os.path.join(path_out, 'positions.xyz')

        self.kations = np.zeros(2*np.array(cells) + 1).astype(np.int)
        self.anions = np.zeros(2*np.array(cells)).astype(np.int)

        self.positions = {'Bi': [], 'Y': [], 'O': []}

        self.Bi = 0
        self.Y = 0
        self.O = 0

    def save_positions(self):
        with open(self.path_out, 'w') as file_out:
            file_out.write("{}\n\n".format(self.Bi+self.Y+self.O))
            for atom_type in self.positions:
                for atom in self.positions[atom_type]:
                    file_out.write("{}".format(atom_type))
                    for r in atom:
                        file_out.write("\t{}".format(r))
                    file_out.write("\n")


    def generate_sphere(self, radius):
        center = np.floor(np.array(self.kations.shape)*self.cell_size/2)

        to_change = True
        for index, kation in np.ndenumerate(self.kations):
            position = index[0] * self.cell_size, index[1] * self.cell_size, index[2] * self.cell_size
            if to_change:
                test = np.sqrt((position[0]-center[0])**2 +
                               (position[1]-center[1])**2 +
                               (position[2]-center[2])**2)
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
            position = (index[0]*self.cell_size + self.cell_size * 0.5,
                        index[1]*self.cell_size + self.cell_size * 0.5,
                        index[2]*self.cell_size + self.cell_size * 0.5)

            if random.uniform(0, 1) > 0.75:
                self.positions['O'].append(position)
                self.O += 1

        self.save_positions()


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

        self.anions[:, :, -1] = self.anions[:, :, +1]
        self.anions[:, :, +0] = self.anions[:, :, -2]
        self.anions[:, -1, :] = self.anions[:, +1, :]
        self.anions[:, +0, :] = self.anions[:, -2, :]
        self.anions[-1, :, :] = self.anions[+1, :, :]
        self.anions[+0, :, :] = self.anions[-2, :, :]

        for index, atom in np.ndenumerate(self.anions):
            position = (index[0]*self.cell_size + self.cell_size * 0.5,
                        index[1]*self.cell_size + self.cell_size * 0.5,
                        index[2]*self.cell_size + self.cell_size * 0.5)
            if atom == 1:
                self.positions['O'].append(position)
                self.O += 1

        self.save_positions()

    def generate_plane(self, thickness):
        center = np.floor(np.array(self.kations.shape)*self.cell_size/3)

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
            position = (index[0]*self.cell_size + self.cell_size * 0.5,
                        index[1]*self.cell_size + self.cell_size * 0.5,
                        index[2]*self.cell_size + self.cell_size * 0.5)

            if random.uniform(0, 1) > 0.75:
                self.positions['O'].append(position)
                self.O += 1

        self.save_positions()


if __name__ == "__main__":
    @click.command()
    @click.option('--cells', nargs=3, type=int, prompt="Number of cells x y z", help="Number of cells in system.")
    @click.option('--structure', prompt="Type of structure (random, sphere, plane): ", help="Type of structure (random, sphere, plane).")
    def main(cells, structure):
        path_out = str(cells[0])+'_'+str(cells[1])+'_'+str(cells[2])+'_'+structure
        if not os.path.exists(path_out):
            os.makedirs(path_out)
        if not os.path.exists(os.path.join(path_out, 'heat_map')):
                    os.makedirs(os.path.join(path_out, 'heat_map'))
        if structure == 'random':
            GenerateXYZ(cells, path_out).generate_random()
        if structure == 'sphere':
            GenerateXYZ(cells, path_out).generate_sphere(5)
        if structure == 'plane':
            GenerateXYZ(cells, path_out).generate_plane(1)
    main()
