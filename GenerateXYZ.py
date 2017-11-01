import sys
import random
import numpy as np
import click


class GenerateXYZ:
    def __init__(self, cells, cell_size, file_out_name):

        self.cells = cells
        self.cell_size = cell_size
        self.file_out_name = file_out_name

        self.kations_size = 2*cells + 1
        self.kations = np.zeros((self.kations_size, self.kations_size, self.kations_size)).astype(np.int)

        self.anion_size = 2*cells
        self.anions = np.zeros((self.anion_size, self.anion_size, self.anion_size)).astype(np.int)

        self.positions = {'Bi': [], 'Y': [], 'O': []}

        self.Bi = 0
        self.Y = 0
        self.O = 0

    def generate_sphere(self, radius):
        center = (self.kations_size*self.cell_size/2,
                  self.kations_size*self.cell_size/2,
                  self.kations_size*self.cell_size/2)

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

        with open(self.file_out_name, 'w') as file_out:
            file_out.write("{}\n\n".format(self.Bi+self.Y+self.O))
            for atom_type in self.positions:
                for atom in self.positions[atom_type]:
                    file_out.write("{}".format(atom_type))
                    for r in atom:
                        file_out.write("\t{}".format(r))
                    file_out.write("\n")

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

        with open(self.file_out_name, 'w') as file_out:
            file_out.write("{}\n\n".format(self.Bi+self.Y+self.O))
            for atom_type in self.positions:
                for atom in self.positions[atom_type]:
                    file_out.write("{}".format(atom_type))
                    for r in atom:
                        file_out.write("\t{}".format(r))
                    file_out.write("\n")

    def generate_plane(self, thickness):
        center = (self.kations_size*self.cell_size/2,
                  self.kations_size*self.cell_size/2,
                  self.kations_size*self.cell_size/2)

        to_change = True
        for index, kation in np.ndenumerate(self.kations):
            position = index[0] * self.cell_size, index[1] * self.cell_size, index[2] * self.cell_size
            if to_change:
                test = np.abs(position[0]-center[0])
                if test > thickness:
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

        with open(self.file_out_name, 'w') as file_out:
            file_out.write("{}\n\n".format(self.Bi+self.Y+self.O))
            for atom_type in self.positions:
                for atom in self.positions[atom_type]:
                    file_out.write("{}".format(atom_type))
                    for r in atom:
                        file_out.write("\t{}".format(r))
                    file_out.write("\n")

    def genetate_ox_position(self):
        return [(index[0]*self.cell_size + self.cell_size * 0.5,
            index[1]*self.cell_size + self.cell_size * 0.5,
            index[2]*self.cell_size + self.cell_size * 0.5)
            for index, anion in np.ndenumerate(self.anions)]



if __name__ == "__main__":
    """
    @click.command()
    @click.option('--cells',prompt="Number of cells", help="Number of cells in system.")
    @click.option('--cell_size',prompt="Size of cell", help="Size of cell in system.")
    @click.option('--file_out_name',prompt="Name of files out", help="Name of files out")
    def main(cells, cell_size, file_out_name):
        GenerateXYZ(int(cells), int(cell_size), file_out_name).generate_random()
    main()
    """
    pos = GenerateXYZ(7, 1, 'test_heat_map').genetate_ox_position()
    print(pos[0]%(0.5, 0,5, 0,5))

