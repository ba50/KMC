import sys
import os
import csv
import numpy as np
from vispy import app, scene, visuals
import click

from display_lines import *


class OxygenPath:
    def __init__(self, start_position_path, when_which_where_path, data_path='.', atom_number=None, steps=None):
        if atom_number:
            atom_number = int(atom_number)

        if steps:
            self.steps = int(steps)
        else:
            dat_path =  os.path.join(data_path, os.path.splitext(os.path.basename(when_which_where_path))[0]+'.dat')
            self.steps = sum(1 for line in open(dat_path)) 

        self.start_positions = []
        with open(start_position_path) as file:
            next(file)
            next(file)
            for line in file:
                if atom_number:
                    if len(self.start_positions) >= atom_number:
                        break

                words = line.split()
                if words[0] == 'O':
                    self.start_positions.append((float(words[1]), float(words[2]), float(words[3])))
                
        self.start_positions = np.array(self.start_positions).astype(np.float32)
        self.direction_vector = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
        
        self.label = os.path.splitext(os.path.basename(when_which_where_path))[0].replace('when_which_where_', '')
        self.atom_number = self.start_positions.shape[0]

        dimensions = self.start_positions.shape[0], self.steps, self.start_positions.shape[1]
        self.when_which_where = np.memmap(when_which_where_path, dtype='float32', mode='r+', shape=(self.steps, 3))

        data_file_name= 'paths_'+self.label+'.bin'
        if not os.path.exists(os.path.join(data_path, data_file_name)):
            print("Generate path file...", end='')
            self.paths = np.memmap(os.path.join(data_path, data_file_name), dtype='float32', mode='w+', shape=dimensions)

            for i in range(0, dimensions[0]):
                self.paths[i, 0, :] = self.start_positions[i]


            for index in np.ndindex(self.when_which_where.shape[0]-1):
                for i in range(0, dimensions[0]):
                    self.paths[i, index[0]+1, :] += self.paths[i, index[0], :]
                    if self.when_which_where[index[0], 1] == i:
                        self.paths[i, index[0]+1, 0] += self.direction_vector[int(self.when_which_where[index[0], 2])][0]
                        self.paths[i, index[0]+1, 1] += self.direction_vector[int(self.when_which_where[index[0], 2])][1]
                        self.paths[i, index[0]+1, 2] += self.direction_vector[int(self.when_which_where[index[0], 2])][2]
            print("Ok")
        else:
            print("Loading path file:", data_file_name, end=' ')
            self.paths = np.memmap(os.path.join(data_path, data_file_name), dtype='float32', mode='r+', shape=dimensions)
            print("Ok")

    def plot_test(self):
        c = Canvas(self.paths)
        app.run()

    def plot(self):
        plot = scene.visuals.create_visual_node(visuals.LineVisual)
        canvas = scene.SceneCanvas(keys='interactive', title='Oxygen Paths', show=True)
        canvas.bgcolor = (1, 1, 1, 1)
        view = canvas.central_widget.add_view()
        view.camera = scene.cameras.FlyCamera(parent=view.scene, fov=60)

        r = np.random.uniform(0, 1)
        g = np.random.uniform(0, 1)
        b = np.random.uniform(0, 1)

        for i in range(self.paths.shape[0]):
            plot(self.paths[i], color=(r, g, b, 1), parent=view.scene)

        app.run()

if __name__ == "__main__":
    @click.command()
    @click.option('--start_path',prompt="Start position file: ", help="Start position file.")
    @click.option('--when_path', prompt="Symulaion file: ", help="When which where file.")
    @click.option('--data_path', default='.', help="Where data are.")
    @click.option('--atom_number', default=None, help="Number of atoms.")
    @click.option('--steps', default=None, help="Number of steps.")
    def main(start_path, when_path, data_path, atom_number, steps):
        path = OxygenPath(start_path, when_path, data_path, atom_number, steps)
        path.plot_test()
                
    main()

