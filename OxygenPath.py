import sys
import os
import csv
import numpy as np
from vispy import app, scene, visuals

steps = 100000  # sum(1 for line in open('when_which_where.dat'))
atom_number = 10


class OxygenPath:
    def __init__(self, start_position_path):
        self.start_positions = []
        with open(start_position_path) as file:
            next(file)
            next(file)
            for line in file:
                words = line.split()
                if words[0] == 'O':
                    self.start_positions.append((float(words[1]), float(words[2]), float(words[3])))

        self.start_positions = np.array(self.start_positions).astype(np.float32)
        self.direction_vector = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
        self.paths = None

    def generate_data(self, atom_ids, when_which_where_path):

        dimensions = atom_number, self.start_positions.shape[1], steps
        paths = np.memmap('paths.bin', dtype='float32', mode='w+', shape=dimensions)

        for i in range(0, atom_number):
            paths[i, :, 0] = self.start_positions[i]

        when_which_where = np.memmap(when_which_where_path, dtype='float32', mode='r+', shape=(steps, 3))

        for index in np.ndindex(when_which_where.shape[0]-1):
            for i in range(0, dimensions[0]):
                paths[i, :, index[0]+1] += paths[i, :, index[0]]
                if when_which_where[index[0], 1] == i:
                    paths[i, 0, index[0]+1] += self.direction_vector[int(when_which_where[index[0], 2])][0]
                    paths[i, 1, index[0]+1] += self.direction_vector[int(when_which_where[index[0], 2])][1]
                    paths[i, 2, index[0]+1] += self.direction_vector[int(when_which_where[index[0], 2])][2]

        self.paths = []
        for i in atom_ids:
            self.paths.append(np.rot90(np.memmap('paths.bin',
                                                 dtype='float32',
                                                 mode='r',
                                                 shape=(1, 3, steps),
                                                 offset=int(i*3*steps*32/8))[0], 3))

    def plot(self):
        plot = scene.visuals.create_visual_node(visuals.LineVisual)
        canvas = scene.SceneCanvas(keys='interactive', title='plot3d', show=True)
        canvas.bgcolor = (1, 1, 1, 1)
        view = canvas.central_widget.add_view()
        view.camera = scene.cameras.FlyCamera(parent=view.scene, fov=60)
        for path in self.paths:
            r = np.random.uniform(0, 1)
            g = np.random.uniform(0, 1)
            b = np.random.uniform(0, 1)
            plot(path, color=(r, g, b, 1), parent=view.scene)
        app.run()

    def make_csv(self, file_name, atom_id):
        with open(file_name, 'w') as file:
            writer = csv.writer(file, delimiter='\t')
            writer.writerows(self.paths[atom_id])


if __name__ == "__main__":
    test = OxygenPath(sys.argv[1])
    test.generate_data(range(0, atom_number), sys.argv[2])
    test.plot()
