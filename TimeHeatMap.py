import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import glob
from sklearn.preprocessing import normalize


class TimeHeatMap:
    x_heat_map_list = []
    im = None
    frame = None
    def __init__(self, path_to_data):
        self.path_to_data = path_to_data

        self.path_out = os.path.join(self.path_to_data, 'data_out')
        if not os.path.exists(self.path_out):
            os.makedirs(self.path_out)

        path_to_data = os.path.join(self.path_to_data, 'heat_map/*')
        path_to_data = sorted(glob.glob(path_to_data), key=lambda i: int(os.path.splitext(os.path.basename(i))[0]))
        for file_in in path_to_data:
            array = []
            with open(file_in) as heat_map_file:
              for line in heat_map_file:
                  array.append([int(word) for word in line.split()])

            array = np.array(array)

            dim = np.max(array[:, 0])+1, np.max(array[:, 1])+1, np.max(array[:, 2])+1
            _heat_map = np.zeros(dim)
            for pos in array:
              _heat_map[pos[0], pos[1], pos[2]] = pos[3]

            _x_heat_map = _heat_map.sum(axis=(1,2))
            self.x_heat_map_list.append(np.rot90(_x_heat_map.reshape((len(_x_heat_map), 1))))

        self.x_heat_map_list = [self.x_heat_map_list[i+1]-self.x_heat_map_list[i] for i in range(len(self.x_heat_map_list)-1)]

    def plot_layer_in_time(self, layer):
        csfont = {'size': 16}
        fig = plt.figure()
        y = [l[0][layer] for l in self.x_heat_map_list]
        y = np.array(y)
        x = np.arange(0, len(self.x_heat_map_list)*10, 10)

        plt.plot(x, y)
        plt.xlabel("Time [ps]", **csfont)
        plt.ylabel("Jumps [au]", **csfont)
        plt.show()

    def plot(self):
        fig = plt.figure()
        self.frame = 0

        self.im = plt.imshow(self.x_heat_map_list[0], interpolation='none', extent=[0, self.x_heat_map_list[0].shape[1],0, 32],  animated=True)
        cbar = fig.colorbar(self.im, orientation='horizontal')
        self.ani = animation.FuncAnimation(fig, self.updatefig, frames=500, interval=1000, blit=True)

        plt.show()

    def save_plot(self):
        fig = plt.figure()
        self.frame = 0

        self.im = plt.imshow(self.x_heat_map_list[0], interpolation='none', extent=[0, self.x_heat_map_list[0].shape[1],0, 32],  animated=True)
        cbar = fig.colorbar(self.im, orientation='horizontal')
        ani = animation.FuncAnimation(fig, self.updatefig, frames=len(self.x_heat_map_list), blit=True)

        ani.save(os.path.join(self.path_out, 'heat_map.mp4'), fps=12, dpi=200)

    def updatefig(self, *args):
        self.frame += 1
        if self.frame >= len(self.x_heat_map_list):
            self.frame=0
        plt.title(str(self.frame*10)+'[ps]')
        self.im.set_array(self.x_heat_map_list[self.frame])
        return self.im,

