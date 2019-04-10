import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class TimeHeatMap:
    load_data_path = None
    save_data_path = None
    file_list = None
    x_heat_map_list = []
    frame = 0
    im = None

    def __init__(self, load_data_path: Path, save_data_path: Path = None):
        """
        Load data from HeatMap.
        :param load_data_path:
        :param save_data_path:
        """
        self.load_data_path = load_data_path
        self.save_data_path = save_data_path

        if not self.save_data_path:
            self.save_data_path = Path(self.load_data_path.parent, 'heat_map_plots')

        if not self.save_data_path.exists():
            self.save_data_path.mkdir()

        self.file_list = sorted(self.load_data_path.glob('*.dat'),
                                key=lambda i: int(os.path.splitext(os.path.basename(i))[0]))

    def process_data(self, mode: list = None):
        heat_map_list = []
        for file_in in self.file_list:
            array = []
            with file_in.open() as heat_map_file:
                for line in heat_map_file:
                    array.append([int(word) for word in line.split()])

            heat_map_list.append(np.array(array))
        if 'mean' in mode:
            self.timed_mean_heat_map(heat_map_list)
        if 'jumps' in mode:
            pass

    def timed_mean_heat_map(self, heat_map_list):
        for heat_map in heat_map_list:
            dim = np.max(heat_map[:, 0])+1, np.max(heat_map[:, 1])+1, np.max(heat_map[:, 2])+1
            _heat_map = np.zeros(dim)
            for pos in heat_map:
                _heat_map[pos[0], pos[1], pos[2]] = pos[3]

            _x_heat_map = _heat_map.sum(axis=(1, 2))
            self.x_heat_map_list.append(np.rot90(_x_heat_map.reshape((len(_x_heat_map), 1))))

        self.x_heat_map_list = [self.x_heat_map_list[i+1]-self.x_heat_map_list[i]
                                for i in range(len(self.x_heat_map_list)-1)]

    def _make_image(self):
        self.im = plt.imshow(self.x_heat_map_list[0],
                             interpolation='none',
                             extent=[0, self.x_heat_map_list[0].shape[1], 0, 32],
                             animated=True)
        plt.title(str(self.frame*10)+'[ps]')
        plt.colorbar()

    def plot_layer_in_time(self, layer):
        cs_font = {'size': 16}
        fig = plt.figure()
        y = [l[0][layer] for l in self.x_heat_map_list]
        y = np.array(y)
        x = np.arange(0, len(self.x_heat_map_list)*10, 10)

        plt.plot(x, y)
        plt.xlabel("Time [ps]", **cs_font)
        plt.ylabel("Jumps [au]", **cs_font)
        plt.show()

    def plot(self):
        fig = plt.figure()
        self._make_image()
        fig.colorbar(self.im, orientation='horizontal')
        self.frame = 0
        animation.FuncAnimation(fig, self.updatefig, frames=5, interval=1000, blit=True)
        plt.show()

    def save_animation(self):
        fig = plt.figure()
        self._make_image()
        ani = animation.FuncAnimation(fig, self.updatefig, frames=len(self.x_heat_map_list), blit=True)
        ani.save(Path(self.save_data_path, 'heat_map.mp4'), fps=5, dpi=200)

    def save_heatmap(self):
        fig = plt.figure()
        self._make_image()
        for idx, _ in enumerate(self.x_heat_map_list):
            fig.savefig(Path(self.save_data_path, 'heatmap_' + str(idx) + '.png'))
            self.updatefig()

    def updatefig(self, *args):
        self.frame += 1
        if self.frame >= len(self.x_heat_map_list):
            self.frame = 0
        plt.title(str(self.frame*10)+'[ps]')
        self.im.set_array(self.x_heat_map_list[self.frame])


if __name__ == '__main__':
    data_path = Path('D:/KMC_data/3_3_3_random/heat_map')
    hm = TimeHeatMap(data_path)
    hm.process_data()
