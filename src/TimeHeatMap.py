import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm


class TimeHeatMap:
    load_data_path = None
    save_data_path = None
    file_list = None

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
            self._timed_mean_heat_map(heat_map_list)
        if 'jumps' in mode:
            jumps_heat_map_list = self._timed_jumps(heat_map_list)
            index = 0
            mead_jumps = [i.mean() for i in jumps_heat_map_list]
            plt.plot(mead_jumps)
            plt.show()

    def _timed_mean_heat_map(self, heat_map_list):
        x_heat_map_list = []
        for heat_map in heat_map_list:
            dim = np.max(heat_map[:, 0])+1, np.max(heat_map[:, 1])+1, np.max(heat_map[:, 2])+1
            _heat_map = np.zeros(dim)
            for pos in heat_map:
                _heat_map[pos[0], pos[1], pos[2]] = pos[3]

            _x_heat_map = _heat_map.sum(axis=(1, 2))
            x_heat_map_list.append(np.rot90(_x_heat_map.reshape((len(_x_heat_map), 1))))

        return [x_heat_map_list[i+1]-x_heat_map_list[i]
                for i in range(len(x_heat_map_list)-1)]

    def _timed_jumps(self, heat_map_list):
        jumps_heat_map_list = []
        for heat_map in tqdm(heat_map_list):
            max_x = np.max(heat_map[:, 0])
            dim = np.max(heat_map[:, 1])+1, np.max(heat_map[:, 2])+1
            _heat_map = np.zeros(dim)
            for pos in heat_map:
                if pos[0] == max_x:
                    _heat_map[pos[1], pos[2]] = pos[3]
            jumps_heat_map_list.append(_heat_map)
        return [jumps_heat_map_list[i+1]-jumps_heat_map_list[i]
                for i in range(len(jumps_heat_map_list)-1)]

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


if __name__ == '__main__':
    data_path = Path('/home/blue/Documents/source/KMC/KMC_data/3_3_3_random/heat_map')
    hm = TimeHeatMap(data_path)
    hm.process_data('jumps')
