import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


class TimeHeatMap:
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
            mean_jumps = [(idx*10.0, i.mean()) for idx, i in enumerate(jumps_heat_map_list)]
            mean_jumps = np.array(mean_jumps)
            self.plot_line(save_file=self.save_data_path / 'timed_jumps.png',
                           x=mean_jumps[:, 0],
                           y=mean_jumps[:, 1],
                           x_label='Time [ps]',
                           y_label='Jumps [au]')

    @staticmethod
    def _timed_mean_heat_map(heat_map_list):
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

    @staticmethod
    def _timed_jumps(heat_map_list):
        jumps_heat_map_list = []
        print('Calculating jumps in time')
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

    @staticmethod
    def plot_line(save_file, x, y,  x_label, y_label, x_size=8, y_size=6, dpi=100):
        _fig = plt.figure(figsize=(x_size, y_size))
        _ax = _fig.add_subplot(111)
        _ax.set_xlabel(x_label)
        _ax.set_ylabel(y_label)
        _ax.plot(x, y)
        plt.savefig(save_file, dpi=dpi)
        plt.close(_fig)


if __name__ == '__main__':
    data_path = Path('C:/Users/barja/source/repos/KMC/KMC/KMC_data/30_7_7_random/heat_map')
    hm = TimeHeatMap(data_path)
    hm.process_data(['jumps'])
