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
        self.load_data_path = load_data_path / 'heat_map'
        self.save_data_path = save_data_path

        if not self.save_data_path:
            self.save_data_path = self.load_data_path.parent / 'heat_map_plots'

        if not self.save_data_path.exists():
            self.save_data_path.mkdir(parents=True, exist_ok=True)

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

            jumps_ll_heat_map_list = jumps_heat_map_list[0]
            jumps_lr_heat_map_list = jumps_heat_map_list[1]
            jumps_cl_heat_map_list = jumps_heat_map_list[2]
            jumps_cr_heat_map_list = jumps_heat_map_list[3]
            jumps_rl_heat_map_list = jumps_heat_map_list[4]
            jumps_rr_heat_map_list = jumps_heat_map_list[5]

            mean_ll_jumps = [(idx*10.0, i.mean()) for idx, i in enumerate(jumps_ll_heat_map_list)]
            mean_lr_jumps = [(idx*10.0, i.mean()) for idx, i in enumerate(jumps_lr_heat_map_list)]
            mean_cl_jumps = [(idx*10.0, i.mean()) for idx, i in enumerate(jumps_cl_heat_map_list)]
            mean_cr_jumps = [(idx*10.0, i.mean()) for idx, i in enumerate(jumps_cr_heat_map_list)]
            mean_rl_jumps = [(idx*10.0, i.mean()) for idx, i in enumerate(jumps_rl_heat_map_list)]
            mean_rr_jumps = [(idx*10.0, i.mean()) for idx, i in enumerate(jumps_rr_heat_map_list)]

            mean_ll_jumps = np.array(mean_ll_jumps)
            mean_lr_jumps = np.array(mean_lr_jumps)
            mean_cl_jumps = np.array(mean_cl_jumps)
            mean_cr_jumps = np.array(mean_cr_jumps)
            mean_rl_jumps = np.array(mean_rl_jumps)
            mean_rr_jumps = np.array(mean_rr_jumps)

            self.plot_line(save_file=self.save_data_path / 'timed_jumps_left_contact_left_jump.png',
                           x=mean_ll_jumps[:, 0],
                           y=mean_ll_jumps[:, 1],
                           x_label='Time [ps]',
                           y_label='Jumps [au]')
            self.plot_line(save_file=self.save_data_path / 'timed_jumps_left_contact_right_jump.png',
                           x=mean_lr_jumps[:, 0],
                           y=mean_lr_jumps[:, 1],
                           x_label='Time [ps]',
                           y_label='Jumps [au]')

            self.plot_line(save_file=self.save_data_path / 'timed_jumps_center_contact_left_jump.png',
                           x=mean_cl_jumps[:, 0],
                           y=mean_cl_jumps[:, 1],
                           x_label='Time [ps]',
                           y_label='Jumps [au]')
            self.plot_line(save_file=self.save_data_path / 'timed_jumps_center_contact_right_jump.png',
                           x=mean_cr_jumps[:, 0],
                           y=mean_cr_jumps[:, 1],
                           x_label='Time [ps]',
                           y_label='Jumps [au]')

            self.plot_line(save_file=self.save_data_path / 'timed_jumps_right_contact_left_jump.png',
                           x=mean_rl_jumps[:, 0],
                           y=mean_rl_jumps[:, 1],
                           x_label='Time [ps]',
                           y_label='Jumps [au]')
            self.plot_line(save_file=self.save_data_path / 'timed_jumps_right_contact_right_jump.png',
                           x=mean_rr_jumps[:, 0],
                           y=mean_rr_jumps[:, 1],
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
        # l - left
        # r - right
        # c - center
        # ll - left contact left jump
        jumps_ll_heat_map_list = []
        jumps_lr_heat_map_list = []
        jumps_cl_heat_map_list = []
        jumps_cr_heat_map_list = []
        jumps_rl_heat_map_list = []
        jumps_rr_heat_map_list = []
        print('Calculating jumps in time')
        for heat_map in tqdm(heat_map_list):
            max_x = np.max(heat_map[:, 0])
            dim = np.max(heat_map[:, 1])+1, np.max(heat_map[:, 2])+1
            _ll_heat_map = np.zeros(dim)
            _lr_heat_map = np.zeros(dim)
            _cl_heat_map = np.zeros(dim)
            _cr_heat_map = np.zeros(dim)
            _rl_heat_map = np.zeros(dim)
            _rr_heat_map = np.zeros(dim)
            for pos in heat_map:
                if pos[0] == 1:
                    if pos[3] == 1:
                        _ll_heat_map[pos[1], pos[2]] = pos[4]
                    if pos[3] == 0:
                        _lr_heat_map[pos[1], pos[2]] = pos[4]
                if pos[0] == max_x // 2:
                    if pos[3] == 1:
                        _cl_heat_map[pos[1], pos[2]] = pos[4]
                    if pos[3] == 0:
                        _cr_heat_map[pos[1], pos[2]] = pos[4]
                if pos[0] == max_x-1:
                    if pos[3] == 1:
                        _rl_heat_map[pos[1], pos[2]] = pos[4]
                    if pos[3] == 0:
                        _rr_heat_map[pos[1], pos[2]] = pos[4]

            jumps_ll_heat_map_list.append(_ll_heat_map)
            jumps_lr_heat_map_list.append(_lr_heat_map)
            jumps_cl_heat_map_list.append(_cl_heat_map)
            jumps_cr_heat_map_list.append(_cr_heat_map)
            jumps_rl_heat_map_list.append(_rl_heat_map)
            jumps_rr_heat_map_list.append(_rr_heat_map)

        delta_ll_heat_map = [jumps_ll_heat_map_list[i+1]-jumps_ll_heat_map_list[i]
                             for i in range(len(jumps_ll_heat_map_list)-1)]
        delta_lr_heat_map = [jumps_lr_heat_map_list[i+1]-jumps_lr_heat_map_list[i]
                             for i in range(len(jumps_lr_heat_map_list)-1)]
        delta_cl_heat_map = [jumps_cl_heat_map_list[i+1]-jumps_cl_heat_map_list[i]
                             for i in range(len(jumps_cl_heat_map_list)-1)]
        delta_cr_heat_map = [jumps_cr_heat_map_list[i+1]-jumps_cr_heat_map_list[i]
                             for i in range(len(jumps_cr_heat_map_list)-1)]
        delta_rl_heat_map = [jumps_rl_heat_map_list[i+1]-jumps_rl_heat_map_list[i]
                             for i in range(len(jumps_rl_heat_map_list)-1)]
        delta_rr_heat_map = [jumps_rr_heat_map_list[i+1]-jumps_rr_heat_map_list[i]
                             for i in range(len(jumps_rr_heat_map_list)-1)]

        out = (delta_ll_heat_map,
               delta_lr_heat_map,
               delta_cl_heat_map,
               delta_cr_heat_map,
               delta_rl_heat_map,
               delta_rr_heat_map)

        return out

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
    data_path = Path('D:/KMC_data/tests/15_7_7_random')
    hm = TimeHeatMap(data_path)
    hm.process_data(['jumps'])
