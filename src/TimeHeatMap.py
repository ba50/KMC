import os
import h5py
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


class TimeHeatMap:
    options = {'mean': False, "jumps": True, "save_raw": True, 'mean_size': 6}

    def __init__(self, load_data_path: Path, save_data_path: Path = None, options=None):
        """
        Load data from HeatMap.
        :param load_data_path:
        :param save_data_path:
        """
        if options:
            self.options.update(options)

        self.load_data_path = load_data_path / 'heat_map'
        self.save_data_path = save_data_path

        if not self.save_data_path:
            self.save_data_path = self.load_data_path.parent / 'heat_map_plots'

        if not self.save_data_path.exists():
            self.save_data_path.mkdir(parents=True, exist_ok=True)

        self.file_list = sorted(self.load_data_path.glob('*.dat'),
                                key=lambda i: int(os.path.splitext(os.path.basename(i))[0]))

    def process_data(self):
        heat_map_list = []
        print('Loading heat map files...')
        if self.options['Len']:
            for file_in in tqdm(self.file_list[:self.options['Len']]):
                array = []
                with file_in.open() as heat_map_file:
                    for line in heat_map_file:
                        array.append([int(word) for word in line.split()])

                heat_map_list.append(np.array(array))
        else:
            for file_in in tqdm(self.file_list):
                array = []
                with file_in.open() as heat_map_file:
                    for line in heat_map_file:
                        array.append([int(word) for word in line.split()])

                heat_map_list.append(np.array(array))

        if self.options['mean']:
            self._timed_mean_heat_map(heat_map_list)
        if self.options['jumps']:
            jumps_heat_map_list = self._time_jumps(heat_map_list,
                                                   ['left', 'center', 'right'],
                                                   mean_size=self.options['mean_size'])

            jumps_ll_heat_map_list = jumps_heat_map_list['left']['left']
            jumps_lr_heat_map_list = jumps_heat_map_list['left']['right']
            jumps_cl_heat_map_list = jumps_heat_map_list['center']['left']
            jumps_cr_heat_map_list = jumps_heat_map_list['center']['right']
            jumps_rl_heat_map_list = jumps_heat_map_list['right']['left']
            jumps_rr_heat_map_list = jumps_heat_map_list['right']['right']

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

            file_name_h5py = h5py.File(str(self.save_data_path / 'timed_jumps_raw_data.h5'), 'w')

            file_name_h5py.create_dataset('timed_jumps_left_contact_left_jump', data=mean_ll_jumps)
            file_name_h5py.create_dataset('timed_jumps_left_contact_right_jump', data=mean_lr_jumps)
            file_name_h5py.create_dataset('timed_jumps_center_contact_left_jump', data=mean_cl_jumps)
            file_name_h5py.create_dataset('timed_jumps_center_contact_right_jump', data=mean_cr_jumps)
            file_name_h5py.create_dataset('timed_jumps_right_contact_left_jump', data=mean_rl_jumps)
            file_name_h5py.create_dataset('timed_jumps_right_contact_right_jump', data=mean_rr_jumps)

            for key in file_name_h5py.keys():
                self.plot_line(save_file=Path(self.save_data_path, key+'.png'),
                               x=file_name_h5py[key][:, 0],
                               y=file_name_h5py[key][:, 1],
                               x_label='Time [ps]',
                               y_label='Jumps [au]',
                               dpi=self.options['dpi'])

    @staticmethod
    def _time_jumps(heat_map_list, cuts_pos, mean_size=3):
        print('Calculating jumps in time')
        jumps_heat_map_list = {key: {'left': [], 'right': []} for key in cuts_pos}
        for heat_map in tqdm(heat_map_list):
            max_x = np.max(heat_map[:, 0])
            dim = mean_size, np.max(heat_map[:, 1])+1, np.max(heat_map[:, 2])+1
            _heat_map = {key: {'left': np.zeros(dim), 'right': np.zeros(dim)} for key in cuts_pos}
            cuts_pos_dict = {'left': 0, 'center': max_x//2-mean_size, 'right': max_x-mean_size}
            for pos in heat_map:
                for key in cuts_pos:
                    for mean_pos in range(mean_size):
                        if pos[0] == cuts_pos_dict[key]+mean_pos:
                            if pos[3] == 1:
                                _heat_map[key]['left'][mean_pos, pos[1], pos[2]] = pos[4]
                            if pos[3] == 0:
                                _heat_map[key]['right'][mean_pos, pos[1], pos[2]] = pos[4]

            for key in _heat_map:
                jumps_heat_map_list[key]['left'].append(_heat_map[key]['left'])
                jumps_heat_map_list[key]['right'].append(_heat_map[key]['right'])

        delta_heat_map = {key: {'left': [], 'right': []} for key in cuts_pos}
        for key in cuts_pos:
            delta_heat_map[key]['left'] = [jumps_heat_map_list[key]['left'][i+1]-jumps_heat_map_list[key]['left'][i]
                                           for i in range(len(jumps_heat_map_list[key]['left'])-1)]
            delta_heat_map[key]['right'] = [jumps_heat_map_list[key]['right'][i+1]-jumps_heat_map_list[key]['right'][i]
                                            for i in range(len(jumps_heat_map_list[key]['right'])-1)]

        return delta_heat_map

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
    def plot_line(save_file: Path, x, y,  x_label, y_label, x_size=8, y_size=6, dpi=100):
        _fig = plt.figure(figsize=(x_size, y_size))
        _ax = _fig.add_subplot(111)
        _ax.set_xlabel(x_label)
        _ax.set_ylabel(y_label)
        _ax.plot(x, y)
        plt.savefig(str(save_file), dpi=dpi)
        plt.close(_fig)


if __name__ == '__main__':
    data_path = Path('D:/KMC_data/tests/15_7_7_random')
    hm = TimeHeatMap(data_path)
    hm.process_data()
