import os
import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool
import matplotlib.pyplot as plt


class TimeHeatMap:
    def __init__(
            self,
            load_data_path: Path,
            save_data_path: Path = None,
            options=None,
            workers: int = 1
    ):
        """
        Load data from HeatMap.
        :param load_data_path:
        :param save_data_path:
        """
        self.options = {'mean': False, "jumps": True, "save_raw": True, 'mean_size': 3}

        if options:
            self.options.update(options)

        self.load_data_path = load_data_path
        if not (self.load_data_path/'heat_map').exists():
            self.load_data_path = None
        else:
            self.save_data_path = save_data_path
            self.workers = workers

            if not self.save_data_path:
                self.save_data_path = self.load_data_path / 'heat_map_plots'

            self.save_data_path.mkdir(parents=True, exist_ok=True)  # TODO: delete exist_ok

            self.file_list = sorted((self.load_data_path/'heat_map').glob('*.dat'),
                                    key=lambda i: float(os.path.splitext(os.path.basename(i))[0]))

    def process_data(self):
        print(f"Loading heat map files... {self.load_data_path.name}")
        if self.options['jumps']:
            positions = ['left', 'center', 'right']
            directions = ['left', 'right']
            jumps_heat_map_list = self._time_jumps(self.file_list, positions, self.workers)

            mean_jumps = {}
            for pos in positions:
                mean_jumps[pos] = {direc: np.array([(i[0], i[1].mean())
                                                    for idx, i in enumerate(jumps_heat_map_list[pos][direc])])
                                   for direc in directions}

            for pos in positions:
                for direc in directions:
                    data = mean_jumps[pos][direc]
                    mean_jumps[pos][direc] = data[data[:, 1] > 10**-5]

            file_name_h5py = h5py.File(str(self.save_data_path / 'timed_jumps_raw_data.h5'), 'w')

            for pos in positions:
                for direc in directions:
                    file_name_h5py.create_dataset(
                        'timed_jumps_'+pos+'_contact_'+direc+'_jump', data=mean_jumps[pos][direc]
                    )

            for key in file_name_h5py.keys():
                self.plot_line(save_file=Path(self.save_data_path, key+'.png'),
                               x=file_name_h5py[key][:, 0],
                               y=file_name_h5py[key][:, 1],
                               x_label='Time [ps]',
                               y_label='Jumps [au]',
                               dpi=self.options['dpi'])

    def _time_jumps(self, heat_map_file_list: list, cuts_pos: list, workers: int):
        self.cuts_pos = cuts_pos

        print("\nCalculating jumps in time")
        jumps_heat_map_list = {key: {0: [], 1: []} for key in self.cuts_pos}
        with Pool(workers) as p:
            for data_out in tqdm(p.imap(self.worker, heat_map_file_list, chunksize=1), total=len(heat_map_file_list)):
                for directions in range(2):
                    for key in self.cuts_pos:
                        jumps_heat_map_list[key][directions].append([data_out['time'], data_out[key][directions]])

        delta_heat_map = {key: {'left': [], 'right': []} for key in self.cuts_pos}
        for key in self.cuts_pos:
            delta_heat_map[key]['left'] = []
            delta_heat_map[key]['right'] = []
            for i in range(len(jumps_heat_map_list[key][0]) - 1):
                delta_heat_map[key]['left'].append(
                    [
                        jumps_heat_map_list[key][0][i][0],
                        jumps_heat_map_list[key][0][i+1][1]-jumps_heat_map_list[key][0][i][1]
                    ]
                )

            for i in range(len(jumps_heat_map_list[key][1]) - 1):
                delta_heat_map[key]['right'].append(
                    [
                        jumps_heat_map_list[key][1][i][0],
                        jumps_heat_map_list[key][1][i+1][1]-jumps_heat_map_list[key][1][i][1]
                    ]
                )

        return delta_heat_map

    def worker(self, file_name):
        data_out = {key: {0: None, 1: None} for key in self.cuts_pos}
        heat_map = pd.read_csv(file_name, sep='\t', names=['x', 'y', 'z', 'direction', 'count'])
        max_x = heat_map['x'].max()
        dim = self.options['mean_size'], heat_map['y'].max() + 1, heat_map['z'].max() + 1
        _heat_map = {'x': [], 'y': [], 'z': [], 'cuts_pos': [], 'direction': []}
        cuts_pos_dict = {'left': 0,
                         'center': max_x // 2 - self.options['mean_size'],
                         'right': max_x - self.options['mean_size']}

        for direction in range(2):
            for key in self.cuts_pos:
                for x in range(cuts_pos_dict[key], cuts_pos_dict[key] + dim[0]):
                    for y in range(dim[1]):
                        for z in range(dim[2]):
                            _heat_map['x'].append(x)
                            _heat_map['y'].append(y)
                            _heat_map['z'].append(z)
                            _heat_map['cuts_pos'].append(key)
                            _heat_map['direction'].append(direction)

        _heat_map = pd.DataFrame(_heat_map)
        _heat_map = _heat_map.merge(heat_map, on=['x', 'y', 'z', 'direction'])

        for direction in range(2):
            for key in self.cuts_pos:
                sample = np.zeros(dim)
                data = _heat_map.loc[_heat_map['cuts_pos'] == key]
                data = data.loc[data['direction'] == direction]
                data.apply(lambda _x: self.create_array(_x,
                                                        sample,
                                                        cuts_pos_dict), axis=1)
                data_out[key][direction] = sample
        data_out['time'] = int(file_name.stem)

        return data_out

    @staticmethod
    def create_array(x, data, cuts_pos_dict):
        data[np.clip(x[0]-cuts_pos_dict[x[3]], 0, data.shape[0]), x[1], x[2]] = x[5]

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

        return [x_heat_map_list[i+1]-x_heat_map_list[i] for i in range(len(x_heat_map_list)-1)]

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="path to simulation data")
    parser.add_argument("--workers", type=int, help="number of workers", default=1)
    parser.add_argument("--dpi", type=int, help="Plotting dpi", default=100)

    main_args = parser.parse_args()

    main_args.data_path = Path(main_args.data_path)
    
    sim_path_list = [sim for sim in main_args.data_path.glob("*") if sim.is_dir()]

    for simulation_path in sim_path_list:
        timed_heat_map = TimeHeatMap(
                load_data_path=simulation_path,
                options={'dpi': main_args.dpi},
                workers=main_args.workers
                )
        if timed_heat_map.load_data_path:
            timed_heat_map.process_data()
