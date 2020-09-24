from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


class TimeOxygenMap:
    def __init__(self, load_data_path: Path, save_data_path: Path = None, workers: int = 1):
        """
        Load data from OxygenMap.
        :param load_data_path:
        :param save_data_path:
        """

        self.load_data_path = load_data_path / 'oxygen_map'
        if not self.load_data_path.exists():
            self.load_data_path = None
        else:
            self.save_data_path = save_data_path
            self.workers = workers

            if not self.save_data_path:
                self.save_data_path = self.load_data_path.parent / 'oxygen_map_plots'

            self.save_data_path.mkdir(parents=True)

    def process_data(self):
        print('Loading oxygen map files...')

        timed_o_positions = []
        with (self.load_data_path / 'positions.xyz').open('r') as file_in:
            lines = file_in.readlines()

            step = 0
            while len(lines) > step:
                o_positions = []
                o_number = int(lines[step])
                for pos in range(o_number):
                    line = lines[step+pos+2].split('\t')
                    o_positions.append([float(line[1]),
                                        float(line[2]),
                                        float(line[3])])
                o_positions = np.array(o_positions)
                timed_o_positions.append(o_positions)
                step += o_number+2

        timed_mean_pos = []
        for timed_pos in timed_o_positions:
            timed_mean_pos.append(timed_pos.mean(axis=0))
        timed_mean_pos = np.array(timed_mean_pos)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(timed_mean_pos[:, 0])
        ax.plot(timed_mean_pos[:, 1])
        ax.plot(timed_mean_pos[:, 2])
        plt.savefig(str(self.save_data_path/'mean_path.png'))
        plt.close(fig)
