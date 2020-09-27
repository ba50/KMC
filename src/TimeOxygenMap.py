from pathlib import Path

import numpy as np

from src.utils import plotting


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

            self.save_data_path.mkdir(parents=True)  # TODO: Delete exist_ok

    def process_data(self, time_end, filter_level=0.004):
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
                timed_o_positions.append(o_positions)
                step += o_number+2

        timed_o_positions = np.array(timed_o_positions)

        pos_diff = np.diff(timed_o_positions, axis=0).mean(axis=(1, 2))

        filter_indexes = np.where(pos_diff > filter_level)

        timed_o_positions = timed_o_positions[filter_indexes].astype(np.int)

        with (self.load_data_path / 'positions_filter.xyz').open('w') as file_out:
            for step in range(timed_o_positions.shape[0]):
                file_out.write(str(o_number) + '\n\n')
                for ion_index in range(timed_o_positions.shape[1]):
                    file_out.write(
                        f'O\t{timed_o_positions[step, ion_index, 0]}'
                        f'\t{timed_o_positions[step, ion_index, 1]}'
                        f'\t{timed_o_positions[step, ion_index, 2]}\n'
                    )
            file_out.write('\n')

        timed_mean_pos = np.zeros((timed_o_positions.shape[0], 4))
        timed_mean_pos[:, 0] = np.arange(start=0, stop=time_end, step=time_end/timed_o_positions.shape[0])
        timed_mean_pos[:, 1:] = timed_o_positions.mean(axis=1)

        plotting.plot_line(save_file=self.save_data_path/'mean_path.png',
                           x_list=[timed_mean_pos[:, 0], timed_mean_pos[:, 0], timed_mean_pos[:, 0]],
                           y_list=[timed_mean_pos[:, 1], timed_mean_pos[:, 2], timed_mean_pos[:, 3]],
                           label_list=['mean_x', 'mean_y', 'mean_z'],
                           x_label='Time [ps]',
                           y_label='Mean position [au]',
                           dpi=250)
