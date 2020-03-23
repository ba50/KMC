import os
import numpy as np
from pathlib import Path

import pandas as pd

from src.GenerateXYZ import GenerateXYZ
from src.fit_sin import Function


def generate_sim_input(_row, _path_to_data, _structure: GenerateXYZ):
    (_path_to_data / 'heat_map').mkdir(parents=True, exist_ok=True)
    (_path_to_data / 'oxygen_map').mkdir(parents=True, exist_ok=True)

    with (_path_to_data / 'input.kmc').open('w') as file_out:
        file_out.write("{}\t# cell_type\n".format(_row['cell_type'].lower()))
        file_out.write("{}\t# size_x\n".format(_row['size_x']))
        file_out.write("{}\t# size_y\n".format(_row['size_y']))
        file_out.write("{}\t# size_z\n".format(_row['size_z']))
        file_out.write("{}\t# therm_time\n".format(_row['thermalization_time']))
        file_out.write("{}\t# time_start\n".format(_row['time_start']))
        file_out.write("{}\t# time_end\n".format(_row['time_end']))
        file_out.write("{}\t# window\n".format(_row['window']))
        file_out.write("{}\t# window_epsilon\n".format(_row['window_epsilon']))
        file_out.write("{}\t# left_contact_sw\n".format(_row['contact_switch_left']))
        file_out.write("{}\t# right_contact_sw\n".format(_row['contact_switch_right']))
        file_out.write("{}\t# left_contact\n".format(_row['contact_left']))
        file_out.write("{}\t# right_contact\n".format(_row['contact_right']))
        file_out.write("{}\t# amplitude\n".format(_row['amplitude']))
        file_out.write("{}\t# frequency\n".format(_row['frequency']))
        file_out.write("{}\t# periods\n".format(_row['periods']))
        file_out.write("{}\t# energy_base\n".format(_row['energy_base']))
        file_out.write("{}\t# temperature\n".format(_row['temperature']))

        _structure.save_positions(_path_to_data/'positions.xyz')


def get_size(amp, tau):
    size = []
    for x in range(len(freq_list)):
        y = int(Function.exp_decay(x, amp=amp, tau=tau))
        if (y % 2) == 0:
            size.append(y - 1)
        else:
            size.append(y)
    return size


def get_sim_version(path: Path):
    sim_paths = list(path.parent.glob('%s*' % path.name))
    return len(sim_paths)


if __name__ == '__main__':
    split = 1
    base_periods = 1.5
    window_points = 200
    low_freq = 7
    high_freq = 10
    save_path = Path('D:/KMC_data/data_2020_01_19')
    save_path = Path(str(save_path) + '_v' + str(get_sim_version(save_path)))
    save_path.mkdir(parents=True)

    freq_list = []
    for i in range(low_freq+1, high_freq):
        freq_list.extend(np.logspace(i-1, i, num=2, endpoint=False))

    simulations = pd.DataFrame({'frequency': freq_list})

    simulations['cell_type'] = 'random'
    simulations['thermalization_time'] = 0
    simulations['window_epsilon'] = 0.5*10**-2
    simulations['contact_switch_left'] = 2
    simulations['contact_switch_right'] = 2
    simulations['contact_left'] = 500
    simulations['contact_right'] = 500
    simulations['amplitude'] = .02
    simulations['energy_base'] = 0.0

    simulations['periods'] = simulations['frequency'].map(
        lambda freq: np.clip(freq / freq_list[0] * base_periods, 0, 4.0)
    )

    start_stop = {'time_start': [], 'time_end': [], 'periods': [], 'frequency': [], 'split': []}
    for index, row in simulations.iterrows():
        total_time = row['periods']/row['frequency']*10**12
        last_step = 0
        steps = np.ceil(total_time / split).astype(int)
        for split_step, next_step in enumerate(range(steps, int(total_time)+steps, steps)):
            start_stop['time_start'].append(last_step)
            start_stop['time_end'].append(next_step)
            start_stop['periods'].append(row['periods'])
            start_stop['frequency'].append(row['frequency'])
            start_stop['split'].append(split_step)
            last_step = next_step
    start_stop = pd.DataFrame(start_stop)
    simulations = simulations.merge(start_stop, on=['frequency', 'periods'])

    simulations['window'] = simulations[['periods', 'frequency']].apply(
        lambda x: (x[0]/(x[1]*10.0**-12))/window_points,
        axis=1
    )

    version = ['a']

    freq_list = simulations['frequency']
    simulations = simulations.loc[np.repeat(simulations.index.values, len(version))].reset_index()
    simulations['version'] = np.array([[x for x in version] for _ in range(len(freq_list))]).flatten()
    freq_list = set(simulations['frequency'])
    simulations['index'] = np.array([[i for _ in range(len(version)*split)] for i in range(len(freq_list))]).flatten()

    temperature = np.linspace(1, 5, 4)

    freq_list = simulations['frequency']
    simulations = simulations.loc[np.repeat(simulations.index.values, len(temperature))].reset_index()
    simulations['temperature'] = np.array([[x for x in temperature] for _ in range(len(freq_list))]).flatten()

    # simulations['size_x'] = np.flip(np.repeat(np.clip(np.array(get_size(11, 5)), 5, 11), len(version)))
    # simulations['size_y'] = np.flip(np.repeat(np.clip(np.array(get_size(7, 5)), 5, 7), len(version)))
    # simulations['size_z'] = np.flip(np.repeat(np.clip(np.array(get_size(7, 5)), 5, 7), len(version)))

    simulations['size_x'] = 15
    simulations['size_y'] = 7
    simulations['size_z'] = 7

    select_columns = ['size_x', 'size_y', 'size_z', 'cell_type', 'index', 'version', 'split', 'temperature']
    simulations['sim_name'] = simulations[select_columns].apply(
        lambda x: '_'.join([str(x[0]), str(x[1]), str(x[2]), x[3], str(x[4]), x[5], str(x[6]), str(x[7])]), axis=1
    )

    simulations['path_to_data'] = simulations['sim_name'].map(lambda x: save_path / x)

    simulations.to_csv(save_path / 'simulations.csv', index=False)
    for _, row in simulations.iterrows():
        path_to_data = save_path / row['sim_name']
        sim_structure = GenerateXYZ((row['size_x'], row['size_y'], row['size_z']))
        sim_structure.generate_random()
        generate_sim_input(row, path_to_data, sim_structure)
