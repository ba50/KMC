import os
import numpy as np
from pathlib import Path

import pandas as pd

from src.GenerateXYZ import GenerateXYZ


def generate_sim_input(_row, _path_to_data, _structure: GenerateXYZ):
    (_path_to_data / 'heat_map').mkdir(parents=True, exist_ok=True)

    with (_path_to_data / 'input.kmc').open('w') as file_out:
        file_out.write("{}\t# cell_type\n".format(_row['cell_type'].lower()))
        file_out.write("{}\t# size_x\n".format(_row['size_x']))
        file_out.write("{}\t# size_y\n".format(_row['size_y']))
        file_out.write("{}\t# size_z\n".format(_row['size_z']))
        file_out.write("{}\t# therm_time\n".format(_row['thermalization_time']))
        file_out.write("{}\t# sim_time\n".format(_row['time_end']))
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

        _structure.save_positions(_path_to_data/'positions.xyz')


def exp_decay(x, amp=50, tau=5):
    return (amp * np.exp(-x / tau)).astype(int)


def get_size(amp, tau):
    size = []
    for x in range(len(freq_list)):
        y = exp_decay(x, amp=amp, tau=tau)
        if (y % 2) == 0:
            size.append(y - 1)
        else:
            size.append(y)
    return size


def get_sim_version(path: Path):
    sim_paths = list(path.parent.glob('%s*' % path.name))
    return len(sim_paths)


if __name__ == '__main__':
    workers = 3
    base_periods = 0.5
    window_points = 200
    low_freq = 4
    high_freq = 9
    bin_path = Path('/home/b.jasik/Documents/source/KMC/build/KMC')
    save_path = Path('KMC_data/data_2019_11_20')
    save_path = Path(str(save_path) + '_v' + str(get_sim_version(save_path)))
    save_path.mkdir(parents=True)

    freq_list = []
    for i in range(low_freq+1, high_freq):
        freq_list.extend(np.logspace(i-1, i, num=2, endpoint=False))

    simulations = pd.DataFrame({'frequency': freq_list})

    simulations['version'] = 'a'
    simulations['cell_type'] = 'random'
    simulations['size_x'] = 30
    simulations['size_y'] = 7
    simulations['size_z'] = 7
    simulations['time_end'] = 0
    simulations['thermalization_time'] = 200
    simulations['window'] = 100
    simulations['window_epsilon'] = 0.01
    simulations['contact_switch_left'] = 0
    simulations['contact_switch_right'] = 0
    simulations['contact_left'] = 1
    simulations['contact_right'] = 1
    simulations['amplitude'] = .05
    simulations['energy_base'] = 0.0

    simulations['periods'] = simulations['frequency'].map(
        lambda freq: np.clip(freq / freq_list[0] * base_periods, 0, 4)
    )

    simulations['window'] = simulations[['periods', 'frequency']].apply(
        lambda x: (x[0]/(x[1]*10.0**-12))/window_points,
        axis=1
    )

    version = ['a', 'b', 'c']

    simulations = simulations.loc[np.repeat(simulations.index.values, 3)].reset_index()
    simulations['version'] = np.array([[x for x in version] for _ in range(len(freq_list))]).flatten()

    simulations['size_x'] = np.flip(np.repeat(np.clip(np.array(get_size(11, 5)), 5, 11), len(version)))
    simulations['size_y'] = np.flip(np.repeat(np.clip(np.array(get_size(7, 5)), 5, 7), len(version)))
    simulations['size_z'] = np.flip(np.repeat(np.clip(np.array(get_size(7, 5)), 5, 7), len(version)))

    simulations['sim_name'] = simulations.apply(
        lambda x: '_'.join([str(x[4]), str(x[5]), str(x[6]), x[3], str(x[0]), x[2]]), axis=1
    )

    simulations['path_to_data'] = simulations['sim_name'].map(lambda x: save_path / x)
    simulations['commend'] = simulations['path_to_data'].map(lambda x: str(bin_path)+' '+str(x)+'\n')

    simulations.to_csv(save_path / 'simulations.csv', index=False)
    for index, chunk in enumerate(np.split(simulations, workers)):
        if os.name == 'nt':
            f_out = Path(save_path, 'run_%s.ps1' % index).open('w')
        else:
            f_out = Path(save_path, 'run_%s.run' % index).open('w')
        for _, row in chunk.iterrows():
            path_to_data = save_path / row['sim_name']
            sim_structure = GenerateXYZ((row['size_x'], row['size_y'], row['size_z']))
            sim_structure.generate_random()
            generate_sim_input(row, path_to_data, sim_structure)
            commend = str(bin_path)+" "+str(path_to_data)+"\n"
            f_out.write(commend)
        f_out.close()
