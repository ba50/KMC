import os
import argparse
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




def main(args):
    save_path = Path(str(args.save_path) + '_v' + str(get_sim_version(args.save_path)))
    save_path.mkdir(parents=True)

    freq_list = []
    for i in range(args.low_freq+1, args.high_freq):
        freq_list.extend(np.logspace(i-1, i, num=2, endpoint=False))

    simulations = pd.DataFrame({'frequency': freq_list})

    simulations['cell_type'] = args.cell_type
    simulations['thermalization_time'] = args.thermalization_time
    simulations['window'] = args.window
    simulations['window_epsilon'] = args.window_epsilon
    simulations['contact_switch_left'] = args.contact_switch_left
    simulations['contact_switch_right'] = args.contact_switch_right
    simulations['contact_left'] = args.contact_left
    simulations['contact_right'] = args.contact_right
    simulations['amplitude'] = args.amplitude
    simulations['energy_base'] = args.energy_base

    simulations['periods'] = simulations['frequency'].map(
        lambda freq: np.clip(freq / freq_list[0] * args.base_periods, 0, 4)
    )

    start_stop = {'time_start': [], 'time_end': [], 'periods': [], 'frequency': [], 'split': []}
    for index, row in simulations.iterrows():
        total_time = row['periods']/row['frequency']*10**12
        last_step = 0
        steps = np.ceil(total_time / args.split).astype(int)
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
        lambda x: (x[0]/(x[1]*10.0**-12))/args.window_points,
        axis=1
    )

    version = ['a', 'b', 'c']

    freq_list = simulations['frequency']
    simulations = simulations.loc[np.repeat(simulations.index.values, len(version))].reset_index()
    simulations['version'] = np.array([[x for x in version] for _ in range(len(freq_list))]).flatten()
    freq_list = set(simulations['frequency'])
    simulations['index'] = np.array([[i for _ in range(len(version)*args.split)] for i in range(len(freq_list))]).flatten()

    # simulations['size_x'] = np.flip(np.repeat(np.clip(np.array(get_size(11, 5)), 5, 11), len(version)))
    # simulations['size_y'] = np.flip(np.repeat(np.clip(np.array(get_size(7, 5)), 5, 7), len(version)))
    # simulations['size_z'] = np.flip(np.repeat(np.clip(np.array(get_size(7, 5)), 5, 7), len(version)))

    simulations['size_x'] = args.model_size[0]
    simulations['size_y'] = args.model_size[1]
    simulations['size_z'] = args.model_size[2]

    select_columns = ['size_x', 'size_y', 'size_z', 'cell_type', 'index', 'version', 'split']
    simulations['sim_name'] = simulations[select_columns].apply(
        lambda x: '_'.join([str(x[0]), str(x[1]), str(x[2]), x[3], str(x[4]), x[5], str(x[6])]), axis=1
    )

    simulations['path_to_data'] = simulations['sim_name'].map(lambda x: save_path / x)
    simulations['commend'] = simulations['path_to_data'].map(lambda x: str(args.bin_path)+' '+str(x)+'\n')

    simulations.to_csv(save_path / 'simulations.csv', index=False)

    for _, row in simulations.iterrows():
        path_to_data = save_path / row['sim_name']
        sim_structure = GenerateXYZ((row['size_x'], row['size_y'], row['size_z']))
        if args.cell_type == 'random':
            sim_structure.generate_random()
        elif args.cell_type == 'sphere':
            sim_structure.generate_sphere(5)
        elif args.cell_type == 'plane':
            sim_structure.generate_plane()
        else:
            print('no type')
        generate_sim_input(row, path_to_data, sim_structure)
        commend = str(args.bin_path)+" "+str(path_to_data)+"\n"
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--bin_path", required=True, help="path to compile file")
    parser.add_argument("--save_path", required=True, help="path to save models")

    parser.add_argument("--split", type=int, help="number of subparts", default=1)
    parser.add_argument("--base_periods", type=float, help="base sin period", default=0.5)
    parser.add_argument("--window_points", type=int, help="points in window", default=200)
    parser.add_argument("--low_freq", type=int, help="low freq, pow of 10", default=4)
    parser.add_argument("--high_freq", type=int, help="hie freq, pow of 10", default=10)

    parser.add_argument("--cell_type", choices=['random', 'sphere', 'plane'], default='random')
    parser.add_argument("--model_size", type=int, nargs='+', default=[5, 5, 5])
    parser.add_argument("--thermalization_time", type=int, default=0)
    parser.add_argument("--window", type=int, default=100)
    parser.add_argument("--window_epsilon", type=float, default=.01)
    parser.add_argument("--contact_switch_left", type=bool, default=False)
    parser.add_argument("--contact_switch_right", type=bool, default=False)
    parser.add_argument("--contact_left", type=float, default=1.)
    parser.add_argument("--contact_right", type=float, default=1.)
    parser.add_argument("--amplitude", type=float, default=.1)
    parser.add_argument("--energy_base", type=float, default=.0)
    args = parser.parse_args()

    args.bin_path = Path(args.bin_path)
    args.save_path = Path(args.save_path)
    main(args)

