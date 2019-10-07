import os
import numpy as np
from copy import copy
from pathlib import Path

import pandas as pd

from src.GenerateXYZ import GenerateXYZ


def generate_simulation(params_dict):
    if not (params_dict['path_to_data'] / 'heat_map').exists():
        (params_dict['path_to_data'] / 'heat_map').mkdir(parents=True, exist_ok=True)

    with (params_dict['path_to_data'] / 'input.kmc').open('w') as file_out:
        file_out.write("{}\t# Cell type\n".format(params_dict['cell_type'].lower()))
        file_out.write("{}\t# X number of cells\n".format(params_dict['size'][0]))
        file_out.write("{}\t# Y number of cells\n".format(params_dict['size'][1]))
        file_out.write("{}\t# Z number of cells\n".format(params_dict['size'][2]))
        file_out.write("{}\t# Thermalization time\n".format(params_dict['thermalization_time']))
        file_out.write("{}\t# Time to end simulation\n".format(params_dict['time_end']))
        file_out.write("{}\t# Left contact switch\n".format(params_dict['contact_switch'][0]))
        file_out.write("{}\t# Right contact switch\n".format(params_dict['contact_switch'][1]))
        file_out.write("{}\t# Left contact\n".format(params_dict['contact'][0]))
        file_out.write("{}\t# Right contact\n".format(params_dict['contact'][1]))
        file_out.write("{}\t# Amplitude of sine function\n".format(params_dict['energy_params'][0]))
        file_out.write("{}\t# Frequency base of sine function\n".format(params_dict['energy_params'][1]))
        file_out.write("{}\t# Frequency power of sine function\n".format(params_dict['energy_params'][2]))
        file_out.write("{}\t# Period of sine function\n".format(params_dict['energy_params'][3]))
        file_out.write("{}\t# Delta energy base\n".format(params_dict['energy_params'][4]))

    if params_dict['cell_type'] == 'Random':
        GenerateXYZ(params_dict['size'], params_dict['path_to_data']).generate_random()
    if params_dict['cell_type'] == 'Sphere':
        GenerateXYZ(params_dict['size'], params_dict['path_to_data']).generate_sphere(5)
    if params_dict['cell_type'] == 'Plane':
        GenerateXYZ(params_dict['size'], params_dict['path_to_data']).generate_plane(1)


if __name__ == '__main__':
    version = '12'
    workers = 4
    bin_path = Path('C:/Users/barja/source/repos/KMC/KMC/build/KMC.exe')
    save_path = Path('D:/KMC_data/data_2019_10_05')
    save_path.mkdir(parents=True, exist_ok=True)

    params_base = {'cell_type': 'Random',
                   'size': [30, 7, 7],
                   'time_end': 0,
                   'thermalization_time': 200,
                   'contact_switch': (0, 0),
                   'contact': (1, 1)}

    freq_list = np.logspace(8, 9, num=8)
    repeat_list = ['a', 'b', 'c']

    params = {}
    sym_path_list = []
    for index, freq in enumerate(freq_list):
        for s in repeat_list:
            temp_dict = copy(params_base)
            sym_name = '_'.join([str(params_base['size'][0]),
                                 str(params_base['size'][1]),
                                 str(params_base['size'][2]),
                                 params_base['cell_type'].lower(),
                                 str(index),
                                 s])
            temp_dict['path_to_data'] = save_path / sym_name
            sym_path_list.append(temp_dict['path_to_data'])
            power = int(np.log10(freq))

            temp_dict['energy_params'] = 0.008, freq/10**power, power, np.ceil(freq/freq_list[0]), 0
            generate_simulation(temp_dict)

    run_df = {'commend': []}
    for s in repeat_list:
        for sim_path in sym_path_list:
            if sim_path.stem[-1] == s:
                run_df['commend'].append(str(bin_path)+' '+str(sim_path))

    run_df = pd.DataFrame(run_df)
    run_df['commend'] = run_df['commend'].map(lambda x: x+'\n')
    for index, split in enumerate(np.split(run_df, workers)):
        if os.name == 'nt':
            with Path(save_path, 'run_v%s_%s.ps1' % (version, index)).open('w') as f_out:
                f_out.writelines(split['commend'])
        else:
            with Path(save_path, 'run_v%s_%s.run' % (version, index)).open('w') as f_out:
                f_out.writelines(split['commend'])
