import os
import numpy as np
from copy import copy
from pathlib import Path

import pandas as pd

from src.GenerateXYZ import GenerateXYZ


def generate_sim_input(params_dict, structure: GenerateXYZ):
    (params_dict['path_to_data'] / 'heat_map').mkdir(parents=True, exist_ok=True)

    with (params_dict['path_to_data'] / 'input.kmc').open('w') as file_out:
        file_out.write("{}\t# cell_type\n".format(params_dict['cell_type'].lower()))
        file_out.write("{}\t# size_x\n".format(params_dict['size'][0]))
        file_out.write("{}\t# size_y\n".format(params_dict['size'][1]))
        file_out.write("{}\t# size_z\n".format(params_dict['size'][2]))
        file_out.write("{}\t# therm_time\n".format(params_dict['thermalization_time']))
        file_out.write("{}\t# sim_time\n".format(params_dict['time_end']))
        file_out.write("{}\t# window\n".format(params_dict['window']))
        file_out.write("{}\t# window_epsilon\n".format(params_dict['window_epsilon']))
        file_out.write("{}\t# left_contact_sw\n".format(params_dict['contact_switch'][0]))
        file_out.write("{}\t# right_contact_sw\n".format(params_dict['contact_switch'][1]))
        file_out.write("{}\t# left_contact\n".format(params_dict['contact'][0]))
        file_out.write("{}\t# right_contact\n".format(params_dict['contact'][1]))
        file_out.write("{}\t# amplitude\n".format(params_dict['energy_params']['amplitude']))
        file_out.write("{}\t# frequency_base\n".format(params_dict['energy_params']['frequency_base']))
        file_out.write("{}\t# frequency_power\n".format(params_dict['energy_params']['frequency_power']))
        file_out.write("{}\t# period\n".format(params_dict['energy_params']['periods']))
        file_out.write("{}\t# energy_base\n".format(params_dict['energy_params']['energy_base']))

        structure.save_positions(params_dict['path_to_data']/'positions.xyz')


if __name__ == '__main__':
    workers = 4
    base_periods = 0.5
    low_freq = 4
    high_freq = 9
    bin_path = Path('C:/Users/barja/source/repos/KMC/KMC/build/KMC.exe')
    save_path = Path('D:/KMC_data/data_2019_10_14')
    save_path.mkdir(parents=True, exist_ok=True)

    freq_list = []
    for i in range(low_freq+1, high_freq):
        freq_list.extend(np.logspace(i-1, i, num=2, endpoint=False))

    simulations = pd.DataFrame({'frequency': freq_list})

    simulations['version'] = 'a'
    simulations['cell_type'] = 'Random'
    simulations['size_x'] = 30
    simulations['size_y'] = 7
    simulations['size_z'] = 7
    simulations['time_end'] = 0
    simulations['thermalization_time'] = 200
    simulations['window'] = 10
    simulations['window_epsilon'] = 0.01
    simulations['contact_switch_left'] = 0
    simulations['contact_switch_right'] = 0
    simulations['contact_left'] = 1
    simulations['contact_right'] = 1

    simulations = simulations.loc[np.repeat(simulations.index.values, 3)].reset_index(drop=True)
    simulations['version'] = np.array([[x for x in ['a', 'b', 'c']] for _ in range(len(freq_list))]).flatten()
    simulations['size_x'] = simulations.index.map(lambda x: (30 * np.exp(-x/30)).astype(int))
    print()
    exit()

    params = {}
    sym_path_list = []
    sim_structure = GenerateXYZ(params_base['size'])
    sim_structure.generate_random()
    for index, freq in enumerate(freq_list):
        for s in repeat_list:
            params_per_sim = copy(params_base)
            temp_dict = {}
            sym_name = '_'.join([str(params_base['size'][0]),
                                 str(params_base['size'][1]),
                                 str(params_base['size'][2]),
                                 params_base['cell_type'].lower(),
                                 str(index),
                                 s])
            temp_dict['path_to_data'] = save_path / sym_name
            sym_path_list.append(temp_dict['path_to_data'])

            temp_dict['energy_params'] = {'amplitude': 0.008,
                                          'frequency': freq,
                                          'periods': freq/freq_list[0]*base_periods,
                                          'energy_base': 0}
            params_per_sim.update(temp_dict)
            generate_sim_input(params_per_sim, sim_structure)

    run_df = {'commend': []}
    for s in repeat_list:
        for sim_path in sym_path_list:
            if sim_path.stem[-1] == s:
                run_df['commend'].append(str(bin_path)+' '+str(sim_path))

    run_df = pd.DataFrame(run_df)
    run_df['commend'] = run_df['commend'].map(lambda x: x+'\n')
    for index, split in enumerate(np.split(run_df, workers)):
        if os.name == 'nt':
            with Path(save_path, 'run_%s.ps1' % index).open('w') as f_out:
                f_out.writelines(split['commend'])
        else:
            with Path(save_path, 'run_%s.run' % index).open('w') as f_out:
                f_out.writelines(split['commend'])
