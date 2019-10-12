import json
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from utils.config import get_config


if __name__ == '__main__':
    base_path = Path('D:/KMC_data/data_2019_10_11')
    sim_key = '30_7_7_random_'

    data = {'timed_jumps_center_contact_left_jump': [],
            'timed_jumps_center_contact_right_jump': [],
            'timed_jumps_left_contact_left_jump': [],
            'timed_jumps_left_contact_right_jump': [],
            'timed_jumps_right_contact_left_jump': [],
            'timed_jumps_right_contact_right_jump': []}

    for y in range(7, -1, -1):
        for key in data.keys():
            points = {}
            for sub_point in ['a', 'b', 'c']:
                simulation = base_path / (sim_key + str(y) + '_' + sub_point)
                sim_config = get_config(simulation / 'input.kmc')
                with (simulation / 'heat_map_plots' / 'data_out.json').open('r') as json_file:
                    phi = json.load(json_file)
                if key in phi:
                    points[sub_point] = phi[key]['phi_mean_rad']
                else:
                    points[sub_point] = None

            base = sim_config['frequency_base']
            power = sim_config['frequency_power']
            row = {'freq': base*10**power}
            row.update(points)
            data[key].append(row)

    for key in data.keys():
        data[key] = pd.DataFrame(data[key])
        data[key]['mean'] = data[key][['a', 'b', 'c']].apply(lambda x: x.mean(), axis=1)
        data[key]['SEM'] = data[key][['a', 'b', 'c']].apply(lambda x: x.std()/np.sqrt(len(x)), axis=1)

        _fig = plt.figure(figsize=(8, 6))
        _ax = _fig.add_subplot(111)
        _ax.errorbar(data[key]['freq'], data[key]['mean'], yerr=data[key]['SEM'], fmt='--o')
        plt.savefig(base_path / ('phi(f)_%s.png' % key),
                    dpi=1000,
                    bbox_inches='tight')
        plt.close(_fig)
