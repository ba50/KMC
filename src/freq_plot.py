import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import matplotlib.pyplot as plt

from utils.config import get_config


if __name__ == '__main__':
    suffix = '5.0'
    base_path = Path('D:\\KMC_data\\data_2020_01_20_v0')
    sim_path_list = [sim for sim in base_path.glob(f"*_{suffix}") if sim.is_dir()]

    data = {
        'timed_jumps_center_contact_left_jump': [],
        'timed_jumps_center_contact_right_jump': [],
        'timed_jumps_left_contact_left_jump': [],
        'timed_jumps_left_contact_right_jump': [],
        'timed_jumps_right_contact_left_jump': [],
        'timed_jumps_right_contact_right_jump': [],
        'frequency': []
    }

    for sim in sim_path_list:
        sim_config = get_config(sim / 'input.kmc')
        with (sim / 'heat_map_plots' / 'data_out.json').open('r') as f_in:
            phi = json.load(f_in)
        phi = pd.DataFrame(phi)
        for key, value in phi.items():
            data[key].append(phi[key]['phi_mean_rad'])
        data['frequency'].append(sim_config['frequency'])

    data = pd.DataFrame(data)

    for column in tqdm(data.columns.difference(['frequency'])):
        plot_data = {'frequency': [], 'mean': [], 'sem': []}
        for freq, chunk in data[['frequency', column]].groupby('frequency'):
            plot_data['frequency'].append(freq)
            plot_data['mean'].append(chunk[column].mean(axis=0))
            plot_data['sem'].append(chunk[column].std(axis=0)/np.sqrt(len(chunk)))
        plot_data = pd.DataFrame(plot_data)

        _fig = plt.figure(figsize=(8, 6))
        _ax = _fig.add_subplot(111)
        _ax.errorbar(plot_data['frequency'], plot_data['mean'], yerr=plot_data['sem'], fmt='--o')
        _ax.set_xlabel('Frequency [Hz]')
        _ax.set_ylabel('delta phi [rad]')

        plt.xscale('log')
        plt.savefig(base_path / f"phi(f)_{column}_{suffix}.png",
                    dpi=1000,
                    bbox_inches='tight')
        plt.close(_fig)
