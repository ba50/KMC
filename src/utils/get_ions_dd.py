from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.GenerateXYZ import GenerateXYZ
from src.utils.config import get_config


if __name__ == "__main__":
    sim_path = Path("F:\\KMC_data\\data_2021_03_24_v0\\25_11_11_random_0_a_0_1.0")
    data_path = sim_path / 'oxygen_map\\positions.xyz'

    config = get_config(sim_path / 'input.kmc')

    num_atoms, raw_frames = GenerateXYZ.read_frames_dataframe(data_path)

    time_steps = raw_frames['frame'].unique()
    max_x = max(raw_frames['x'])
    chunks = 8
    chunk_size = max_x//chunks

    ions_dd = {'time': [], 'x': [], 'y': []}
    for time_step in time_steps:
        pos_frame = raw_frames[raw_frames['frame'] == time_step]
        for x_step in range(chunks):

            ions_count = len(pos_frame.loc[(x_step*chunk_size <= pos_frame['x']) & (pos_frame['x'] < x_step*chunk_size+chunk_size)])

            ions_dd['time'].append(time_step)
            ions_dd['x'].append(x_step)
            ions_dd['y'].append(ions_count/num_atoms)
    ions_dd = pd.DataFrame(ions_dd)

    for time_step in time_steps[:50]:
        plt.plot(range(chunks), ions_dd[ions_dd['time'] == time_step]['y'])

    plt.show()
