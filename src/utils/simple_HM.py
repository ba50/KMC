from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.utils.config import get_config


sim_path = Path("F:\\KMC_data\\data_2021_03_23_v1\\11_7_7_random_0_a_0_1.0")

files_in_list = (sim_path / "heat_map").glob('*.dat')

config = get_config(sim_path / "input.kmc")

files_in_list = sorted(list(files_in_list), key=lambda i: float(i.stem))

last_data = pd.read_csv(files_in_list[0], sep='\t', names=['x', 'y', 'z', 'direction', 'count'])
diff_data = last_data

jump_plot = []
for file_in in files_in_list[1:]:
    new_data = pd.read_csv(file_in, sep='\t', names=['x', 'y', 'z', 'direction', 'count'])
    diff_data['count'] = new_data['count'] - last_data['count']
    if file_in.stem == '19500':
        print()

    temp = diff_data[diff_data['direction'] == 0]
    mean = temp['count'].mean()
    jump_plot.append((int(file_in.stem), mean))

    last_data = new_data
jump_plot = np.array(jump_plot)

fig_1 = plt.figure()
plt.plot(jump_plot[:, 0], jump_plot[:, 1])

plt.show()
