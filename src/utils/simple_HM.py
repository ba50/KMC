from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from src.utils.config import get_config

files_in_list = Path('F:\\KMC_data\\data_2021_03_16_v3\\7_7_7_random_0_a_0_1.0\\heat_map').glob('*.dat')

config = get_config(Path('F:\\KMC_data\\data_2021_03_16_v3\\7_7_7_random_0_a_0_1.0\\input.kmc'))

files_in_list = sorted(list(files_in_list), key=lambda i: float(i.stem))

last_data = pd.read_csv(files_in_list[0], sep='\t', names=['x', 'y', 'z', 'direction', 'count'])
diff_data = last_data

jump_plot = []
for file_in in files_in_list[1:]:
    new_data = pd.read_csv(file_in, sep='\t', names=['x', 'y', 'z', 'direction', 'count'])
    diff_data['count'] = new_data['count'] - last_data['count']

    temp = diff_data[diff_data['direction'] == 0]
    jump_plot.append(temp['count'].mean())

    last_data = new_data

fig_1 = plt.figure()
plt.plot(jump_plot)

plt.show()
