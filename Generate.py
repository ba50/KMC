import numpy as np
from copy import copy
from pathlib import Path 

from Launcher import gen_sym


path_to_gen = 'D:\KMC_data\data_2019_09_08'
path_to_gen = Path(path_to_gen)

params_base = {'cell_type': 'Random',
               'size': [30, 7, 7],
               'time_end': 0,
               'thermalization_time': 200,
               'contact_switch': (0, 0),
               'contact': (1, 1)}

freq_list = np.arange(16*10**-6, 4*10**-4, 5*10**-5)
repeat_list = ['a', 'b', 'c'] 

file_list = {path_to_gen/('run_v5_%s.ps1' % s): [] for s in repeat_list}
[file.parent.mkdir(parents=True, exist_ok=True) for file in file_list.keys()]

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
        temp_dict['path_to_data'] = path_to_gen / sym_name
        sym_path_list.append(temp_dict['path_to_data'])
        power = int(np.log10(freq))

        temp_dict['energy_params'] = (0.008, freq/10**power, power, np.ceil(freq/freq_list[0]), 0)
        gen_sym(temp_dict)


for s in repeat_list: 
    for sym_path in sym_path_list:
        if sym_path.stem[-1] == s:
            file_list[path_to_gen/('run_v5_%s.ps1' % s)].append(str(sym_path))


for key in file_list:
    file_list[key] = ["../build/KMC.exe %s" % s for s in file_list[key]]

for key in file_list:
    file_list[key] = "\n".join(file_list[key])


for key in file_list:
    key.write_text(file_list[key])

