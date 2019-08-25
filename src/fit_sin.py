import json
import time
from pathlib import Path
from scipy import optimize
import matplotlib.pyplot as plt

import h5py
import numpy as np
from tqdm import tqdm


class Function:
    @staticmethod
    def sin_add_line(x, amp, sin_a, sin_b, line_a, line_b):
        return amp * np.sin(2 * np.pi * sin_a * x + sin_b) + line_a * x + line_b

    @staticmethod
    def line(x, a, b):
        return a*x + b

    @staticmethod
    def sin(x, amp, a, b):
        return amp*np.sin(2*np.pi*a*x+b)

    @staticmethod
    def mse(y_true, y_pred):
        return ((y_true - y_pred) ** 2).mean(axis=0)

    @staticmethod
    def mape(y_true, y_pred):
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def get_config(path: Path):
    str_index = [0]
    int_index = [*range(2, 10)]
    float_index = [*range(10, 13)]
    with path.open() as _config:
        lines = _config.readlines()
    _config = {}
    for index, line in enumerate(lines):
        if index in str_index:
            _data = line.split('#')
            _key = _data[1].strip()
            _config[_key] = _data[0].strip()
        if index in int_index:
            _data = line.split('#')
            _key = _data[1].strip()
            _config[_key] = int(_data[0].strip())
        if index in float_index:
            _data = line.split('#')
            _key = _data[1].strip()
            _config[_key] = float(_data[0].strip())

    return _config


"""
simulations = [Path('D:/KMC_data/data_2019_07_24/amplitude/30_7_7_random_01'),
               Path('D:/KMC_data/data_2019_07_24/amplitude/30_7_7_random_04'),
               Path('D:/KMC_data/data_2019_07_24/freq/30_7_7_random_01'),
               Path('D:/KMC_data/data_2019_07_24/freq/30_7_7_random_0025')]
"""
simulations = [Path('D:/KMC_data/tests/15_7_7_random')]

for simulation in simulations:
    print('Start to calculate %s' % simulation.name)
    with h5py.File(str(simulation/'heat_map_plots'/'timed_jumps_raw_data.h5')) as hf:
        config = get_config(simulation / 'input.kmc')

        time.sleep(1)
        phi_array = []
        for key in tqdm(hf.keys()):
            data = hf[key]

            data_x = np.linspace(data[:].min(), data[:].max(), 100)
            params, params_covariance = optimize.curve_fit(Function.sin_add_line,
                                                           data[:, 0],
                                                           data[:, 1],
                                                           p0=[config['Amplitude of sine function'],
                                                               config['Frequency base of sine function'] *
                                                               10**config['Frequency power of sine function'],
                                                               1,
                                                               1,
                                                               1])
            phi_array.append(params[2])

            data_y = []
            for x_0 in data_x:
                data_y.append(Function.sin_add_line(x_0, params[0], params[1], params[2], params[3], params[4]) -
                              Function.line(x_0, params[3], params[4]))
            data_y = np.array(data_y)

            X_data_y = data[:, 1] - Function.line(data[:, 0], params[3], params[4])
            Y_data_y = Function.sin(data[:, 0], params[0], params[1], params[2])

            mse = Function.mse(X_data_y, Y_data_y)
            mape = Function.mape(X_data_y, Y_data_y)

            _fig = plt.figure(figsize=(8, 6))
            _ax = _fig.add_subplot(111)
            _ax.plot(data[:, 0], X_data_y, label='Data', linestyle='--')
            _ax.plot(data_x, data_y, label='Fitted function')
            _ax.plot(data_x, Function.sin(data_x,
                                          config['Amplitude of sine function'],
                                          config['Frequency base of sine function'] * 10 **
                                          config['Frequency power of sine function'],
                                          0), label='Original function')

            plt.legend(loc='upper right')
            plt.text(0,
                     X_data_y.max(),
                     'A=%.2e; f=%.2e; phi=%.2f \n MSE=%.2e; MAPE=%d%%' % (abs(params[0]),
                                                                          params[1],
                                                                          params[2],
                                                                          mse,
                                                                          mape))
            plt.savefig(simulation / 'heat_map_plots' / ('fit_sin_%s.png' % key),
                        dpi=1000,
                        bbox_inches='tight')
            plt.close(_fig)

        phi_array = np.array(phi_array)
        data_out = {'phi_mean_rad': phi_array.mean(),
                    'phi_mean_deg': phi_array.mean() * 180 / np.pi,
                    'phi_std': phi_array.std()}
        with (simulation / 'heat_map_plots' / 'data_out.log').open('w') as f_out:
            json.dump(data_out, f_out)

        time.sleep(1)
