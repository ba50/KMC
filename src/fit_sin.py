import json
from pathlib import Path
from multiprocessing import Pool

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import optimize
import matplotlib.pyplot as plt


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
    float_index = [*range(1, 15)]
    with path.open() as _config:
        lines = _config.readlines()
    _config = {}
    for index, line in enumerate(lines):
        if index in str_index:
            _data = line.split('#')
            _key = _data[1].strip()
            _config[_key] = _data[0].strip()

        if index in float_index:
            _data = line.split('#')
            _key = _data[1].strip()
            _config[_key] = float(_data[0].strip())

    return _config


def generate_phi(sym: Path):
    # print("Start to calculate %s\n" % sym.name)
    hf = h5py.File(str(sym / 'heat_map_plots' / 'timed_jumps_raw_data.h5'))
    config = get_config(sym / 'input.kmc')
    data_out = {}
    for key in hf.keys():
        print(key)
        mean_signal = []
        data = hf[key]

        test_period = int(len(data)/config['Period of sine function'])

        for i in range(test_period):
            point = []
            for step in range(len(data)):
                if step % test_period == i:
                    point.append(data[step, 1])
            mean_signal.append(point)
        mean_signal = pd.DataFrame(mean_signal, columns=['y_'+str(i)
                                                         for i in range(int(config['Period of sine function']))])
        mean_signal['x'] = data[:test_period, 0]

        mean_signal['y_mean'] = mean_signal[['y_'+str(i)
                                             for i in range(int(config['Period of sine function']))]].apply(
            lambda x: x.mean(), axis=1
        )

        data_x = np.linspace(mean_signal['x'].min(), mean_signal['x'].max(), 1000)
        try:
            params, params_covariance = optimize.curve_fit(Function.sin_add_line,
                                                           mean_signal['x'],
                                                           mean_signal['y_mean'],
                                                           p0=[config['Amplitude of sine function'],
                                                               config['Frequency base of sine function'] *
                                                               10 ** config['Frequency power of sine function'],
                                                               1,
                                                               1,
                                                               1])

            data_y = []
            for x_0 in data_x:
                data_y.append(
                    Function.sin_add_line(x_0, params[0], params[1], params[2], params[3], params[4]) -
                    Function.line(x_0, params[3], params[4]))
            data_y = np.array(data_y)

            X_data_y = mean_signal['y_mean'] - Function.line(mean_signal['x'], params[3], params[4])
            Y_data_y = Function.sin(mean_signal['x'], params[0], params[1], params[2])

            mse = Function.mse(X_data_y, Y_data_y)
            mape = Function.mape(X_data_y, Y_data_y)

            _fig = plt.figure(figsize=(8, 6))
            _ax = _fig.add_subplot(111)
            _ax.plot(mean_signal['x'], X_data_y, label='Data', linestyle='--')
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
            plt.savefig(sym / 'heat_map_plots' / ('fit_sin_%s.png' % key),
                        dpi=1000,
                        bbox_inches='tight')
            plt.close(_fig)

            data_out[key] = {
                'phi_mean_rad': params[2],
                'phi_mean_deg': params[2] * 180 / np.pi
            }
        except RuntimeError as e:
            print("\nError in %s: %s\n" % (sym.name, e))

    return sym, data_out


def run(workers: int = 1):
    phi_dict = {}
    base_path = Path('D:/KMC_data/data_2019_09_08')
    sim_key = '30_7_7_random_'
    for y in tqdm(range(7, -1, -1)):
        phi_dict[y] = []
        sym = [base_path/(sim_key+str(y)+'_'+x) for x in ['a', 'b', 'c']]
        with Pool(workers) as p:
            data_out = p.map(generate_phi, sym)

        for save_path, save_data in data_out:
            with (save_path / 'heat_map_plots' / 'data_out.log').open('w') as f_out:
                json.dump(save_data, f_out)

    exit()

    with (base_path[0] / 'mean_data_out.log').open('w') as f_out:
        for y in range(0, 8):
            phi_dict[y] = np.array(phi_dict[y]['phi_mean_red']).mean()
            json.dump(phi_dict[y], f_out)


if __name__ == '__main__':
    run(3)
