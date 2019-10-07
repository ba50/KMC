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
    def __init__(self, sine_frequency):
        self.sine_frequency = sine_frequency

    def sin_add_line(self, x, fit_sine_amp, fit_sine_phi, fit_line_a, fit_line_b):
        return fit_sine_amp * np.sin(2 * np.pi * self.sine_frequency * x + fit_sine_phi) + fit_line_a * x + fit_line_b

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
    hf = h5py.File(str(sym / 'heat_map_plots' / 'timed_jumps_raw_data.h5'), 'r')
    config = get_config(sym / 'input.kmc')
    data_out = {}
    for key in hf.keys():
        signal_x = hf[key][:, 0]
        signal_y = hf[key][:, 1]

        fit_x = np.linspace(signal_x.min(), signal_x.max(), 1000)

        fit_function = Function(config['Frequency base of sine function'] *
                                10 ** config['Frequency power of sine function'] * 10**-12)
        try:
            params, params_covariance = optimize.curve_fit(fit_function.sin_add_line,
                                                           signal_x,
                                                           signal_y,
                                                           p0=[config['Amplitude of sine function'],
                                                               1,
                                                               1,
                                                               1])

            params = {'fit_sine_amp': params[0],
                      'fit_sine_phi': params[1],
                      'fit_line_a': params[2],
                      'fit_line_b': params[3]}

            fit_y = []
            for step_x in fit_x:
                fit_y.append(
                    fit_function.sin_add_line(step_x,
                                              params['fit_sine_amp'],
                                              params['fit_sine_phi'],
                                              params['fit_line_a'],
                                              params['fit_line_b']) -
                    Function.line(step_x, params['fit_line_a'], params['fit_line_b']))
            fit_y = np.array(fit_y)

            origin_y = signal_y - Function.line(signal_x, params['fit_line_a'], params['fit_line_b'])
            ideal_y = Function.sin(signal_x,
                                   params['fit_sine_amp'],
                                   fit_function.sine_frequency,
                                   params['fit_sine_phi'])

            mse = Function.mse(origin_y, ideal_y)
            mape = Function.mape(origin_y, ideal_y)

            _fig = plt.figure(figsize=(8, 6))
            _ax = _fig.add_subplot(111)
            _ax.plot(signal_x, origin_y, label='Data', linestyle='--')
            _ax.plot(fit_x, fit_y, label='Fitted function')
            _ax.plot(fit_x, Function.sin(fit_x,
                                         config['Amplitude of sine function'],
                                         config['Frequency base of sine function'] * 10 **
                                         (config['Frequency power of sine function'] - 12),
                                         0), label='Original function')

            plt.legend(loc='upper right')
            plt.text(0,
                     ideal_y.max(),
                     'A=%.2e; phi=%.2f \n MSE=%.2e; MAPE=%d%%' % (abs(params['fit_sine_amp']),
                                                                  params['fit_sine_phi'],
                                                                  mse,
                                                                  mape))
            plt.savefig(sym / 'heat_map_plots' / ('fit_sin_%s.png' % key),
                        dpi=1000,
                        bbox_inches='tight')
            plt.close(_fig)

            data_out[key] = {
                'phi_mean_rad': params['fit_sine_phi'],
                'phi_mean_deg': params['fit_sine_phi'] * 180 / np.pi
            }
        except RuntimeError as e:
            print("\nError in %s: %s\n" % (sym.name, e))

    return sym, data_out


def run(workers: int = 1):
    phi_dict = {}
    base_path = Path('D:/KMC_data/data_2019_10_05')
    sim_key = '30_7_7_random_'
    for y in tqdm(range(7, -1, -1)):
        phi_dict[y] = []
        sym = [base_path/(sim_key+str(y)+'_'+x) for x in ['a', 'b', 'c']]
        with Pool(workers) as p:
            data_out = p.map(generate_phi, sym)

        for save_path, save_data in data_out:
            with (save_path / 'heat_map_plots' / 'data_out.log').open('w') as f_out:
                json.dump(save_data, f_out)


if __name__ == '__main__':
    workers = 1
    run(workers)
