import json
from pathlib import Path
from multiprocessing import Pool

import h5py
import numpy as np
import pandas as pd
from scipy import optimize
import matplotlib.pyplot as plt

from utils.config import get_config


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


def generate_phi(sym: Path):
    hf = h5py.File(str(sym / 'heat_map_plots' / 'timed_jumps_raw_data.h5'), 'r')
    config = get_config(sym / 'input.kmc')
    data_out = {}
    for key in hf.keys():
        sim_signal = pd.DataFrame({'x': hf[key][:, 0], 'y': hf[key][:, 1]})

        fit_signal = pd.DataFrame({'x': np.linspace(sim_signal['x'].min(), sim_signal['x'].max(), 1000)})

        fit_function = Function(config['frequency_base'] *
                                10 ** config['frequency_power'] * 10**-12)
        try:
            params = {}
            for _ in range(3):
                params, params_covariance = optimize.curve_fit(fit_function.sin_add_line,
                                                               sim_signal['x'],
                                                               sim_signal['y'],
                                                               p0=[config['amplitude'],
                                                                   1,
                                                                   1,
                                                                   1])

                params = {'fit_sine_amp': params[0],
                          'fit_sine_phi': params[1],
                          'fit_line_a': params[2],
                          'fit_line_b': params[3]}

                fit_y = []
                for step_x in fit_signal['x']:
                    fit_y.append(
                        fit_function.sin_add_line(step_x,
                                                  params['fit_sine_amp'],
                                                  params['fit_sine_phi'],
                                                  params['fit_line_a'],
                                                  params['fit_line_b']) -
                        Function.line(step_x, params['fit_line_a'], params['fit_line_b']))
                fit_signal['y'] = np.array(fit_y)

                sim_signal['y'] = sim_signal['y'].map(lambda x: x if abs(x) < sim_signal['y'].std()*15 else None)
                sim_signal = sim_signal.dropna()

            origin_y = sim_signal['y'] - Function.line(sim_signal['x'], params['fit_line_a'], params['fit_line_b'])
            ideal_y = Function.sin(sim_signal['x'],
                                   params['fit_sine_amp'],
                                   fit_function.sine_frequency,
                                   params['fit_sine_phi'])

            mse = Function.mse(origin_y, ideal_y)
            mape = Function.mape(origin_y, ideal_y)

            _fig = plt.figure(figsize=(8, 6))
            _ax = _fig.add_subplot(111)
            _ax.plot(sim_signal['x'], origin_y, label='Data', linestyle='--')
            _ax.plot(fit_signal['x'], fit_signal['y'], label='Fitted function')
            _ax.plot(fit_signal['x'], Function.sin(fit_signal['x'],
                                                   config['amplitude'],
                                                   config['frequency_base'] * 10 **
                                                   (config['frequency_power'] - 12),
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


if __name__ == '__main__':
    workers = 3
    base_path = Path('D:/KMC_data/data_2019_10_11')

    sim_path_list = [sim for sim in base_path.glob("*") if sim.is_dir()]

    with Pool(workers) as p:
        _data_out = p.map(generate_phi, sim_path_list)

    for save_path, save_data in _data_out:
        with (save_path / 'heat_map_plots' / 'data_out.json').open('w') as f_out:
            json.dump(save_data, f_out)
