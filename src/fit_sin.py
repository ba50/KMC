import json
from pathlib import Path
from multiprocessing import Pool

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import optimize
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

from utils.config import get_config


def reduce_nose(_sim_signal, _ideal_sim_signal, wight):
    std = 0
    for i, value in np.ndenumerate(_sim_signal['y']):
        std += (value - _ideal_sim_signal['y'][i[0]]) ** 2
    std = np.sqrt(std / (len(_sim_signal['y']) - 1))

    _sim_signal['y'] = _sim_signal['y'].map(lambda x: x if abs(x) < std * wight else None)
    return _sim_signal.dropna().reset_index(drop=True)


def fit_curve_signal(fit_function, sim_signal, fit_signal, config):
    params, params_covariance = optimize.curve_fit(fit_function.fit_sin,
                                                   sim_signal['time'],
                                                   sim_signal['y'],
                                                   p0=[config['amplitude'], 0, 0])
    params = {'fit_sine_amp': params[0], 'fit_sine_phi': params[1], 'const': params[2]}
    while abs(params['fit_sine_phi']) > 2 * np.pi:
        params['fit_sine_phi'] -= 2 * np.pi*np.sign(params['fit_sine_phi'])

    fit_y = []
    for step in fit_signal['time']:
        fit_y.append(
            fit_function.fit_sin(step,
                                 params['fit_sine_amp'],
                                 params['fit_sine_phi'],
                                 params['const'])
        )
    fit_signal['y'] = np.array(fit_y)
    return params, fit_signal


class Function:
    def __init__(self, sine_frequency):
        self.sine_frequency = sine_frequency

    def fit_sin_add_line(self, x, fit_sine_amp, fit_sine_phi, fit_line_a, fit_line_b):
        return fit_sine_amp * np.sin(2 * np.pi * self.sine_frequency * x + fit_sine_phi) + fit_line_a * x + fit_line_b

    def fit_sin(self, x, fit_sine_amp, fit_sine_phi, const):
        return (fit_sine_amp * np.sin(2 * np.pi * self.sine_frequency * x + fit_sine_phi)) + const

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

    @staticmethod
    def exp_decay(x, amp: float = 50, tau: float = 5):
        return amp * np.exp(-x / tau)


def generate_phi(sym: Path):
    print(sym)
    hf = h5py.File(str(sym / 'heat_map_plots' / 'timed_jumps_raw_data.h5'), 'r')
    config = get_config(sym / 'input.kmc')
    data_out = {}
    for key in tqdm(hf.keys()):
        sim_signal = pd.DataFrame({'x': hf[key][:, 0], 'y': hf[key][:, 1]})
        field_signal = pd.read_csv(sym/'field_plot.csv')
        fit_signal = pd.DataFrame(
            {'time': np.linspace(field_signal['time'].min(), field_signal['time'].max(), len(field_signal['time']))}
        )

        sim_signal['time'] = np.linspace(fit_signal['time'].min(), fit_signal['time'].max(), len(sim_signal['x']))
        fit_function = Function(config['frequency']*10**-12)
        try:
            for _ in range(1):
                params, fit_signal = fit_curve_signal(fit_function, sim_signal, fit_signal, config)

                ideal_sim_signal = pd.DataFrame(
                    {'time': sim_signal['time'],
                     'y': fit_function.sin(
                         sim_signal['time'],
                         params['fit_sine_amp'],
                         fit_function.sine_frequency,
                         params['fit_sine_phi']) + params['const']
                     }
                )

                temp_sim_signal = reduce_nose(sim_signal, ideal_sim_signal, 1)
                if len(temp_sim_signal) > 100:
                    sim_signal = temp_sim_signal
                else:
                    print('Too small len')
                    break

            sim_signal.dropna(inplace=True)
            params, fit_signal = fit_curve_signal(fit_function, sim_signal, fit_signal, config)

            _fig, _ax1 = plt.subplots()
            _ax2 = _ax1.twinx()
            _ax1.plot(sim_signal['time'], sim_signal['y'], linestyle='--', color='b')
            _ax1.plot(fit_signal['time'], fit_signal['y'], color='r', linestyle='-', label='Fitted func')
            # _ax1.plot(ideal_sim_signal['time'], ideal_sim_signal['y'], color='r', label='Fitted func')
            _ax2.plot(
                fit_signal['time'],
                Function.sin(fit_signal['time'], config['amplitude'], fit_function.sine_frequency, 0),
                linestyle='-',
                color='g',
                label='Field'
            )

            _ax1.set_xlabel('Time [ps]')
            _ax1.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
            _ax1.set_ylabel('Data', color='b')
            _ax2.set_ylabel('Field [eV]', color='g')

            _ax1.legend(loc='upper left')
            _ax2.legend(loc='upper right')
            plt.text(0,
                     config['amplitude'] * .75,
                     "A=%.2e [eV]\nphi=%.2f [rad]" % (abs(params['fit_sine_amp']),
                                                      params['fit_sine_phi']))

            plt.savefig(sym / 'heat_map_plots' / ('fit_sin_%s.png' % key))
            plt.close(_fig)

            data_out[key] = {
                'phi_mean_rad': params['fit_sine_phi'],
                'phi_mean_deg': params['fit_sine_phi'] * 180 / np.pi
            }
        except RuntimeError as e:
            print("\nError in %s: %s\n" % (sym.name, e))

    return sym, data_out


if __name__ == '__main__':
    workers = 1
    base_path = Path('D:/KMC_data/data_2019_12_16_v2')

    sim_path_list = [sim for sim in base_path.glob("*") if sim.is_dir()]

    with Pool(workers) as p:
        _data_out = p.map(generate_phi, sim_path_list)

    for save_path, save_data in _data_out:
        with (save_path / 'heat_map_plots' / 'data_out.json').open('w') as f_out:
            json.dump(save_data, f_out)
