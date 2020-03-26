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


def generate_phi(inputs):
    sim = inputs[0]
    config = inputs[1]
    hf = h5py.File(str(sim / 'heat_map_plots' / 'timed_jumps_raw_data.h5'), 'r')
    sim_config = get_config(sim / 'input.kmc')
    data_out = []
    for key in tqdm(hf.keys()):
        direction_dict = {}
        field_signal = pd.read_csv(sim/'field_plot.csv')
        fit_signal = pd.DataFrame(
            {'time': np.linspace(field_signal['time'].min(), field_signal['time'].max(), len(field_signal['time']))}
        )
        fit_function = Function(sim_config['frequency']*10**-12)

        sim_signal = None
        try:
            done = False
            mean_std = config['mean_std']
            while not done:
                sim_signal = pd.DataFrame({'x': hf[key][:, 0], 'y': hf[key][:, 1]})
                sim_signal['time'] = np.linspace(fit_signal['time'].min(),
                                                 fit_signal['time'].max(),
                                                 len(sim_signal['x']))
                original_len = len(sim_signal)
                for _ in range(config['repeat']):
                    params, fit_signal = fit_curve_signal(fit_function, sim_signal, fit_signal, sim_config)

                    ideal_sim_signal = pd.DataFrame(
                        {'time': sim_signal['time'],
                         'y': fit_function.sin(
                             sim_signal['time'],
                             params['fit_sine_amp'],
                             fit_function.sine_frequency,
                             params['fit_sine_phi']) + params['const']
                         }
                    )

                    temp_sim_signal = reduce_nose(sim_signal, ideal_sim_signal, mean_std)
                    if len(temp_sim_signal) > original_len * .75:
                        done = True
                        sim_signal = temp_sim_signal
                    else:
                        done = False
                        mean_std += config['auto_tune_step']
                        break

            print(f"{sim} end at {mean_std} mean std")
            sim_signal.dropna(inplace=True)
            params, fit_signal = fit_curve_signal(fit_function, sim_signal, fit_signal, sim_config)

            _fig, _ax1 = plt.subplots()
            _ax2 = _ax1.twinx()
            _ax1.plot(sim_signal['time'], sim_signal['y'], linestyle='--', color='b')
            _ax1.plot(fit_signal['time'], fit_signal['y'], color='r', linestyle='-', label='Fitted func')
            # _ax1.plot(ideal_sim_signal['time'], ideal_sim_signal['y'], color='r', label='Fitted func')
            _ax2.plot(
                fit_signal['time'],
                Function.sin(fit_signal['time'], sim_config['amplitude'], fit_function.sine_frequency, 0),
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
                     sim_config['amplitude'] * .75,
                     "A=%.2e [eV]\nphi=%.2f [rad]" % (abs(params['fit_sine_amp']),
                                                      params['fit_sine_phi']))

            plt.savefig(sim / 'heat_map_plots' / ('fit_sin_%s.png' % key))
            plt.close(_fig)

            direction_dict['phi_rad'] = params['fit_sine_phi']
            direction_dict['phi_deg'] = params['fit_sine_phi'] * 180 / np.pi

        except RuntimeError as e:
            direction_dict['phi_rad'] = None
            direction_dict['phi_deg'] = None
            print("\nError in %s: %s\n" % (sim.name, e))

        direction_dict['path'] = sim
        direction_dict['direction'] = (lambda split: split[4])(key.split('_'))
        direction_dict['location'] = (lambda split: split[2])(key.split('_'))
        direction_dict['version'] = (lambda split: split[5])(sim.name.split('_'))
        direction_dict['temp_mul'] = (lambda split: split[-1])(sim.name.split('_'))
        direction_dict['frequency_index'] = (lambda split: split[4])(sim.name.split('_'))
        data_out.append(direction_dict)

    return data_out


def main():
    config = {
        'workers': 3,
        'base_path': Path('D:\\KMC_data\\data_2020_01_20_v0'),
        'mean_std': 2.,
        'auto_tune_step': 2.,
        'repeat': 3,
        'original_len': .75  # in %
    }

    sim_path_list = [(sim, config) for sim in config['base_path'].glob("*") if sim.is_dir()]

    with Pool(config['workers']) as p:
        _data_out = p.map(generate_phi, sim_path_list)
        _data_out = [item for sublist in _data_out for item in sublist]
        _data_out = pd.DataFrame(_data_out)
        _data_out.to_csv(config['base_path']/'simulations_data.csv', index=False)


if __name__ == '__main__':
    main()
