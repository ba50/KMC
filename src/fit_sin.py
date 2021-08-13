import json
import argparse
from pathlib import Path
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy import optimize
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

from utils.config import get_config


def fit_curve_signal(fit_function, sim_signal, config):
    params, params_covariance = optimize.curve_fit(fit_function.fit_sin,
                                                   sim_signal['time'],
                                                   sim_signal['y'],
                                                   p0=[config['amplitude'], 0, 0])
    params = {'fit_sine_amp': params[0], 'fit_sine_phi': params[1], 'const': params[2]}
    while abs(params['fit_sine_phi']) > 2 * np.pi:
        params['fit_sine_phi'] -= 2 * np.pi*np.sign(params['fit_sine_phi'])

    fit_y = []
    fit_signal = pd.DataFrame(
        {'time': np.linspace(sim_signal['time'].iloc[0], sim_signal['time'].iloc[-1], len(sim_signal['time']))}
    )
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


def generate_phi(sim_path):
    data = pd.read_csv(sim_path / 'ions_density.csv', sep=',')
    sim_config = get_config(sim_path / 'input.kmc')

    fit_function = Function(sim_config['frequency'] * 10 ** -12)
    sim_signal = pd.DataFrame(
        {
            'time': data['time'],
            'y': data['last_points']
        }
    )
    params, fit_signal = fit_curve_signal(fit_function, sim_signal, sim_config)

    _fig, _ax1 = plt.subplots()
    _ax2 = _ax1.twinx()
    _ax1.plot(sim_signal['time'], sim_signal['y'], linestyle='--', color='b')
    _ax1.plot(fit_signal['time'], fit_signal['y'], color='r', linestyle='-', label='Fitted func')
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

    plt.savefig(sim_path / 'fit_sin_ion_dd.png')
    plt.close(_fig)
    direction_dict = {
        'phi_rad': params['fit_sine_phi'],
        'phi_deg': params['fit_sine_phi'] * 180 / np.pi,
        'path': sim_path,
        'version': (lambda split: split[5])(sim_path.name.split('_')),
        'temp_mul': (lambda split: split[-1])(sim_path.name.split('_')),
        'frequency_index': (lambda split: split[4])(sim_path.name.split('_'))
    }
    return direction_dict


def main(args):
    sim_path_list = [sim for sim in args.data_path.glob("*") if sim.is_dir()]

    with Pool(args.workers) as p:
        data_out = p.map(generate_phi, sim_path_list)
        data_out = pd.DataFrame(data_out)
        data_out.to_csv(args.data_path/'simulations_data.csv', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=Path, required=True, help="path to data from simulation")
    parser.add_argument("--workers", type=int, help="number of workers", default=1)
    args = parser.parse_args()

    main(args)

