import numpy as np
import matplotlib.pyplot as plt
import click
from os import path


@click.command()
@click.option('--path_to_data', prompt="Path to data", help=" Path to data.")
@click.option('--file_name', prompt="Filename", help="Filename.")
def main(path_to_data, file_name):
    param_file = path.join(path_to_data, 'param_'+file_name+'.dat')
    time_file = path.join(path_to_data, 'time_vector_'+file_name+'.npy')
    data_file = path.join(path_to_data, 'update_vector_'+file_name+'.npy')

    shape = np.genfromtxt(param_file).astype(np.int)
    oxygen_path = np.load(data_file, mmap_mode='r')
    time = np.load(time_file)
    oxygen_path = oxygen_path.reshape(shape[0], shape[1], 3)
    plt.figure()
    plt.plot(time)
    plt.figure()
    plt.plot(time, oxygen_path[:, :, 0].mean(axis=1))
    plt.plot(time, oxygen_path[:, :, 1].mean(axis=1))
    plt.plot(time, oxygen_path[:, :, 2].mean(axis=1))
    plt.show()


main()
