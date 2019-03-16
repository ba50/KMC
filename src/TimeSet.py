import numpy as np
import matplotlib.pyplot as plt
import click
import os.path as path

if __name__ == '__main__':
    @click.command()
    @click.option("--path_to_data", prompt="Path to data", help="Path to data")
    @click.option("--time_limit",
                  prompt="Time limit",
                  help="Time limit",
                  type=float)
    @click.option("--layer",
                  prompt="Layer number",
                  help="Layer number",
                  type=float)
    @click.option("--file_name", prompt="File name", help="File name")
    def main(path_to_data,  time_limit, layer,  file_name):
        param_file = path.join(path_to_data, 'params.dat')
        time_file = path.join(path_to_data, 'time_vector.npy')
        data_file = path.join(path_to_data, 'update_vector.npy')

        shape = np.genfromtxt(param_file).astype(np.int)
        oxygen_path = np.load(data_file)
        time = np.load(time_file)
        time_diff = np.diff(time)
        oxygen_path = oxygen_path.reshape(shape[0], shape[1], 3)

        time_step = []
        time_delta = 0
        steps = 0
        for i in range(1, oxygen_path.shape[0]):
            for j in range(oxygen_path.shape[1]):
                if oxygen_path[i, j, 2] == layer:
                    if not oxygen_path[i-1, j, 2] == oxygen_path[i, j, 2]:
                        time_delta += time_diff[i]
                        steps += 1
                        if time_delta > time_limit:
                            time_step.append(steps)
                            steps = 0
                            time_delta = 0

        time_step = np.array(time_step)

        plt.plot(time_step)
        plt.show()

    main()
