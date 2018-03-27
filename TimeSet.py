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
                  type=int)
    @click.option("--file_name", prompt="File name", help="File name")
    def main(path_to_data,  time_limit, file_name):
        time_file = np.load(path.join(path_to_data,
                                      'time_vector_'+file_name+'.npy'))
        plt.plot(np.diff(time_file))
        plt.show()
        exit()



        time_step = []
        time_delta = 0
        time_end = 0
        steps = 0
        for i in range(len(time_file)):
            time_delta += time_file[i] - time_end
            steps += 1
            if time_delta > time_limit:
                time_step.append(steps)
                steps = 0
                time_end = time_file[i]
                time_delta = 0

        time_step = np.array(time_step)

        plt.plot(time_step)
        plt.show()

    main()

