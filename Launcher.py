import numpy as np
import subprocess
import time


class Launcher:
    def __init__(self, command, n_processes):
        self.command = command
        self.n_processes = n_processes
        self.global_index = -1
        self.processes = None

    def __make_process(self):
        self.global_index += 1
        print("Make new process: {}".format(self.global_index+1))
        return subprocess.Popen((self.command[self.global_index]).split(), stdout=subprocess.PIPE)

    def __str__(self):
        temp = [str(i.poll()) + ", " for i in self.processes]
        return "".join(temp)

    def run(self):
        self.processes = [self.__make_process() for i in range(self.n_processes)]
        while True:
            for index in range(len(self.processes)):
                if self.processes[index].poll() == 0:
                    self.processes[index] = self.__make_process()
            if self.global_index >= len(self.command):
                break
            time.sleep(30)

commands = [
    './KMC     7          1000000     0.04        7_random.xyz    7_random_04',
    './KMC     9          1000000     0.04        9_random.xyz    9_random_04',
    './KMC     11         1000000     0.04        11_random.xyz   11_random_04',

    './KMC     7          1000000     0.04        7_sphere.xyz        7_sphere_04',
    './KMC     9          1000000     0.04        9_sphere.xyz        9_sphere_04',
    './KMC     11         1000000     0.04        11_sphere.xyz       11_sphere_04',

    './KMC     7          1000000     0.04        7_plane.xyz    7_plane_04',
    './KMC     9          1000000     0.04        9_plane.xyz    9_plane_04',
    './KMC     11         1000000     0.04        11_plane.xyz   11_plane_04']

test = Launcher(commands, 3)
test.run()
