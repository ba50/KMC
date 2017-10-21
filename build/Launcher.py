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
        return subprocess.Popen((self.command[self.global_index]+"_"+str(self.global_index)).split(), stdout=subprocess.PIPE)

    def __str__(self):
        temp = [str(i.poll()) + ", " for i in self.processes]
        return "".join(temp)

    def run(self):
        self.processes = [self.__make_process() for i in range(self.n_processes)]
        while True:
            if self.global_index > len(self.command):
                break
            for index in range(len(self.processes)):
                if self.processes[index].poll() == 0:
                    self.processes[index] = self.__make_process()
            time.sleep(30)


cells = range(7, 13, 2)
energies = np.arange(0, 0.02, 0.02)
step = 1e6
types = ['random', 'sphere', 'plane']




commands = [
    './KMC     7          1000000     0.00        7_random.xyz    7_random_none',
    './KMC     9          1000000     0.00        9_random.xyz    9_random_none',
    './KMC     11         1000000     0.00        11_random.xyz   11_random_none',
    './KMC     7          1000000     0.02        7_random.xyz    7_random_02',
    './KMC     9          1000000     0.02        9_random.xyz    9_random_02',
    './KMC     11         1000000     0.02        11_random.xyz   11_random_02',

    './KMC     7          1000000     0.00        7_shere.xyz       7_shere_none',
    './KMC     9          1000000     0.00        9_shere.xyz       9_shere_none',
    './KMC     11         1000000     0.00        11_shere.xyz       11_shere_none',
    './KMC     7          1000000     0.02        7_shere.xyz        7_shere_02',
    './KMC     9          1000000     0.02        9_shere.xyz        9_shere_02',
    './KMC     11         1000000     0.02        11_shere.xyz       11_shere_02',

    './KMC     7          1000000     0.00        7_plane.xyz    7_palne_none',
    './KMC     9          1000000     0.00        9_plane.xyz    9_palne_none',
    './KMC     11         1000000     0.00        11_plane.xyz   11_plane_none'
    './KMC     7          1000000     0.02        7_plane.xyz    7_plane_02',
    './KMC     9          1000000     0.02        9_plane.xyz    9_plane_02',
    './KMC     11         1000000     0.02        11_plane.xyz   11_plane_02']

#test = Launcher(commands, 3)
#test.run()
