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


cells = ['7']
energies = ['0.00']
cell_types = ['random']
time_end = str(1)

commands = []
for cell in cells:
    for energy in energies:
        for cell_type in cell_types:
            commands.append(
                './build/KMC '+
                cell+' '+
                time_end + ' ' +
                energy+' '+
                cell+'_'+cell_type+'.xyz '+
                cell+'_'+cell_type+'_'+energy.replace('0.', '')
            )

test = Launcher(commands, 3)
test.run()
