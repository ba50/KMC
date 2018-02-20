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
        if self.global_index < len(self.command):
            print(self.command[self.global_index])
            return subprocess.Popen((self.command[self.global_index]).split(), stdout=subprocess.PIPE)

    def __str__(self):
        temp = [str(i.poll()) + ", " for i in self.processes]
        return "".join(temp)

    def run(self):
        self.processes = [self.__make_process() for i in range(self.n_processes)]
        for i, proc in enumerate(self.processes):
            if i < len(self.command):
                print(proc.communicate())

        while True:
            for index in range(len(self.processes)):
                if self.global_index < len(self.command):
                    if self.processes[index].poll() == 0:
                        self.processes[index] = self.__make_process()
                else:
                    exit()
            time.sleep(30)

threads = 3
cells = ['11']
energies = ['0.00']
cell_types = ['random']
time_end = str(40)

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

test = Launcher(commands, threads)
test.run()
