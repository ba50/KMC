import subprocess
import time


class GenerateWorkers:
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

        if len(self.processes) < 2:
            running = False
        else:
            running = True

        while running:
            for index in range(len(self.processes)):
                print(self.global_index)
                if self.global_index < len(self.command):
                    if self.processes[index].poll() == 0:
                        self.processes[index] = self.__make_process()
                else:
                    running = False
                    print("Exit generate worker!")

            time.sleep(.5)

