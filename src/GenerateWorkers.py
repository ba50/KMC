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
        self.processes = [self.__make_process() for _ in range(self.n_processes)]

        if len(self.processes) < 2:
            running = False
        else:
            running = True

        while running:
            for index in range(len(self.processes)):
                if not self.processes[index] is None:
                    print('index: %d\t->\t%s' % (index, self.processes[index].stdout.readline()))
                    if self.global_index < len(self.command):
                        if self.processes[index].poll() == 0:
                            self.processes[index] = self.__make_process()
                else:
                    if all(i is None for i in self.processes):
                        print(self.global_index)
                        print(self.n_processes)
                        print(len(self.command))
                        running = False
                        print("Exit generate worker!")

            time.sleep(.5)


if __name__ == '__main__':

    base = '../build/KMC.exe '

    """
    commends = [base+'D:/KMC_data/30_7_7_random_1',
                base+'D:/KMC_data/30_7_7_random_2',
                base+'D:/KMC_data/30_7_7_random_3',
                base+'D:/KMC_data/30_7_7_random_4',
                base+'D:/KMC_data/30_7_7_random_5',
                base+'D:/KMC_data/30_7_7_random_6',
                base+'D:/KMC_data/30_7_7_random_7',
                base+'D:/KMC_data/30_7_7_random_8',
                base+'D:/KMC_data/30_7_7_random_9',
                base+'D:/KMC_data/30_7_7_random_10',
                ]
    """

    commends = [
        base+'D:/KMC_data/data_2019_05_29/amplitude/30_7_7_random_1',
        base+'D:/KMC_data/data_2019_05_29/amplitude/30_7_7_random_02',
        base+'D:/KMC_data/data_2019_05_29/amplitude/30_7_7_random_004',
        base+'D:/KMC_data/data_2019_05_29/frequency/30_7_7_random_0005',
        base+'D:/KMC_data/data_2019_05_29/frequency/30_7_7_random_005',
        base+'D:/KMC_data/data_2019_05_29/frequency/30_7_7_random_05',
        base+'D:/KMC_data/data_2019_05_29/size/30_3_3_random',
        base+'D:/KMC_data/data_2019_05_29/size/30_7_7_random',
        base+'D:/KMC_data/data_2019_05_29/size/30_15_15_random',
        base+'D:/KMC_data/data_2019_05_29/size/30_30_30_random'
    ]

    workers = GenerateWorkers(commends, 4)
    workers.run()
