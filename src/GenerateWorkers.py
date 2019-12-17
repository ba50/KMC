import time
import queue
import subprocess
from pathlib import Path

import pandas as pd


class GenerateWorkers:
    global_index = 0
    processes = {}

    def __init__(self, commands, workers):
        self.commands = commands
        self.device_list = queue.Queue(workers)

        for i in range(workers):
            self.device_list.put(str(i))

    def __make_process(self, device_name):
        row = self.commands.iloc[self.global_index]
        commend = [str(row['program']), str(row['data_path'])]
        print("Run %s on %s" % (" ".join(commend), device_name))
        return subprocess.Popen(commend, stdout=subprocess.PIPE)

    def __str__(self):
        temp = [str(i.poll()) + ", " for i in self.processes]
        return "".join(temp)

    def run(self):
        while True:
            for device_name, process in self.processes.items():
                print('device_name: %s\t->\t%s' % (device_name, process.stdout.readline()))
                # print(device_name, ": ", process.poll())
                if process.poll() is not None:
                    self.device_list.put(device_name)

            if not self.device_list.empty():
                device_name = self.device_list.get()
                self.processes[device_name] = self.__make_process(device_name)
                self.global_index += 1

            if self.global_index > len(self.commands) - 1:
                break

            time.sleep(0.01)

        print("End of queue")


if __name__ == '__main__':
    _workers = 1
    program_path = 'C:/Users/barja/source/repos/KMC/x64/Release/KMC.exe'
    data_path = 'D:/KMC_data/data_2019_12_17_v3'

    program_path = Path(program_path)
    data_path = Path(data_path)

    commends = pd.DataFrame({'data_path': [i for i in data_path.glob('*') if i.is_dir()]})
    commends['program'] = program_path

    swarm = GenerateWorkers(commends, _workers)
    swarm.run()
