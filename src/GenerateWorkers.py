import time
import queue
import argparse
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


def main(args):
    commends = pd.DataFrame({'data_path': [i for i in args.data_path.glob('*') if i.is_dir()]})
    commends['program'] = args.program_path

    swarm = GenerateWorkers(commends, args.workers)
    swarm.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--program_path", required=True, help="path to program bin")
    parser.add_argument("--data_path", required=True, help="path to data")
    parser.add_argument("--workers", type=int, help="number of workers", default=1)
    args = parser.parse_args()

    args.program_path = Path(args.program_path)
    args.data_path = Path(args.data_path)
    main(args)

