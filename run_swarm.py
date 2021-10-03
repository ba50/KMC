import argparse
import queue
import subprocess
import time
from pathlib import Path

import pandas as pd


class GenerateSwarm:
    global_index = 0
    processes = {}

    def __init__(self, commands, workers):
        self.commands = commands
        self.device_list = queue.Queue(workers)

        for i in range(workers):
            self.device_list.put(str(i))

    def __make_process(self, device_name):
        row = self.commands.iloc[self.global_index]
        commend = [str(row["program"]), str(row["data_path"])]
        print("Run %s on %s" % (" ".join(commend), device_name))
        return subprocess.Popen(commend, stdout=None)

    def __str__(self):
        temp = [str(i.poll()) + ", " for i in self.processes]
        return "".join(temp)

    def run(self):
        while True:
            for device_name, process in self.processes.items():
                if process.poll() is not None:
                    self.device_list.put(device_name)

            if not self.device_list.empty() and self.global_index < len(self.commands):
                device_name = self.device_list.get()
                self.processes[device_name] = self.__make_process(device_name)
                self.global_index += 1

            if all(self.processes) is None:
                break

            time.sleep(1)

        print("End of queue")


def main(args):
    commends = pd.DataFrame(
        {"data_path": [i for i in args.data_path.glob("*") if i.is_dir()]}
    )
    commends["program"] = args.program_path
    assert len(commends) != 0, "No simulations to run"

    swarm = GenerateSwarm(commends, args.workers)
    swarm.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--program-path", required=True, help="path to program bin")
    parser.add_argument("--data-path", required=True, help="path to data")
    parser.add_argument("--workers", type=int, help="number of workers", default=1)
    main_args = parser.parse_args()

    main_args.program_path = Path(main_args.program_path)
    main_args.data_path = Path(main_args.data_path)
    main(main_args)
