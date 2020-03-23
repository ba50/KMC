import sys
import subprocess
from pathlib import Path
from multiprocessing import Pool


def make_process(_data_path):
    program_path = 'C:/Users/barja/source/repos/KMC/x64/Release/KMC.exe'
    subprocess.run([program_path, str(_data_path)], stdout=sys.stdout)


def main(workers, data_path):
    data_path = Path(data_path)

    paths = [i for i in data_path.glob('*') if i.is_dir()]

    with Pool(workers) as p:
        p.map(make_process, paths)


if __name__ == '__main__':
    _workers = 1
    _data_path = 'D:/KMC_data/data_2020_01_19_v0'

    main(_workers, _data_path)
