import sys
import subprocess
from pathlib import Path
from multiprocessing import Pool


if __name__ == '__main__':
    _workers = 32
    program_path = '/home/b.jasik/Documents/source/KMC/build/KMC'
    data_path = '/home/b.jasik/Documents/source/KMC/KMC_data/data_2019_01_14_v0'

    program_path = Path(program_path)
    data_path = Path(data_path)

    paths = [i for i in data_path.glob('*') if i.is_dir()]

    def f(x):
        subprocess.run([program_path, x], stdout=sys.stdout)

    with Pool(_workers) as p:
        p.map(f, paths)
