import sys
from os import path
import numpy as np
import glob

path_to_files = glob.glob(sys.argv[1])
# Get name from path

file_names = [path.splitext(path.basename(path_to_file))[0] for path_to_file in path_to_files]

fp = []
for index, file_name in enumerate(file_names):
    fp.append(np.memmap(file_name + '.bin',
                        dtype='float32',
                        mode='w+',
                        shape=(sum(1 for line in open(path_to_files[index])),3)))

data = [np.genfromtxt(path_to_file) for path_to_file in path_to_files]
for i, x in enumerate(data):
    fp[i][:] = data[i][:]

del fp
del data
