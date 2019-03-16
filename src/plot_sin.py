import numpy as np
import matplotlib.pyplot as plt

with open('./sin.txt', 'r') as file_in:
    data = file_in.readlines()
data = [float(i[:-1]) for i in data]
data = np.array(data)
print(data)

plt.plot(range(data.shape[0]), data)
plt.show()
