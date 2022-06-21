import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

n_ions = [484, 726, 1324, 1692]

time = np.array(
    [
        [8191.04, 8203.59, 8203.42],
        [19860.5, 19693.2, 19765.9],
        [65756.9, 65937.7, 65102.4],
        [1213240, 1213330, 1213630],
    ]
)

time /= 3600  # sec to h

plot_data = pd.DataFrame(
    {"x": time.mean(axis=1), "y": n_ions, "x_sem": time.std(axis=1) / time.shape[1]}
)

labels_font = {'fontname':'Times New Roman', 'size': 24}
ticks_font = {'fontname':'Times New Roman', 'size': 16}

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111)
ax.errorbar(
    plot_data["x"],
    plot_data["y"],
    xerr=plot_data["x_sem"],
    fmt="--o",
)
ax.set_xlabel("Czas [h]", **labels_font)
ax.set_ylabel("Liczba jonów", **labels_font)
plt.xticks(**ticks_font)
plt.yticks(**ticks_font)

plt.xscale("log")
plt.savefig("dev-plot.png", dpi=250, bbox_inches="tight")
plt.close(fig)
