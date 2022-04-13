import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

n_ions = [451, 1838, 5946, 7184, 20330]

time = np.array(
    [
        [32364.4, 31962.6, 32307.5],
        [761076, 730447, 715512],
        [5709550, 5709550, 5709550],
        [10565250, 10733750, 10672500],
        [66266500, 66266500, 66266500],
    ]
)

time /= 60 * 60 * 60  # sec to h

plot_data = pd.DataFrame(
    {"x": time.mean(axis=1), "y": n_ions, "x_sem": time.std(axis=1) / time.shape[1]}
)

print(plot_data)

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111)
ax.errorbar(
    plot_data["x"],
    plot_data["y"],
    xerr=plot_data["x_sem"],
    fmt="--o",
)
ax.set_xlabel("Time [h]")
ax.set_ylabel("Ions number")

# plt.yscale("log")
plt.savefig("dev-plot.png", dpi=250, bbox_inches="tight")
plt.close(fig)
