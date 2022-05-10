import numpy as np
import matplotlib.pyplot as plt

fild = np.array([0.2, 0.5, 2, 5, 8, 10, 20])
n_atoms = np.array([284, 484, 1324])

none = np.nan
data = np.array([
    [none, none, none, none, none, none, 0.226708356],
    [none, none, 0.024238766, 0.027729908, 0.027056424, 0.026047597, 0.023977578],
    [0.148975598, 0.009248805, 0.008739105, 0.009433282, 0.009629373, none, none],
])

fig, ax = plt.subplots(figsize=(16, 12))
im = ax.imshow(data, cmap=plt.get_cmap("winter"))

# Show all ticks and label them with the respective list entries
ax.set_xticks(np.arange(len(fild)), labels=fild, fontsize=24)
ax.set_yticks(np.arange(len(n_atoms)), labels=n_atoms, fontsize=24)

# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

# Loop over data dimensions and create text annotations.
for i in range(len(n_atoms)):
    for j in range(len(fild)):
        text = ax.text(j, i, f"{data[i, j]:.3e}", ha="center", va="center", color="w", fontsize=16)

ax.set_xlabel("Fild [meV]", fontsize=24)
ax.set_ylabel("Ions number", fontsize=24)

plt.savefig(
    f"sigma_plot.png",
    dpi=250,
    bbox_inches="tight",
)
