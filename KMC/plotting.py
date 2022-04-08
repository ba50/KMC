import matplotlib.pyplot as plt


def plot_line(
    save_file,
    x_list,
    y_list,
    label_list,
    x_label,
    y_label,
    x_size=8,
    y_size=6,
    dpi=250,
):
    _fig = plt.figure(figsize=(x_size, y_size))
    _ax = _fig.add_subplot(111)
    _ax.set_xlabel(x_label)
    _ax.set_ylabel(y_label)
    for x, y, label in zip(x_list, y_list, label_list):
        if label:
            _ax.plot(x, y, label=label)
        else:
            _ax.plot(x, y)

    if all(label_list):
        plt.legend()
    plt.savefig(save_file, dpi=dpi, bbox_inches="tight")
    plt.close(_fig)


def plot_log(
    save_file,
    x_list,
    y_list,
    label_list,
    x_label,
    y_label,
    x_size=8,
    y_size=6,
    dpi=1000,
):
    _fig = plt.figure(figsize=(x_size, y_size))
    _ax = _fig.add_subplot(111)
    _ax.set_xlabel(x_label)
    _ax.set_ylabel(y_label)
    for x, y, label in zip(x_list, y_list, label_list):
        if label:
            _ax.semilogy(x, y, label=label)
        else:
            _ax.semilogy(x, y)

    if all(label_list):
        plt.legend()
    plt.savefig(save_file, dpi=dpi, bbox_inches="tight")
    plt.close(_fig)
