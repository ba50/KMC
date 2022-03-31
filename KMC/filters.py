from scipy.signal import butter
from scipy.signal import lfilter


def high_pass(y, high_cut, fs, order=5):
    y_0 = y.mean()
    y -= y_0

    nyq = 0.5 * fs
    normal_cutoff = high_cut / nyq
    b, a = butter(order, normal_cutoff, btype="high")

    return lfilter(b, a, y) + y_0
