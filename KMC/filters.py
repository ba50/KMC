from scipy.signal import butter
from scipy.signal import lfilter


def high_pass(y, high_cut=80, fs=1):
    y_0 = y[0]
    y -= y_0
    nyq = 0.5 * fs
    high = high_cut / nyq
    b, a = butter(7, high, btype="highpass")
    return lfilter(b, a, y) + y_0
