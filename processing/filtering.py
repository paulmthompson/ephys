from scipy.signal import butter, filtfilt


def butter_bandpass(
    lowcut,
    highcut,
    fs,
    order=5,
):
    return butter(order, [lowcut, highcut], fs=fs, btype="band",)


def butter_bandpass_filter(
    data,
    lowcut,
    highcut,
    fs,
    order=5,
):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y
