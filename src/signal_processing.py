import numpy as np
from scipy.signal import butter, filtfilt


def bandpass_filter(signal, sampling_rate, lowcut, highcut, order):
    nyquist = sampling_rate / 2

    if highcut >= nyquist:
        highcut = nyquist - 1

    b, a = butter(
        order,
        [lowcut, highcut],
        btype="bandpass",
        fs=sampling_rate
    )

    filtered_signal = filtfilt(b, a, signal)

    return filtered_signal


def calculate_spectrum(signal_values, sampling_rate):
    centered_signal = signal_values - np.mean(signal_values)
    n = len(centered_signal)

    frequencies = np.fft.rfftfreq(n, d=1 / sampling_rate)
    spectrum = np.abs(np.fft.rfft(centered_signal))

    return frequencies, spectrum


def calculate_rms(signal_values):
    rms_value = np.sqrt(np.mean(signal_values ** 2))

    return rms_value