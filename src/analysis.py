from scipy.signal import find_peaks

from src.data_loader import read_signal_file
from src.signal_processing import bandpass_filter, calculate_rms


def analyze_file(
    file,
    sampling_rate,
    selected_signal,
    lowcut,
    highcut,
    filter_order,
    min_peak_distance_sec,
    peak_prominence
):
    local_df = read_signal_file(file, sampling_rate)

    signal_data = local_df[selected_signal].values

    filtered_signal = bandpass_filter(
        signal=signal_data,
        sampling_rate=sampling_rate,
        lowcut=lowcut,
        highcut=highcut,
        order=filter_order
    )

    min_peak_distance_samples = max(
        1,
        int(min_peak_distance_sec * sampling_rate)
    )

    peaks, _ = find_peaks(
        filtered_signal,
        distance=min_peak_distance_samples,
        prominence=peak_prominence
    )

    duration = float(local_df["time_seconds"].max())

    return {
        "Файл": file.name,
        "Кількість рядків": local_df.shape[0],
        "Тривалість, с": round(duration, 2),
        "RMS початкового сигналу": round(calculate_rms(signal_data), 4),
        "RMS після фільтрації": round(calculate_rms(filtered_signal), 4),
        "Кількість піків": len(peaks)
    }