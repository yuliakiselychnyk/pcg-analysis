import numpy as np
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


def normalize_metric(values):
    values = np.array(values, dtype=float)

    min_value = values.min()
    max_value = values.max()

    if max_value == min_value:
        return np.ones_like(values)

    return (values - min_value) / (max_value - min_value)


def recommend_signal_for_analysis(
    dataframe,
    sampling_rate,
    lowcut,
    highcut,
    filter_order,
    min_peak_distance_sec,
    peak_prominence
):
    signal_names = ["signal_1", "signal_2", "signal_3"]

    results = []

    for signal_name in signal_names:
        signal_data = dataframe[signal_name].values

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

        rms_value = calculate_rms(filtered_signal)

        max_abs_value = np.max(np.abs(filtered_signal))

        if rms_value == 0:
            crest_factor = 0
        else:
            crest_factor = max_abs_value / rms_value

        zero_crossings = np.sum(
            np.diff(np.signbit(filtered_signal)) != 0
        )

        results.append({
            "signal": signal_name,
            "rms_after_filter": rms_value,
            "peaks_count": len(peaks),
            "zero_crossings": zero_crossings,
            "crest_factor": crest_factor
        })

    rms_values = [item["rms_after_filter"] for item in results]
    peaks_values = [item["peaks_count"] for item in results]
    zero_crossing_values = [item["zero_crossings"] for item in results]
    crest_values = [item["crest_factor"] for item in results]

    normalized_rms = normalize_metric(rms_values)
    normalized_peaks = normalize_metric(peaks_values)
    normalized_zero_crossings = normalize_metric(zero_crossing_values)
    normalized_crest = normalize_metric(crest_values)

    for index, item in enumerate(results):
        score = (
            0.35 * normalized_rms[index]
            + 0.30 * normalized_zero_crossings[index]
            + 0.20 * normalized_peaks[index]
            - 0.15 * normalized_crest[index]
        )

        item["score"] = round(float(score), 4)
        item["rms_after_filter"] = round(float(item["rms_after_filter"]), 4)
        item["crest_factor"] = round(float(item["crest_factor"]), 4)

    best_signal = max(results, key=lambda item: item["score"])

    return best_signal["signal"], results