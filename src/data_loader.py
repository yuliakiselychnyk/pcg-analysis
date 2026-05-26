import pandas as pd


def read_signal_file(file, sampling_rate):
    df = pd.read_csv(file, header=None)

    df.columns = [
        "time_index",
        "signal_1",
        "signal_2",
        "signal_3"
    ]

    df["time_seconds"] = df["time_index"] / sampling_rate

    return df