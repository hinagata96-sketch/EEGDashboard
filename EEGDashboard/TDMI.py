import os
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import mutual_info_classif
import matplotlib.pyplot as plt
import mne

# === Frequency Bands ===
band_ranges = {
    "delta": (0.5, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta":  (13, 30),
    "gamma": (30, 45)
}

# === Filters & Features ===
def bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return filtfilt(b, a, data)

def extract_time_features(signal):
    return {
        "mean": np.mean(signal),
        "std": np.std(signal),
        "skew": skew(signal),
        "kurtosis": kurtosis(signal),
        "rms": np.sqrt(np.mean(signal**2)),
        "zero_crossing": np.sum((signal[:-1] * signal[1:]) < 0)
    }

def extract_hjorth(signal):
    first_deriv = np.diff(signal)
    second_deriv = np.diff(first_deriv)

    var_0 = np.var(signal)
    var_1 = np.var(first_deriv)
    var_2 = np.var(second_deriv)

    activity = var_0
    mobility = np.sqrt(var_1 / var_0) if var_0 != 0 else 0
    complexity = (np.sqrt(var_2 / var_1) / mobility) if var_1 != 0 and mobility != 0 else 0

    return {
        "hjorth_activity": activity,
        "hjorth_mobility": mobility,
        "hjorth_complexity": complexity
    }

# === EEG Processor ===
def process_file_time_features(file_path, emotion, subject, trial, fs=128, n_segments=5):
    df = pd.read_csv(file_path)
    ch_names = df.columns.tolist()
    data = df.values.T

    info = mne.create_info(ch_names=ch_names, sfreq=fs, ch_types='eeg')
    raw = mne.io.RawArray(data, info)

    # ICA
    ica = mne.preprocessing.ICA(n_components=min(15, len(ch_names)), method='fastica', random_state=42)
    ica.fit(raw)
    sources = ica.get_sources(raw).get_data()
    stds = np.std(sources, axis=1)
    ica.exclude = [i for i, s in enumerate(stds) if s > np.percentile(stds, 90)]
    raw_clean = ica.apply(raw.copy())

    total_samples = raw_clean.n_times
    seg_len = total_samples // n_segments
    band_features = {band: [] for band in band_ranges}

    for seg_idx in range(n_segments):
        start = seg_idx * seg_len
        stop = start + seg_len
        seg_data = raw_clean.get_data()[:, start:stop]

        for ch_idx, ch_name in enumerate(ch_names):
            signal = seg_data[ch_idx]

            for band, (low, high) in band_ranges.items():
                filtered = bandpass_filter(signal, low, high, fs)

                time_feats = extract_time_features(filtered)
                hjorth_feats = extract_hjorth(filtered)

                feature_dict = {
                    "emotion": emotion,
                    "subject": subject,
                    "trial": trial,
                    "segment": seg_idx + 1,
                    "channel": ch_name,
                    "band": band
                }
                feature_dict.update(time_feats)
                feature_dict.update(hjorth_feats)

                band_features[band].append(feature_dict)

    return band_features

## Batch processing and Colab-specific code removed for Streamlit compatibility.
## Use the functions above in your Streamlit app for interactive file processing.


