
# === CWT Feature Extraction Function ===
import numpy as np, pandas as pd, mne
from scipy.stats import entropy, kurtosis, skew
from sklearn.preprocessing import MinMaxScaler

def extract_cwt_features(
    data, ch_names, fs=128, n_segments=5, wavelet_name="morl", scales=np.arange(1, 65)
):
    band_ranges = {
        "delta": (0.5, 4),
        "theta": (4, 8),
        "alpha": (8, 13),
        "beta":  (13, 30),
        "gamma": (30, 45)
    }
    all_cwt_features = []
    info = mne.create_info(ch_names=ch_names, sfreq=fs, ch_types='eeg')
    raw = mne.io.RawArray(data, info)
    ica = mne.preprocessing.ICA(n_components=min(15, len(ch_names)), random_state=42, method='fastica')
    ica.fit(raw)
    sources = ica.get_sources(raw).get_data()
    stds = np.std(sources, axis=1)
    threshold = np.percentile(stds, 90)
    ica.exclude = [i for i, s in enumerate(stds) if s > threshold]
    raw_clean = ica.apply(raw.copy())
    total_samples = raw_clean.n_times
    seg_len = total_samples // n_segments
    for seg_idx in range(n_segments):
        start = seg_idx * seg_len
        stop = start + seg_len
        seg_data = raw_clean.get_data()[:, start:stop]
        for ch_idx, ch_name in enumerate(ch_names):
            signal = seg_data[ch_idx]
            # pywt removed: CWT feature extraction is disabled due to lack of pywt support on Python 3.13
            continue
                total_energy = np.sum(band_coefs ** 2)
                total_entropy = entropy(band_coefs / (np.sum(band_coefs) + 1e-12))
                coef_mean = np.mean(band_coefs)
                coef_std = np.std(band_coefs)
                coef_kurtosis = kurtosis(band_coefs)
                coef_skewness = skew(band_coefs)
                all_cwt_features.append({
                    "segment": seg_idx + 1,
                    "channel": ch_name,
                    "band": band,
                    "energy": total_energy,
                    "entropy": total_entropy,
                    "mean": coef_mean,
                    "std": coef_std,
                    "skewness": coef_skewness,
                    "kurtosis": coef_kurtosis
                })
    return all_cwt_features
