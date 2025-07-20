import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from TDMI import band_ranges, bandpass_filter, extract_time_features, extract_hjorth
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import mutual_info_classif
import zipfile
import io
import os

def run_tdmi_app():
    st.title("EEG Time-Domain Feature Extractor")
    st.markdown("""
**How segmentation and feature extraction works:**
- Each uploaded CSV file contains signals from 14 channels.
- For each channel, the signal is split into the number of segments you select above.
- Features are extracted from each segment of each channel.
- The results table shows features for every segment, channel, and file.
""")

    # Try to load the image and show a friendly error if it fails
    try:
        st.image("/Users/wankhai/Documents/EEGDashboard/TDSignals.png", caption="Time Domain EEG Signals", use_container_width=True)
    except Exception as e:
        st.warning(f"Could not load image: {e}")

    uploaded_zip = st.file_uploader("Upload ZIP file containing EEG CSVs (organized by class folders)", type=["zip"])

    fs = st.number_input("Sampling Frequency (Hz)", min_value=1, value=128)
    n_segments = st.number_input("Number of Segments", min_value=1, value=5)

    all_band_features = {band: [] for band in band_ranges}

    if uploaded_zip:
        with zipfile.ZipFile(uploaded_zip) as z:
            # Get all folders (classes)
            class_folders = set([os.path.dirname(f) for f in z.namelist() if f.lower().endswith('.csv')])
            for class_folder in class_folders:
                class_label = os.path.basename(class_folder)
                csv_files = [f for f in z.namelist() if f.startswith(class_folder + '/') and f.lower().endswith('.csv')]
                for csv_name in csv_files:
                    with z.open(csv_name) as f:
                        try:
                            df = pd.read_csv(f)
                        except UnicodeDecodeError:
                            f.seek(0)
                            df = pd.read_csv(f, encoding='latin1')
                        ch_names = df.columns.tolist()
                        data = df.values.T
                        total_samples = data.shape[1]
                        seg_len = total_samples // n_segments
                        for seg_idx in range(n_segments):
                            start = seg_idx * seg_len
                            stop = start + seg_len
                            seg_data = data[:, start:stop]
                            for ch_idx, ch_name in enumerate(ch_names):
                                signal = seg_data[ch_idx]
                                for band, (low, high) in band_ranges.items():
                                    if len(signal) > 27:
                                        filtered = bandpass_filter(signal, low, high, fs)
                                    else:
                                        filtered = signal
                                    if filtered.size == 0:
                                        continue  # Skip empty segments
                                    time_feats = extract_time_features(filtered)
                                    hjorth_feats = extract_hjorth(filtered)
                                    feature_dict = {
                                        "file": csv_name,
                                        "class": class_label,
                                        "segment": seg_idx + 1,
                                        "channel": ch_name,
                                        "band": band
                                    }
                                    feature_dict.update(time_feats)
                                    feature_dict.update(hjorth_feats)
                                    all_band_features[band].append(feature_dict)

    selected_band = st.selectbox("Select Frequency Band", list(band_ranges.keys()))
    features_df = pd.DataFrame(all_band_features[selected_band])
    st.write(f"Features for {selected_band} band:", features_df)

    # Add one-hot encoded class columns to features_df
    if not features_df.empty and 'class' in features_df.columns:
        class_dummies = pd.get_dummies(features_df['class'])
        # Convert boolean columns to int (0/1)
        class_dummies = class_dummies.astype(int)
        features_df = pd.concat([features_df, class_dummies], axis=1)
        st.write(f"Features for {selected_band} band (with one-hot class labels):", features_df)

        feature_cols = [col for col in features_df.columns if col not in ['file', 'class', 'segment', 'channel', 'band'] + list(class_dummies.columns)]
        scaler = MinMaxScaler()
        features_df[feature_cols] = scaler.fit_transform(features_df[feature_cols])
        st.write("Normalized Features (with one-hot class labels):", features_df)

        # Save button for current band
        csv = features_df.to_csv(index=False).encode('utf-8')
        st.download_button(f"Download Features CSV for {selected_band} Band", data=csv, file_name=f"features_{selected_band}.csv", mime="text/csv")

        if st.button("Compute MI Scores (One-vs-Rest)"):
            class_names = features_df['class'].unique()
            mi_results = []
            X = features_df[feature_cols].values
            for class_name in class_names:
                y_binary = (features_df['class'] == class_name).astype(int)
                mi_scores = mutual_info_classif(X, y_binary, discrete_features=False)
                for feat, score in zip(feature_cols, mi_scores):
                    mi_results.append({
                        "Class": class_name,
                        "Feature": feat,
                        "MI Score": score
                    })
            mi_df = pd.DataFrame(mi_results).sort_values(["Class", "MI Score"], ascending=[True, False])
            st.write("Mutual Information Scores (Feature vs Each Class, One-vs-Rest):", mi_df)
            for class_name in class_names:
                class_mi = mi_df[mi_df["Class"] == class_name]
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.barh(class_mi["Feature"], class_mi["MI Score"], color='mediumseagreen')
                ax.set_xlabel("Mutual Information Score")
                ax.set_title(f"MI Scores for {selected_band.upper()} Band vs {class_name}")
                ax.invert_yaxis()
                st.pyplot(fig)
    else:
        st.info("No features found for the selected band or file. Please check your data and selection.")
