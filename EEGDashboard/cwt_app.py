import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from CWT import extract_cwt_features
from sklearn.preprocessing import MinMaxScaler
import zipfile
import io
import os

def run_cwt_app():
    st.title("EEG Continuous Wavelet Transform (CWT) Feature Extractor")
    st.markdown("""
**How segmentation and feature extraction works:**
- Each uploaded CSV file contains signals from 14 channels.
- For each channel, the signal is split into the number of segments you select above.
- CWT features are extracted from each segment of each channel and frequency band.
- The results table shows features for every segment, channel, and file.
""")

    # Try to load the image and show a friendly error if it fails
    try:
        st.image("/Users/wankhai/Documents/EEGDashboard/CWT.png", caption="CWT Domain EEG Signals", use_container_width=True)
    except Exception as e:
        st.warning(f"Could not load image: {e}")

    uploaded_zip = st.file_uploader("Upload ZIP file containing EEG CSVs (organized by class folders)", type=["zip"])

    fs = st.number_input("Sampling Frequency (Hz)", min_value=1, value=128)
    n_segments = st.number_input("Number of Segments", min_value=1, value=5)
    wavelet_options = ["morl", "db4", "db6", "sym5", "sym8", "coif5", "coif3"]
    wavelet_name = st.selectbox("Wavelet Name", options=wavelet_options, index=0)
    scales = np.arange(1, 65)

    all_cwt_features = []

    if uploaded_zip:
        with zipfile.ZipFile(uploaded_zip) as z:
            class_folders = set([os.path.dirname(f) for f in z.namelist() if f.lower().endswith('.csv')])
            single_channel_files = []
            for class_folder in class_folders:
                class_label = os.path.basename(class_folder)
                csv_files = [
                    f for f in z.namelist()
                    if f.startswith(class_folder + '/')
                    and f.lower().endswith('.csv')
                    and not f.startswith('__MACOSX/')
                    and not os.path.basename(f).startswith('._')
                ]
                for csv_name in csv_files:
                    with z.open(csv_name) as f:
                        try:
                            df = pd.read_csv(f)
                        except UnicodeDecodeError:
                            f.seek(0)
                            df = pd.read_csv(f, encoding='latin1')
                        ch_names = df.columns.tolist()
                        # Exclude label columns if present
                        data_columns = [col for col in ch_names if col.lower() not in ['label', 'target']]
                        if len(data_columns) == 1:
                            single_channel_files.append(csv_name)
                            continue  # Skip feature extraction for single-channel files
                        data = df.values.T
                        features = extract_cwt_features(
                            data, ch_names, fs=fs, n_segments=n_segments, wavelet_name=wavelet_name, scales=scales
                        )
                        for feat in features:
                            feat["file"] = csv_name
                            feat["class"] = class_label
                            all_cwt_features.append(feat)
            if single_channel_files:
                st.warning(f"The following files have only one channel and were skipped (ICA requires at least 2 channels):\n" + '\n'.join(single_channel_files))

        selected_band = st.selectbox("Select Frequency Band", ["delta", "theta", "alpha", "beta", "gamma"])
        features_df = pd.DataFrame([f for f in all_cwt_features if f["band"] == selected_band])
        if not features_df.empty and 'class' in features_df.columns:
            class_dummies = pd.get_dummies(features_df['class'])
            # Ensure one-hot labels are 0/1 integers
            class_dummies = class_dummies.astype(int)
            features_df = pd.concat([features_df, class_dummies], axis=1)
            st.write(f"Features for {selected_band} band (with one-hot class labels):", features_df)
            feature_cols = [col for col in features_df.columns if col not in ['file', 'class', 'segment', 'channel', 'band'] + list(class_dummies.columns)]
            scaler = MinMaxScaler()
            features_df[feature_cols] = scaler.fit_transform(features_df[feature_cols])
            st.write("Normalized Features (with one-hot class labels):", features_df)
            # Download button for saving frequency band data
            csv_data = features_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label=f"Download {selected_band} Band Features as CSV",
                data=csv_data,
                file_name=f"cwt_{selected_band}_features.csv",
                mime="text/csv"
            )
            if st.button("Compute MI Scores (One-vs-Rest)"):
                class_names = features_df['class'].unique()
                mi_results = []
                X = features_df[feature_cols].values
                from sklearn.feature_selection import mutual_info_classif
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
