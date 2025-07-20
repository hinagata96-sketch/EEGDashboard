import streamlit as st
from TDMI_app import run_tdmi_app
from fdmi_app import run_fdmi_app
from cwt_app import run_cwt_app
from dwt_app import run_dwt_app

st.set_page_config(page_title="EEG Feature Extraction Main Dashboard", layout="wide")
st.title("EEG Feature Extraction Main Dashboard")

st.markdown("""
Welcome to the unified EEG feature extraction dashboard. Select an analysis domain from the sidebar to launch the corresponding tool:
- **Time Domain (TDMI)**
- **Frequency Domain (FDMI)**
- **Continuous Wavelet Transform (CWT)**
- **Discrete Wavelet Transform (DWT)**
""")

app = st.sidebar.radio(
    "Choose Analysis Domain",
    ["Time Domain (TDMI)", "Frequency Domain (FDMI)", "Continuous Wavelet (CWT)", "Discrete Wavelet (DWT)"]
)

if app == "Time Domain (TDMI)":
    run_tdmi_app()
elif app == "Frequency Domain (FDMI)":
    run_fdmi_app()
elif app == "Continuous Wavelet (CWT)":
    run_cwt_app()
elif app == "Discrete Wavelet (DWT)":
    run_dwt_app()
