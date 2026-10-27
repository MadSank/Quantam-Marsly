"""Mars + Moon CNN/QCNN Research Explorer

Main Streamlit Application Entrypoint (Mission Console redesign).
"""

import sys
from pathlib import Path

# Ensure visualization_app is in sys.path
APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import streamlit as st
from data_loading.paths import verify_critical_paths, PROJECT_ROOT
from style.theme import apply_theme
from components.layout import page_header, note_block, section_header

st.set_page_config(
    page_title="Mars & Moon Research Explorer",
    page_icon="🪐",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_theme()

# Standardized Mission Console Page Header
page_header(
    step=None,
    title="🪐 Mars & Moon CNN/QCNN Research Explorer",
    subtitle="Interactive scientific visualization platform for paired classical and quantum machine learning experiments.",
    section_name="MISSION CONSOLE ENTRYPOINT",
)

st.markdown(
    """
    Welcome to the **Mars + Moon Research Explorer**. This platform provides an interactive,
    audited interface to investigate experimental results from two deep learning and quantum neural network pipelines:

    1. **Mars Terrain Prediction (Regression & Landing-Safety Proxy)**:
       - Predicting 5 m mean slope, surface roughness, and hazard fraction from HiRISE orbital imagery.
       - Comparing an end-to-end ResNet-18 CNN against a supervised frozen-embedding QCNN (8-qubit variational circuit).
       - Evaluating leave-one-site-out generalization across 5 independent geological sites (15,233 patches).

    2. **Moon Crater Classification (LROCNet Benchmark)**:
       - Multi-class crater classification (`ejecta`, `oldcrater`, `none`) across 5,000 lunar surface images.
       - Comparing an end-to-end CNN, a frozen-embedding hybrid CNN-QCNN, and a parameter-matched frozen-embedding MLP control.
    """
)

# -----------------------------------------------------------------------------
# SYSTEM & ARTIFACT READINESS (Single Ruled Panel)
# -----------------------------------------------------------------------------
section_header("System & Artifact Readiness", "Verification of pre-computed experiment logs, parquet shards, and manifests.")

path_status = verify_critical_paths()
mars_ok = path_status.get("MARS_REPAIRED_ROOT", False) and path_status.get("MARS_SHARDS_DIR", False)
moon_ok = path_status.get("MOON_LOGS_DIR", False) and path_status.get("MOON_PROVENANCE_MANIFEST", False)
all_ok = all(path_status.values())

with st.container(border=True):
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Mars Artifacts", "Verified" if mars_ok else "Missing", "5 Sites / 15 Runs" if mars_ok else "Check Path")
    with col2:
        st.metric("Moon Artifacts", "Verified" if moon_ok else "Missing", "3 Seeds / 779 Test" if moon_ok else "Check Path")
    with col3:
        st.metric("Overall Integrity", "Ready" if all_ok else "Warnings Found", "Read-Only Mode")

# -----------------------------------------------------------------------------
# STRUCTURED COURSEWORK DEMONSTRATION FLOW (Interactive Route Table)
# -----------------------------------------------------------------------------
section_header("🎓 Coursework Demonstration Flow (6-Step Walkthrough)", "Follow this structured presentation sequence across the dashboard:")

steps = [
    ("Step 1", "Overview & Headline Results", "1_Overview.py", "Task framing, pipeline architectures, and verified headline metrics."),
    ("Step 2", "Training Data Explorer", "2_Training_Data_Explorer.py", "HiRISE calibrated I/F orthoimagery, geological diversity, and lunar distributions."),
    ("Step 3", "Mars Held-Out Test Explorer", "3_Mars_Test_Explorer.py", "Held-out patch inspection, demo presets, and deterministic numerical attribution."),
    ("Step 4", "Moon Multi-Model Test Explorer", "4_Moon_Test_Explorer.py", "Multi-model confidence comparison, crater disagreements, and raw test logits."),
    ("Step 5", "Model Comparison & Metrics", "5_Model_Comparison.py", "Continuous RMSE, zero-crossing paired t-intervals, safety trade-offs, and timing."),
    ("Step 6", "Scientific Methodology", "6_Methodology.py", "Provenance status, embedding disclosures, parameter audit corrections, and EDL caveats."),
]

with st.container(border=True):
    for num, title, page_file, desc in steps:
        c_num, c_link, c_desc = st.columns([1.2, 3.5, 6.3])
        with c_num:
            st.markdown(f"<span style='font-family: var(--mq-font-mono); font-size: 0.8rem; color: var(--mq-rust-hi); font-weight: 600;'>{num}</span>", unsafe_allow_html=True)
        with c_link:
            st.page_link(f"pages/{page_file}", label=title, width="stretch")
        with c_desc:
            st.markdown(f"<span style='font-size: 0.82rem; color: var(--mq-text-2);'>{desc}</span>", unsafe_allow_html=True)

col_action, _ = st.columns([2, 5])
with col_action:
    st.page_link("pages/1_Overview.py", label="Begin Coursework Demonstration (Step 1) ▶", icon="🚀", width="stretch")

# Research Integrity Notice
note_block(
    "integrity",
    "Research Integrity Notice",
    "All displays use genuine pre-computed checkpoints, predictions, and metrics. "
    "No models are retrained in this application, and test-site data is used strictly for evaluation and inspection."
)

# Sidebar metadata
st.sidebar.title("Navigation")
st.sidebar.info(
    "Select a section from the navigation menu above to explore data, individual test patches, aggregate metrics, and scientific methodology."
)
st.sidebar.markdown("---")
st.sidebar.caption(f"**Workspace:** `{PROJECT_ROOT}`")
