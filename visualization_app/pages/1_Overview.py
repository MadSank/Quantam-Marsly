"""Overview page: Executive summary, headline metrics, workflow diagrams, and research guardrails (Mission Console styling)."""

import sys
from pathlib import Path

# Add visualization_app to path
APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import streamlit as st
import pandas as pd
from style.theme import apply_theme, COLOR_CNN, COLOR_QCNN, COLOR_MLP, COLOR_BASELINE
from data_loading.mars_data import (
    load_mars_aggregate_metrics,
    load_mars_dataset_metadata,
)
from data_loading.moon_data import load_moon_recomputed_metrics
from components.layout import page_header, note_block, section_header
from components.navigation import render_walkthrough_footer

apply_theme()

# Standardized Mission Console Page Header
page_header(
    step=1,
    title="Mars & Moon CNN/QCNN Research Explorer",
    subtitle="Executive overview of experimental setups, verified headline results, and scientific guardrails.",
    section_name="TASK & ARCHITECTURE OVERVIEW",
)

# Prominent Research Integrity Callout Block
note_block(
    "integrity",
    "Research Integrity & Contextual Guardrails",
    '<ul style="margin: 0; padding-left: 20px;">'
    '<li><strong>Pipeline Comparison:</strong> In both experiments, the QCNN head is fitted on top of a <em>supervised, frozen representation</em> produced by a trained classical CNN. This evaluates whether a quantum circuit head can optimize representation mapping; it is <em>not</em> an end-to-end quantum supremacy or speedup claim.</li>'
    '<li><strong>Mars Safety Proxy:</strong> The "safe" label is a derived research proxy (<code>mean slope &lt; 8.0°</code> AND <code>hazard fraction &lt; 0.05</code>). It is <strong>not</strong> an operational landing safety certification.</li>'
    '<li><strong>Moon Parameter Count:</strong> The classical CNN contains <strong>59,347 trainable parameters during training</strong> (the 195 recorded in legacy JSON reflects only the classifier head post-encoder freeze).</li>'
    '<li><strong>Uncertainty Reporting:</strong> All intervals and deviations are empirical <strong>sample standard deviations (SD)</strong> across 5 geographic sites or 3 random seeds, not asymptotic confidence bounds.</li>'
    '</ul>'
)

# Walkthrough presentation guide (Collapsed by default to eliminate duplicate clutter)
with st.expander("Coursework Demonstration Walkthrough (6-Step Presentation Guide)", expanded=False):
    st.markdown(
        """
        Follow this curated 6-step walkthrough during your demonstration or assessment review:
        
        1. **Step 1: Task Framing & Headline Results (`1_Overview`)**
           - *Review:* Orbital dataset scale (15,233 Mars patches / 5,000 Moon images), end-to-end vs frozen-head pipeline architectures, and headline verified metrics.
        2. **Step 2: Dataset & Ground Truth Diversity (`2_Training_Data_Explorer`)**
           - *Review:* High-resolution HiRISE calibrated I/F orthoimagery across 5 distinct landing sites, DTM terrain features, and Moon crater class distributions.
        3. **Step 3: Mars Sample Inspection & Low-Risk Proxy (`3_Mars_Test_Explorer`)**
           - *Review:* Interactive patch browsing across held-out folds, demo presets (model disagreements, CNN false positives, QCNN conservatism), and deterministic numerical attribution.
        4. **Step 4: Moon Multi-Model Sample Inspection (`4_Moon_Test_Explorer`)**
           - *Review:* 3-model comparative confidence (CNN vs Hybrid QCNN vs MLP control), crater classification disagreements, confusion pair isolation, and raw test logits.
        5. **Step 5: Aggregate Metrics, Error Bars & Trade-offs (`5_Model_Comparison`)**
           - *Review:* Continuous regression RMSE bars (±1 Sample SD), zero-crossing paired t-intervals (*df*=4), safety trade-offs (QCNN lower FPR vs CNN higher safe recall), and execution timing.
        6. **Step 6: Methodological Caveats & Provenance (`6_Methodology`)**
           - *Review:* Dataset provenance, Moon test embedding computation disclosure, 59,347 trainable parameter correction, and operational landing proxy caveats.
        """
    )

# -----------------------------------------------------------------------------
# TWO-COLUMN EXECUTIVE SUMMARY PANELS
# -----------------------------------------------------------------------------
col_mars, col_moon = st.columns(2)

with col_mars:
    with st.container(border=True):
        st.subheader("🔴 Mars Terrain Prediction")
        st.markdown("<span style='font-size: 0.85rem; color: var(--mq-text-2);'>HiRISE Orthoimage & DTM Continuous Regression</span>", unsafe_allow_html=True)
        
        m_col1, m_col2, m_col3 = st.columns(3)
        m_col1.metric("Sites", "5 Sites", "Leave-One-Out")
        m_col2.metric("Patches", "15,233", "64×64 @ 2m/px")
        m_col3.metric("Runs", "15 Pairs", "5 Folds × 3 Seeds")
        
        st.markdown(
            """
            - **Primary Objective:** Predict terrain roughness, mean slope, and obstacle hazard fraction to evaluate safe landing proxy zones.
            - **Models Evaluated:** Randomly initialized ResNet-18 CNN (11.2M params) vs. Supervised Frozen Representation + 8-qubit QCNN head (4,152 params).
            - **Key Dynamic:** QCNN achieves lower regression error on all 3 continuous terrain targets across sites, but behaves more conservatively on safety thresholds (lower FPR, but lower safe recall/coverage).
            """
        )

with col_moon:
    with st.container(border=True):
        st.subheader("🌕 Moon Crater Classification")
        st.markdown("<span style='font-size: 0.85rem; color: var(--mq-text-2);'>LROC Lunar Reconnaissance Orbiter Morphology</span>", unsafe_allow_html=True)
        
        l_col1, l_col2, l_col3 = st.columns(3)
        l_col1.metric("Images", "5,000", "3608 tr / 613 v / 779 te")
        l_col2.metric("Classes", "3 Classes", "Ejecta / Old / None")
        l_col3.metric("Seeds", "3 Seeds", "Seeds 42, 43, 44")
        
        st.markdown(
            """
            - **Primary Objective:** Classify lunar morphological structures from orbital 227×227 grayscale-in-RGB imagery.
            - **Models Evaluated:** End-to-end CNN (59.3k trainable params), Hybrid CNN-QCNN (568 head params), and matched Classical Frozen-Embedding MLP control (547 head params).
            - **Key Dynamic:** The dataset is imbalanced (68.4% `none` in test). While the majority baseline gets 68.4% accuracy, its balanced accuracy is only 33.3%. Hybrid QCNN achieves higher balanced accuracy and macro-F1 than the CNN.
            """
        )

# -----------------------------------------------------------------------------
# PIPELINE ARCHITECTURE DIAGRAMS
# -----------------------------------------------------------------------------
section_header("Experimental Pipeline Architectures", "High-level representations of classical backbones and quantum circuit heads.")

tab_arch_mars, tab_arch_moon = st.tabs(["Mars Architecture Workflow", "Moon Architecture Workflow"])

with tab_arch_mars:
    with st.container(border=True):
        st.markdown(
            """
            ```mermaid
            graph LR
                A["HiRISE Calibrated I/F<br/>(64x64 float16 patch)"] --> B["ResNet-18 Classical Backbone<br/>(11,242,563 params)"]
                B --> C["512-dim Feature Vector"]
                
                subgraph "Branch 1: End-to-End CNN"
                    C --> D["Linear Regressor<br/>(1,539 params)"]
                    D --> E["Continuous Terrain Outputs:<br/>Slope, Roughness, Hazard"]
                end
                
                subgraph "Branch 2: Supervised Hybrid QCNN"
                    C --> F["Bottleneck Projection<br/>(512 -> 8 dims)"]
                    F --> G["Angle Embedding<br/>(8 Qubits)"]
                    G --> H["Variational Quantum Circuit<br/>(Entangling layers & rotations)"]
                    H --> I["Pauli-Z Expectation Values<br/>(8 Measurements)"]
                    I --> J["Linear Readout<br/>(8 -> 3 targets)"]
                    J --> E
                end
                
                E --> K{"Decision Proxy Rule<br/>Slope < 8° & Hazard < 0.05"}
                K --> L["Safe Proxy Classification<br/>(Safe vs Unsafe)"]
            ```
            """
        )

with tab_arch_moon:
    with st.container(border=True):
        st.markdown(
            """
            ```mermaid
            graph LR
                A["LROC Lunar Image<br/>(227x227 RGB)"] --> B["Classical CNN Feature Extractor<br/>(Conv + Pooling layers)"]
                
                subgraph "Stage 1: CNN Baseline"
                    B --> C1["Linear Classifier<br/>(3 classes)"]
                    C1 --> D1["Logits & Softmax Probabilities<br/>(59,347 params trained end-to-end)"]
                end
                
                subgraph "Stage 2: Representation Freeze"
                    B -.->|"Freeze Weights"| E["Cached 8D Feature Embeddings"]
                end
                
                subgraph "Stage 3: Head Comparison"
                    E --> F["8-Qubit QCNN Circuit Head<br/>(568 trainable params)"]
                    E --> G["Parameter-Matched MLP Head<br/>(547 trainable params)"]
                    F --> D2["Hybrid QCNN Predictions"]
                    G --> D3["MLP Control Predictions"]
                end
            ```
            """
        )

# -----------------------------------------------------------------------------
# VERIFIED HEADLINE EXPERIMENTAL RESULTS
# -----------------------------------------------------------------------------
section_header("Verified Headline Experimental Results", "Strictly verified aggregate numbers from offline repaired runs.")

tab_metrics_mars, tab_metrics_moon = st.tabs(["Mars Verified Results", "Moon Verified Results"])

with tab_metrics_mars:
    st.markdown("#### Mars Terrain Continuous & Safety Metrics (Repaired 5-Site / 3-Seed Aggregate)")
    
    mars_summary_data = [
        {
            "Terrain Target / Metric": "5 m Mean Slope (°)",
            "CNN (RMSE / MAE)": "10.250 / 8.712",
            "QCNN (RMSE / MAE)": "8.190 / 6.837",
            "Numerically Lower Error": "QCNN (-2.060° RMSE)",
            "Sample 95% t-Interval (Diff)": "[-8.11, 3.99] (crosses 0)",
        },
        {
            "Terrain Target / Metric": "Roughness (m)",
            "CNN (RMSE / MAE)": "1.566 / 1.172",
            "QCNN (RMSE / MAE)": "1.364 / 1.027",
            "Numerically Lower Error": "QCNN (-0.202 m RMSE)",
            "Sample 95% t-Interval (Diff)": "[-0.847, 0.444] (crosses 0)",
        },
        {
            "Terrain Target / Metric": "Hazard Fraction",
            "CNN (RMSE / MAE)": "0.453 / 0.392",
            "QCNN (RMSE / MAE)": "0.377 / 0.314",
            "Numerically Lower Error": "QCNN (-0.076 RMSE)",
            "Sample 95% t-Interval (Diff)": "[-0.274, 0.121] (crosses 0)",
        },
        {
            "Terrain Target / Metric": "Fixed Safety Recall (Safe)",
            "CNN (RMSE / MAE)": "5.56%",
            "QCNN (RMSE / MAE)": "2.77%",
            "Numerically Lower Error": "CNN higher recall",
            "Sample 95% t-Interval (Diff)": "[-0.111, 0.055] (crosses 0)",
        },
        {
            "Terrain Target / Metric": "Fixed Unsafe FPR (Hazard)",
            "CNN (RMSE / MAE)": "5.27%",
            "QCNN (RMSE / MAE)": "2.78%",
            "Numerically Lower Error": "QCNN lower FPR",
            "Sample 95% t-Interval (Diff)": "[-0.076, 0.026] (crosses 0)",
        },
        {
            "Terrain Target / Metric": "Fixed Safe Coverage",
            "CNN (RMSE / MAE)": "5.67%",
            "QCNN (RMSE / MAE)": "3.02%",
            "Numerically Lower Error": "CNN higher coverage",
            "Sample 95% t-Interval (Diff)": "[-0.098, 0.045] (crosses 0)",
        },
    ]
    st.dataframe(pd.DataFrame(mars_summary_data), width="stretch", hide_index=True)
    st.caption("Values verified from runs/pipeline_repair_audit/retraining_seedset/metrics/fixed_threshold_aggregate.csv and paired difference tables.")

with tab_metrics_moon:
    st.markdown("#### Moon Crater Classification Metrics (Verified Across Seeds 42, 43, 44)")
    
    moon_summary_data = [
        {
            "Model Variant": "Classical CNN (End-to-End)",
            "Accuracy (Mean ± SD)": "0.530 ± 0.035",
            "Balanced Accuracy (Mean ± SD)": "0.558 ± 0.053",
            "Macro-F1 (Mean ± SD)": "0.465 ± 0.024",
            "Trainable Parameters": "59,347 (during training)",
        },
        {
            "Model Variant": "Hybrid CNN-QCNN (Quantum Head)",
            "Accuracy (Mean ± SD)": "0.529 ± 0.039",
            "Balanced Accuracy (Mean ± SD)": "0.608 ± 0.006",
            "Macro-F1 (Mean ± SD)": "0.497 ± 0.026",
            "Trainable Parameters": "568 (head only)",
        },
        {
            "Model Variant": "Frozen-Embedding MLP (Control Head)",
            "Accuracy (Mean ± SD)": "0.451 ± 0.037",
            "Balanced Accuracy (Mean ± SD)": "0.587 ± 0.021",
            "Macro-F1 (Mean ± SD)": "0.440 ± 0.028",
            "Trainable Parameters": "547 (head only)",
        },
        {
            "Model Variant": "Train-Majority Baseline ('none')",
            "Accuracy (Mean ± SD)": "0.684 ± 0.000",
            "Balanced Accuracy (Mean ± SD)": "0.333 ± 0.000",
            "Macro-F1 (Mean ± SD)": "0.271 ± 0.000",
            "Trainable Parameters": "0 (deterministic)",
        },
    ]
    st.dataframe(pd.DataFrame(moon_summary_data), width="stretch", hide_index=True)
    st.caption("Values verified from data/raw/moon/lrocnet/experiment_lunar_cnn_qcnn/runs/results_verification/recomputed_metrics.csv.")

render_walkthrough_footer(
    current_page_num=1,
    current_page_name="Task & Architecture Overview",
    next_page_path="pages/2_Training_Data_Explorer.py",
    next_page_name="Dataset & Ground Truth",
    next_page_takeaway="Explore the raw HiRISE and LROC datasets, spatial distributions, class imbalances, and preprocessing pipelines.",
)
