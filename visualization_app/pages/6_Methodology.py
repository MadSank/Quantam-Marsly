"""Methodology, Provenance & Scientific Limitations (Mission Console styling)."""

import sys
from pathlib import Path

# Add visualization_app to path
APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import streamlit as st
from style.theme import apply_theme
from components.layout import page_header, note_block, section_header
from components.navigation import render_walkthrough_footer

apply_theme()

# Standardized Mission Console Page Header
page_header(
    step=6,
    title="Scientific Methodology, Provenance & Limitations",
    subtitle="Comprehensive documentation of data sources, model architectures, statistical protocols, and documented caveats.",
    section_name="METHODOLOGICAL CAVEATS & PROVENANCE",
)

note_block(
    "integrity",
    "Executive Research Disclosure",
    "This dashboard presents experimental results from coursework evaluations of classical convolutional neural networks (CNNs) "
    "and variational quantum convolutional neural networks (QCNNs). All statements, metrics, and comparisons adhere strictly "
    "to the audited evidence documented in <code>FINAL_READINESS_AUDIT.md</code>."
)

# -----------------------------------------------------------------------------
# STRUCTURED METHODOLOGY TABS
# -----------------------------------------------------------------------------
tab_prov, tab_arch, tab_regime, tab_eval, tab_limits = st.tabs([
    "📁 Dataset Provenance & Spatial Structure",
    "📐 Model Architectures & Circuits",
    "⚖️ Training Regimes & Asymmetry",
    "📊 Evaluation Protocols & Statistics",
    "⚠️ Known Scientific Limitations",
])

with tab_prov:
    with st.container(border=True):
        st.markdown(
            """
            ### Mars HiRISE Orthoimage & DTM Dataset
            - **Source:** High Resolution Imaging Science Experiment (HiRISE) camera onboard NASA's Mars Reconnaissance Orbiter (MRO).
            - **Sample Count:** 15,233 valid coverage-eligible terrain patches cropped to 64×64 pixels at 2 m/pixel spatial resolution.
            - **Geographic Replication:** Distributed across 5 distinct landing sites on Mars:
              - `site_01`: Jezero Crater (2,195 patches)
              - `site_02`: Gale Crater (3,674 patches)
              - `site_03`: Mawrth Vallis (3,112 patches)
              - `site_04`: Oxia Planum (3,281 patches)
              - `site_05`: Eberswalde Crater (2,971 patches)
            - **Channels:** Channel 0 stores calibrated I/F (radiance/irradiance ratio), Channel 1 stores DTM validity masks, and Channel 2 stores image coverage validity masks.
            
            ---
            
            ### Moon LROCNet Dataset
            - **Source:** Lunar Reconnaissance Orbiter Camera (LROC) Narrow Angle Camera (NAC) surface images.
            - **Sample Count:** 5,000 images cropped to 227×227 RGB (grayscale content encoded in 3-channel RGB).
              - Training split: 3,608 images (358 ejecta, 594 oldcrater, 2,656 none)
              - Validation split: 613 images (55 ejecta, 118 oldcrater, 440 none)
              - Test split: 779 images (89 ejecta, 157 oldcrater, 533 none)
            - **Provenance Status:** `FILE_FACTS_CONFIRMED_SOURCE_ID_UNVERIFIED`.
              - While image dimensions, formats, and file hashes are confirmed, the original dataset release does not provide verified source observation IDs or geographic coordinates.
              - Filename prefixes (e.g., `M107280100RC`) and pyramid indices are unverified candidate strings; geographic independence between train and test splits cannot be verified.
            """
        )

with tab_arch:
    with st.container(border=True):
        st.markdown(
            """
            ### Mars Pipeline Architectures
            1. **Classical End-to-End Baseline (ResNet-18):**
               - Standard ResNet-18 architecture modified for single-channel grayscale input (channel 0).
               - Output head: Linear layer mapping 512-dimensional bottleneck features to 3 continuous terrain targets.
               - Total Trainable Parameters: **11,242,563**.
            2. **Hybrid Variational QCNN Head:**
               - Uses the representation learned by the classical ResNet-18 model, frozen after convergence.
               - Feature Bottleneck: Linear projection mapping 512 dimensions down to 8 dimensions with tanh activation.
               - Quantum Circuit: 8-qubit variational circuit using PennyLane with parameterized angle embedding ($\text{RY}$ rotations), alternating circular CNOT entanglement gates, and parameterized single-qubit rotations.
               - Measurement: Expectation values of Pauli-Z operators on all 8 qubits.
               - Readout Head: Small linear layer mapping 8 expectation values to 3 terrain targets (48 parameters).
               - Total Head Trainable Parameters: **4,152** (4,104 in projection + 48 in readout).

            ---

            ### Moon Pipeline Architectures
            1. **Classical CNN:**
               - Convolutional backbone with 3 convolutional blocks (Conv2d, BatchNorm, ReLU, MaxPool) followed by linear classifier.
               - Total Trainable Parameters during Training: **59,347**. (Note: the 195 reported in legacy JSON reflects only the classifier head after freezing the encoder).
            2. **Hybrid CNN-QCNN:**
               - Pre-trained CNN encoder frozen to extract 8-dimensional cached feature embeddings.
               - Quantum head: 8-qubit variational circuit with trainable rotation angles (**568** trainable parameters).
            3. **Matched Classical Control (Frozen-Embedding MLP):**
               - Multi-layer perceptron head with identical 8-dimensional input, matching capacity with **547** trainable parameters.
            """
        )

with tab_regime:
    with st.container(border=True):
        st.markdown(
            """
            ### Crucial Distinction: Pipeline Comparison vs Architecture Comparison
            - In both experiments, the classical CNN is trained **end-to-end** from randomly initialized weights using standard gradient backpropagation across millions/thousands of parameters.
            - In contrast, the hybrid QCNN is fitted as a **second-stage head on top of a frozen, supervised representation** generated by the trained CNN.
            - Therefore, these experiments evaluate whether an 8-qubit variational quantum circuit can act as an effective non-linear regressor/classifier on top of high-level classical embeddings.
            - **These results do NOT constitute an architecture-only comparison or a demonstration of quantum supremacy.**
            """
        )

with tab_eval:
    with st.container(border=True):
        st.markdown(
            """
            ### Mars Leave-One-Site-Out (LOSO) Protocol
            - **Primary Evaluation:** The 15-run repaired seedset (`runs/pipeline_repair_audit/retraining_seedset`) across 5 folds and 3 random seeds (42, 43, 44).
            - Five geological sites serve as the true replication units. In each fold, 3 sites are allocated to training, 1 site to validation, and 1 site strictly held out for testing.
            - **Spatial Correlation:** Patches within a single landing site share lighting, geological epoch, and sensor geometry. They are not independent statistical observations.
            - **Nominal Paired t-Intervals (degrees of freedom $df = 4$):** Paired differences are computed across the 5 site-level means. Because every paired regression and safety interval crosses zero (Slope: [-8.11, 3.99]°, Roughness: [-0.017, 0.009] m, Hazard: [-0.013, 0.004]), the data does not establish a statistically significant advantage for either model.
            - **Fixed-Threshold Trade-off:** CNN achieves higher safe recall (5.56% vs 2.77%) and safe coverage (5.67% vs 3.02%), while QCNN is more conservative with lower false positive rate (2.78% vs 5.27%).
            - **Domain Shift Under Validation Calibration:** When thresholds are calibrated on the validation site to target a 5% FPR, spatial domain shift causes test FPR to jump to **39.65%** for CNN and **33.94%** for QCNN on unseen test sites.

            ---

            ### Moon Multi-Seed Evaluation & Test Embedding Workflow Disclosure
            - Evaluated on a single fixed partition of 779 test images across 3 random seeds (42, 43, 44).
            - Seeds measure sensitivity to weight initialization and head optimization, not geographic or site generalization.
            - **Moon Test Embedding Computation Disclosure:** In the completed Moon experiment run (`experiment_lunar_cnn_qcnn`), test image embeddings (8-dimensional feature representations) were computed via the frozen CNN backbone prior to fitting the variational QCNN and MLP heads. Importantly, **test labels were never used** during head training, loss calculation, backpropagation, model checkpoint selection, or threshold tuning. The representations are deterministic, frozen feature mappings. However, in standard strict ML workflows, test data should remain untouched until final evaluation. This workflow execution artifact is disclosed here for complete scientific integrity.
            """
        )

with tab_limits:
    with st.container(border=True):
        st.markdown(
            """
            ### 1. "Not a Validated Landing Safety Assessment"
            - The safe/unsafe labels are computed from a bivariate proxy rule: `mean slope < 8.0°` and `hazard fraction < 0.05`.
            - Real spacecraft entry, descent, and landing (EDL) operations require terrain relative navigation, hazard detection algorithms, slope over multiple baselines, thermal inertia, radar reflectivity, and propulsion margins.
            - This proxy is an illustrative benchmark for spatial machine learning, **Not an operational landing safety certification**.

            ---

            ### 2. "No Quantum Speedup or Superiority Claimed"
            - All quantum circuits were executed via numerical state-vector simulation on classical CPU/GPU hardware.
            - **Latency Non-Comparability:** Recorded Moon inference times (e.g. ~4.6 ms for CNN vs ~2.8 ms for QCNN) cannot be compared directly: CNN includes full image encoding from 227×227 RGB pixels, whereas QCNN timing measured only head evaluation on pre-computed 8D embeddings.
            - No quantum speedup, quantum advantage, or quantum superiority is claimed or supported by this data.

            ---

            ### 3. "Model Capacity & Parameter Counts"
            - **Moon Classical CNN:** Trained with **59,347** trainable parameters during training (the 195 recorded in legacy JSON reflects only the linear classifier head post-encoder freeze).
            - **Moon Hybrid QCNN Head:** **568** trainable parameters in variational circuit rotations.
            - **Moon Frozen MLP Head:** **547** trainable parameters in matched control head.
            - **Mars ResNet-18:** **11,242,563** parameters trained end-to-end vs. **4,152** parameters in the QCNN projection and readout head.

            ---

            ### 4. "Severe Class Imbalance on Lunar Data"
            - The Moon dataset contains 73.6% `none` in training and 68.4% `none` in testing.
            - A trivial majority baseline achieves 68.4% raw accuracy but only 33.3% balanced accuracy and 0.271 macro-F1.
            - Evaluators must prioritize **Balanced Accuracy** (macro-averaged recall) and **Macro-F1** over raw accuracy.
            - Hybrid QCNN achieves higher mean balanced accuracy (0.608 vs 0.558) and macro-F1 (0.497 vs 0.465) while raw accuracy is virtually tied (0.529 vs 0.530). This is preliminary and does not constitute proven quantum advantage.
            """
        )

render_walkthrough_footer(
    current_page_num=6,
    current_page_name="Methodological Caveats & Provenance",
    next_page_path=None,
    next_page_name=None,
    prev_page_path="pages/5_Model_Comparison.py",
    prev_page_name="Aggregate Metrics, Error Bars & Trade-offs",
)
