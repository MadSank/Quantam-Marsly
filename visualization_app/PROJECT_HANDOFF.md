# Project Handoff: Mars + Moon CNN/QCNN Research Explorer

**Document Version:** 1.0  
**Target Audience:** Autonomous AI coding agents, software architects, and UI/UX designers  
**Target Repository:** `D:\Mars-QCNN`  
**Application Root:** `D:\Mars-QCNN\visualization_app`  
**Authoritative Audit Reference:** `D:\Mars-QCNN\FINAL_READINESS_AUDIT.md`  
**Current Date:** October 20, 2026  

---

## Executive Summary for Reviewing AI Models

This document is a self-contained, comprehensive technical specification and status report of the **Mars + Moon CNN/QCNN Research Explorer**, a scientific Streamlit dashboard implemented in `D:\Mars-QCNN\visualization_app`. 

It enables an AI assistant operating without direct filesystem access to understand:
1. The exact scientific tasks, architectures, datasets, and primary findings.
2. The current codebase structure, design patterns, and verified functionality.
3. The exact schemas and locations of all input artifacts (both used and unused).
4. All non-negotiable scientific reporting constraints and mandatory caveats.
5. Verified test evidence (27 unit/acceptance tests passing).
6. Concrete opportunities and safe implementation boundaries for subsequent UI/UX enhancements.

---

## A. Project Overview

### 1. What the Application Demonstrates
The dashboard provides an interactive, scientifically rigorous visualization platform for two machine learning research pipelines investigating classical Convolutional Neural Networks (CNNs) versus Variational Quantum Convolutional Neural Networks (QCNNs):

1. **Mars Terrain Prediction (Continuous Regression & Landing-Safety Proxy):**
   - High-resolution HiRISE orbital imagery and Digital Terrain Models (DTMs) from 5 Martian landing sites.
   - Evaluates whether an 8-qubit variational quantum circuit head fitted onto a supervised, frozen classical representation improves continuous terrain parameter estimation (slope, roughness, obstacle hazard) and landing-safety proxy classification across unseen geographic sites.
2. **Moon Crater Classification (LROCNet Multi-Class Benchmark):**
   - 5,000 Lunar Reconnaissance Orbiter Camera (LROC) images classified into `ejecta`, `oldcrater`, and `none`.
   - Benchmarks an end-to-end Classical CNN against a frozen-embedding Hybrid CNN-QCNN and an architectural matched classical control (Frozen-Embedding MLP).

### 2. Intended Audience & Presentation Use Case
- **Audience:** Academic reviewers, planetary scientists, quantum machine learning (QML) researchers, and coursework evaluators.
- **Use Case:** Coursework defense, experimental validation, peer-review demonstration, and error analysis. The dashboard allows non-technical and technical evaluators to inspect individual test samples, verify model predictions against ground truth, and explore aggregate statistical distributions.

### 3. Technology Stack
- **Framework:** Streamlit (>= 1.37.0) with native multipage routing.
- **Interactive Visualization:** Plotly (>= 5.22.0) for responsive charts, grouped bars with error bounds, and confusion matrix heatmaps.
- **Data Processing:** Pandas (>= 2.1.0), NumPy (>= 1.26.0), PyArrow (>= 14.0.0).
- **Image Processing:** Pillow (>= 10.0.0) for LROCNet JPEGs; memory-mapped NumPy for HiRISE `.npy` shards.
- **Machine Learning Utilities:** Scikit-learn (>= 1.3.0) for evaluation metric recomputation.
- **Configuration & Parsers:** PyYAML (>= 6.0.1).
- **Testing:** Pytest (>= 8.0.0).
- **Environment:** Dedicated Python 3.11 virtual environment under `visualization_app\.venv`.
- **Deliberate Absence:** No PyTorch, PennyLane, or GPU drivers are required at application runtime. All visualizations read strictly pre-computed predictions, metrics, and memory-mapped image shards (ensuring zero latency overhead and sub-second page loads).

---

## B. Current Application Structure

### 1. Folder Tree
```
D:\Mars-QCNN\visualization_app\
├── .venv/                         # Isolated Python 3.11 virtual environment
├── README.md                      # Operational setup and usage guide
├── requirements.txt               # Pinned runtime dependencies
├── app.py                         # Application entry point & artifact readiness monitor
├── data_loading/                  # Centralized data access layer
│   ├── __init__.py
│   ├── paths.py                   # Path constants & directory validation
│   ├── mars_data.py               # HiRISE shards, metadata & prediction loaders
│   └── moon_data.py               # LROCNet JSON, images & log loaders
├── style/                         # Aesthetics and CSS injection
│   ├── __init__.py
│   └── theme.py                   # Color tokens, badge HTML & styling rules
├── components/                    # Reusable UI widgets
│   ├── __init__.py
│   ├── image_cards.py             # Mars patch and lunar image cards
│   ├── filters.py                 # Pagination and filter widgets
│   ├── charts.py                  # Plotly grouped bars and line charts
│   ├── confusion_matrix.py        # Heatmap confusion matrix renderer
│   └── metric_tables.py           # Styled tabular displays & caveats
├── pages/                         # Multipage views
│   ├── 1_Overview.py              # Executive summary & headline results
│   ├── 2_Training_Data_Explorer.py# Dataset browser (15,233 Mars + 5,000 Moon)
│   ├── 3_Mars_Test_Explorer.py    # Mars held-out test patch inspector
│   ├── 4_Moon_Test_Explorer.py    # Moon test image multi-model inspector
│   ├── 5_Model_Comparison.py      # Statistical comparisons & loss curves
│   └── 6_Methodology.py           # Scientific limitations & provenance
└── tests/                         # Automated pytest acceptance test suite
    ├── test_m1_data_loading.py
    ├── test_m2_overview.py
    ├── test_m3_training_explorer.py
    ├── test_m4_mars_test_explorer.py
    ├── test_m5_moon_test_explorer.py
    ├── test_m6_model_comparison.py
    └── test_m7_methodology.py
```

### 2. Module & Page Responsibilities
- **`app.py`:** Main entry point. Initializes wide layout, injects global CSS, checks presence of critical Mars/Moon directories via `verify_critical_paths()`, and displays the system readiness dashboard.
- **`data_loading/paths.py`:** Defines absolute paths to experiment directories. Dynamically checks `$env:MARS_QCNN_ROOT` (fallback to project parent `D:\Mars-QCNN`).
- **`data_loading/mars_data.py`:** Houses `@st.cache_data` and `@st.cache_resource` loaders for Mars parquet metadata, `.npy` memory maps, paired test predictions, and aggregate metric CSVs.
- **`data_loading/moon_data.py`:** Loads per-seed `results.json`, JPEGs, `provenance_manifest.csv`, and training histories. Computes numerically stable softmax activations from saved logits and injects the corrected CNN parameter count (`59,347`).
- **`style/theme.py`:** Defines consistent color hex constants (`#4FC3F7` CNN Blue, `#AB47BC` QCNN Purple, `#78909C` MLP Gray, `#66BB6A` Safe Green, `#EF5350` Unsafe Red) and CSS classes.
- **`components/`:** Decoupled visual builders:
  - `charts.py`: Generates Plotly grouped bar figures with sample SD error bars and epoch loss curves.
  - `confusion_matrix.py`: Produces row-normalized 3×3 heatmaps with counts and percentages.
  - `metric_tables.py`: Formats paired differences (with explicit zero-crossing flags), parameter counts, and latency disclosures.
  - `filters.py`: Encapsulates pagination controls (`render_pagination_controls`).
  - `image_cards.py`: Standardizes image rendering with status badges.
- **`pages/1_Overview.py`:** Executive landing page with key statistics, Mermaid pipeline flowcharts, and verified headline metrics.
- **`pages/2_Training_Data_Explorer.py`:** Dual-tab dataset gallery allowing users to browse Mars patches (with 2nd–98th percentile stretch) and Moon crater images with manifest metadata.
- **`pages/3_Mars_Test_Explorer.py`:** Deep inspection of 15 fold/seed test sets. Shows paired ResNet-18 vs QCNN predictions against ground truth, absolute error bars, and a mathematical rule attribution table.
- **`pages/4_Moon_Test_Explorer.py`:** Side-by-side comparison of 3 models across 779 test images with softmax confidence bars, disagreement filters, and raw logit tables.
- **`pages/5_Model_Comparison.py`:** Quantitative evaluation center with 4 tabs: Mars regression/safety, Moon classification/confusion matrices, epoch training loss curves, and capacity/latency analysis.
- **`pages/6_Methodology.py`:** Academic documentation detailing data provenance, model architectures, spatial cross-validation, and mandatory scientific disclaimers.

### 3. User Navigation & Workflow
1. User enters on **`app.py`** and confirms all artifacts show **"Verified"**.
2. Evaluator visits **`1_Overview`** to understand the research scope, architectures, and headline findings.
3. Evaluator browses **`2_Training_Data_Explorer`** to understand raw sensor inputs (HiRISE 2 m/px I/F and LROC 227×227 RGB).
4. Evaluator navigates to **`3_Mars_Test_Explorer`** to filter test patches where CNN and QCNN disagree on safety, examining why each model passed or failed the threshold.
5. Evaluator navigates to **`4_Moon_Test_Explorer`** to inspect class predictions and softmax confidence distributions across models on difficult craters.
6. Evaluator analyzes **`5_Model_Comparison`** to inspect aggregate statistics, 95% t-intervals, and confusion matrices.
7. Evaluator reads **`6_Methodology`** for formal research disclosures and limitations.

### 4. Commands to Launch and Test
```powershell
# Activate dedicated application environment
cd D:\Mars-QCNN\visualization_app
.\.venv\Scripts\activate

# Launch Streamlit application (default port 8501)
.\.venv\Scripts\streamlit run app.py

# Run complete automated test suite (27 acceptance tests)
.\.venv\Scripts\pytest tests/ -v
```

---

## C. Existing Functionality

Every feature listed below has been implemented, executed, and verified live in the workspace.

### Page 1: Overview (`1_Overview.py`)
- **System Readiness Cards:** Shows patch counts (15,233 Mars, 5,000 Moon), site counts (5 Mars sites), and evaluated runs (15 Mars pairs, 3 Moon seeds).
- **Research Integrity Callout:** Prominently details pipeline comparison nature, safety proxy definitions, corrected Moon parameters, and sample SD reporting.
- **Interactive Architecture Flowcharts:** Rendered via Mermaid diagrams detailing:
  - Mars: HiRISE Patch (64×64) -> ResNet-18 Backbone (11.2M params) -> 512D Vector -> [Linear Head (1,539 params)] vs [8D Bottleneck -> 8-Qubit QCNN Circuit -> Linear Readout (4,152 params)] -> Bivariate Safe Rule.
  - Moon: LROC Image (227×227) -> CNN Feature Extractor -> [End-to-End Linear Classifier (59.3k params)] vs [Frozen 8D Embedding -> 8-Qubit QCNN Head (568 params) vs Matched MLP Head (547 params)].
- **Headline Result Tables:** Tables matching `FINAL_READINESS_AUDIT.md` for both Mars regression/safety and Moon classification.

### Page 2: Training Data Explorer (`2_Training_Data_Explorer.py`)
- **Mars Tab:**
  - Site selector (`site_01` to `site_05` or All).
  - Perspective Fold Selector (Folds 0 to 4): Dynamic banner indicates which site is Test, which is Validation, and which 3 are Training for that fold.
  - Role filter: Filter to view only Training, Validation, or Test patches for the selected perspective fold.
  - Safe Proxy filter: Safe only, Unsafe only, or All.
  - 12-item pagination controls with Prev/Next buttons.
  - 3×4 patch grid with per-tile 2nd–98th percentile stretch applied to channel 0.
  - Detailed inspection panel: enlarged 256px patch, ground truth safe/unsafe badge, continuous target table with threshold check (`PASS`/`FAIL`), and HiRISE sensor metadata (row, col, valid coverage, DTM/ortho product IDs).
- **Moon Tab:**
  - Split selector (`train`: 3,608, `val`: 613, `test`: 779, or All: 5,000).
  - Class selector (`ejecta`, `oldcrater`, `none`, or All).
  - 12-item pagination controls.
  - 3×4 image grid displaying actual 227×227 RGB JPEGs.
  - Detailed inspection panel: enlarged 280px image, ground truth class badge, provenance warning callout, dimensions, SHA-256 hash, and candid candidate filename prefix.

### Page 3: Mars Test Explorer (`3_Mars_Test_Explorer.py`)
- **Run Selectors:** Fold selector (0 to 4 with test site label) and Random Seed selector (42, 43, 44).
- **Summary Metrics Bar:** Test patches count (2,195 for Fold 0), filtered count, model agreement percentage, CNN safe accuracy, and QCNN safe accuracy.
- **Filter Controls:**
  - All Patches.
  - Models Disagree on Safe (`CNN != QCNN`).
  - Both Models Correct.
  - Both Models Wrong.
  - Ground Truth Safe / Unsafe.
  - CNN False Positive / False Negative.
  - QCNN False Positive / False Negative.
- **Sort Options:** Tile ID, CNN Slope Error, QCNN Slope Error, Slope Error Gap `|CNN - QCNN|`, CNN Hazard Error, QCNN Hazard Error.
- **Inspection View:**
  - HiRISE 256px patch with ground truth status and intra-site grid coordinates (`row`, `col`).
  - Paired Model Prediction Cards: Side-by-side ResNet-18 vs QCNN cards displaying predicted slope, roughness, hazard, safe badge, correctness badge, and signed error ($\Delta$).
  - Absolute Error Comparison Bar Chart: Plotly chart showing absolute errors for slope, roughness, and hazard on that specific patch.
  - Bivariate Threshold Rule Table: Full breakdown table showing slope check ($< 8.0^\circ$), hazard check ($< 0.05$), and combined outcome across CNN, QCNN, and Ground Truth.
  - Explainability Callout: Explains that the decision breakdown is an explicit mathematical post-processing rule, not a pixel heatmap.

### Page 4: Moon Test Explorer (`4_Moon_Test_Explorer.py`)
- **Evaluation Setup:** Random Seed selector (42, 43, 44).
- **Summary Metrics Bar:** 779 test images, CNN test accuracy, Hybrid QCNN accuracy, Frozen MLP accuracy, and total model disagreements.
- **Filter Controls:**
  - Class filter (`ejecta`, `oldcrater`, `none`).
  - Outcome filters: Disagreements only, Hybrid beats CNN, CNN beats Hybrid, All Correct, All Wrong, Model-specific correct/wrong.
- **Sort Options:** Sample order, CNN confidence, Hybrid confidence, MLP confidence, Hybrid vs CNN confidence gap.
- **Inspection View:**
  - Actual 227×227 lunar surface image.
  - Three Model Prediction Cards: Classical CNN, Hybrid QCNN, and Frozen MLP displaying predicted class, confidence percentage, and correct/wrong status.
  - Softmax Predicted Probability Distribution Chart: Plotly horizontal grouped bars showing probabilities allocated across all 3 classes for all 3 models.
  - Raw Logits & Derived Probabilities Table: Displays saved raw logits and normalized softmax percentages ($\sum p_i = 1.0$).
  - Explainability Callout: States that softmax values indicate predictive confidence, not causal explanations.

### Page 5: Model Comparison (`5_Model_Comparison.py`)
- **Tab 1: Mars Comparative Evaluation:**
  - Toggle between Fixed Threshold and Validation-Calibrated regimes.
  - Grouped bar chart of RMSE by target with sample SD error bars across 5 sites.
  - Grouped bar chart of safety rates (Recall, FPR, Coverage).
  - Paired Differences Table: Displays QCNN minus CNN mean difference, nominal 95% t-interval, and explicit "Crosses 0 (No clear winner)" indicator.
  - Expandable full aggregate and per-site/per-seed tables.
- **Tab 2: Moon Multi-Model Benchmark:**
  - Grouped bar chart comparing Accuracy, Balanced Accuracy, and Macro-F1 across 4 models (CNN, Hybrid QCNN, Frozen MLP, Majority Baseline).
  - Majority baseline callout explaining why 68.4% raw accuracy masks 33.3% balanced accuracy.
  - Seed-selectable Confusion Matrix Heatmaps: 3 Plotly heatmaps side-by-side with counts and row recall percentages.
  - Per-class Precision, Recall, and F1 table.
- **Tab 3: Training Epoch Curves:**
  - Mars subtab: Fold and seed selectable line plots of training and validation loss over 30 epochs for CNN vs QCNN.
  - Moon subtab: Seed selectable line plots of training loss and validation macro-F1 over epochs across models.
- **Tab 4: Capacity & Timing Analysis:**
  - Parameter count table highlighting verified CNN count (`59,347` during training) vs QCNN head (`4,152` Mars, `568` Moon).
  - Latency analysis table detailing the non-comparability caveat (CNN includes encoder, QCNN/MLP are head-only on cached embeddings).

### Page 6: Methodology & Limitations (`6_Methodology.py`)
- Detailed academic expanders covering:
  1. Dataset Provenance & Spatial Structure (HiRISE 5 landing sites, LROCNet unverified coordinates).
  2. Model Architectures & Quantum Circuits (PennyLane 8-qubit variational circuit with angle embedding and CNOT entanglement).
  3. Training Regimes & Methodological Asymmetry (pipeline comparison vs equal-capacity comparison).
  4. Evaluation Protocols & Spatial Statistics (LOSO cross-validation, 5-site degrees of freedom $df=4$, zero-crossing intervals).
  5. Known Scientific Limitations & Disclaimers (not landing safety, no quantum speedup, lunar class imbalance).

---

## D. Data and Artifact Map

### 1. Mars Experiment Artifacts

#### Data Source & Geometry
- **Location:** `D:\Mars-QCNN\runs\controlled_evaluation\processed`
- **Metadata:** `metadata.parquet` (15,233 rows × 31 columns).
  - Key columns: `tile_id`, `image_id`, `site_id` (`site_01` to `site_05`), `row`, `col`, `valid_coverage`, `dtm_valid_coverage`, `image_valid_coverage`, `mean_slope_5m`, `roughness_m`, `hazard_fraction`, `safe_label`, `shard_path`, `shard_idx`, `fold`.
  - Format: Fast, indexed Parquet file.
- **Image Shards:** `shards/site_{01-05}_tiles.npy`
  - Array shape: `(N, 3, 64, 64)` of dtype `float16`.
  - Channel 0: Calibrated I/F (radiance/irradiance ratio).
  - Channel 1: DTM elevation validity mask.
  - Channel 2: Image coverage validity mask.
- **Dataset Metadata:** `dataset_metadata.json` (describes patch GSD of 2.0 m/px, native HiRISE resolution, and site product IDs).

#### Repaired Retraining Seedset (Primary Mars Evaluation)
- **Root Directory:** `D:\Mars-QCNN\runs\pipeline_repair_audit\retraining_seedset`
- **Models:**
  - `model1_resnet18`: Randomly initialized ResNet-18 (11,242,563 trainable parameters).
  - `model2_qcnn`: Supervised frozen CNN representation + 8D bottleneck + 8-qubit variational circuit (4,152 trainable head parameters).
- **Run Grid:** 5 Folds (0 to 4) × 3 Seeds (42, 43, 44) = 15 runs per model (30 total runs).
- **Test Predictions:**
  - `runs/model1_resnet18/fold_{f}_seed_{s}/test_predictions.parquet`
  - `runs/model2_qcnn/fold_{f}_seed_{s}/test_predictions.parquet`
  - Schema (14 columns): `tile_id`, `image_id`, `fold`, `seed`, `model_name`, `variant`, `pred_mean_slope_5m`, `true_mean_slope_5m`, `pred_roughness_m`, `true_roughness_m`, `pred_hazard_fraction`, `true_hazard_fraction`, `pred_safe`, `true_safe`.
  - **Tile Alignment Verified:** In all 15 fold/seed pairs, CNN and QCNN parquet files have identical ordered `tile_id` records (45,699 total paired rows).
- **Validation Predictions:** `validation_predictions.parquet` in each run directory (same 14-column schema; currently unused by the UI).
- **Training Epoch Logs:** `epoch_log.csv` in each run directory (`epoch`, `train_loss`, `val_loss`, `lr`, `epoch_wall_time_sec`, `cumulative_wall_time_sec`).
- **Parameter Counts:** `param_counts.json` in each run directory.
- **Config Snapshots:** `configs/fold_{f}_seed_{s}.yaml` (all hyperparameter settings).
- **Completion Records:** `protocol/fold_{f}_seed_{s}_completion.json` (device, timing, split counts).
- **Aggregated Metrics (Metrics Directory):**
  - `metrics/fixed_threshold_aggregate.csv`: 33 rows × 7 columns (`model`, `metric`, `n_sites`, `mean_site_metric`, `sd_across_sites`, `t95_low`, `t95_high`).
  - `metrics/fixed_threshold_per_site_seed.csv`: 30 rows × 24 columns.
  - `metrics/fixed_threshold_site_mean.csv`: 10 rows × 24 columns.
  - `metrics/fixed_threshold_paired_site_differences.csv`: Paired site differences (QCNN minus CNN) and nominal 95% t-intervals.
  - `metrics/validation_calibrated_test_aggregate.csv`: Aggregate metrics under validation-selected operating points.
  - `metrics/validation_calibrated_test_paired_site_differences.csv`: Paired differences under validation calibration.
  - `metrics/thresholds_selected_on_validation.csv`: Selected slope and hazard thresholds per fold/seed.
  - `metrics/fpr_vs_safe_recall.png` and `fpr_vs_safe_coverage.png`: Pre-rendered curve plots.

#### Model Checkpoints (Unused at Runtime)
- `runs/model*/fold_*_seed_*/best_checkpoint.pt`: PyTorch weights (~45 MB for CNN, ~23 KB for QCNN). Intentionally not loaded to avoid runtime dependencies and live inference overhead.

---

### 2. Moon Experiment Artifacts (LROCNet)

#### Data Source & Geometry
- **Location:** `D:\Mars-QCNN\data\raw\moon\lrocnet`
- **Images:** 5,000 JPEG images at 227×227 RGB resolution:
  - `train/`: 3,608 images (`ejecta`: 358, `oldcrater`: 594, `none`: 2,656)
  - `val/`: 613 images (`ejecta`: 55, `oldcrater`: 118, `none`: 440)
  - `test/`: 779 images (`ejecta`: 89, `oldcrater`: 157, `none`: 533)
- **Provenance Manifest:** `provenance_manifest.csv` (5,000 rows × 15 columns).
  - Columns: `relative_image_path`, `existing_split`, `class_label`, `width_px`, `height_px`, `color_mode`, `sha256`, `source_observation_id`, `source_image_id`, `site_id`, `filename_prefix_candidate_unverified`, `pyr_index_candidate_unverified`, `exif_field_count`, `provenance_status`, `provenance_notes`.
  - **Confirmed Status:** `source_observation_id`, `source_image_id`, and `site_id` are strictly `NaN`/empty.

#### Experiment Directory
- **Root:** `D:\Mars-QCNN\data\raw\moon\lrocnet\experiment_lunar_cnn_qcnn`
- **Models:**
  - `cnn`: Classical end-to-end CNN (59,347 trainable parameters during training).
  - `hybrid_qcnn`: Frozen CNN embedding + 8-qubit variational circuit head (568 trainable parameters).
  - `frozen_embedding_mlp`: Matched classical control head on frozen embeddings (547 trainable parameters).
  - `majority_train_baseline`: Deterministic majority-class predictor (predicts `none`).
- **Results Logs:** `runs/logs/seed_{42,43,44}/results.json`
  - Per-seed structured JSON:
    - `test.sample_ids`: 779 relative paths (e.g. `test/ejecta/M1122721870LC_pyr-1052.jpg`).
    - `test.ground_truth`: 779 integers (`0: ejecta`, `1: oldcrater`, `2: none`).
    - `test.predictions`: Dictionary containing integer prediction arrays for `cnn`, `hybrid_qcnn`, `frozen_embedding_mlp`.
    - `test.logits`: Dictionary containing raw float arrays `[779, 3]` for each model.
    - `test.{model}.confusion_matrix`: 3×3 integer arrays.
    - `test.{model}.per_class`: Precision, recall, and F1 per class.
    - `test.inference.{model}`: Timing records (`total_seconds`, `milliseconds_per_image`).
    - `parameter_counts`: `{cnn: {total: 59347, trainable: 195}, hybrid_head: {total: 568, trainable: 568}, frozen_mlp_head: {total: 547, trainable: 547}}`.
- **Training Histories:** `runs/logs/seed_{42,43,44}/history.csv` (`model`, `epoch`, `train_loss`, `val_macro_f1`, `seconds`).
- **Verification Audit:**
  - `runs/results_verification/recomputed_metrics.csv`: Independently verified mean ± sample SD across seeds 42, 43, 44.
  - `runs/results_verification/results_audit.md`: Formal verification report proving metrics match saved logits.
  - `runs/metrics.csv`: 12 rows × 7 columns (per-seed metrics).

---

## E. Experiment Facts and Mandatory Reporting Constraints

To ensure scientific integrity, any future UI or text modifications must respect the following verified facts:

### 1. Mars Terrain Experiment Constraints
- **Primary Repaired Run:** Primary results are strictly those in `runs\pipeline_repair_audit\retraining_seedset`. Never mix historical one-seed results from `runs\controlled_evaluation` or uncalibrated `runs\real_v1` into primary displays.
- **Terrain Targets & Proxy Rule:** The 3 regression targets are continuous terrain measurements: 5 m mean slope (°), roughness (m), and hazard fraction. The safe-proxy rule is deterministically:
  $$\text{Safe Proxy} = (\text{Mean Slope} < 8.0^\circ) \land (\text{Hazard Fraction} < 0.05)$$
- **Not Validated Landing Safety:** This bivariate rule is an illustrative research proxy. It is not an operational entry, descent, and landing (EDL) flight assessment.
- **Statistical Significance (Zero-Crossing Intervals):** While QCNN achieves numerically lower mean RMSE on slope ($8.190^\circ$ vs $10.250^\circ$), roughness ($1.364$ m vs $1.566$ m), and hazard ($0.377$ vs $0.453$), the paired equal-site nominal 95% t-intervals (QCNN minus CNN) cross zero for every target (slope: $[-8.11, 3.99]$, roughness: $[-0.847, 0.444]$, hazard: $[-0.274, 0.121]$). **Do not declare QCNN a statistically proven winner.**
- **Safety Trade-off:** QCNN produces a lower false-positive rate ($2.78\%$ vs $5.27\%$), but also lower safe recall ($2.77\%$ vs $5.56\%$) and coverage ($3.02\%$ vs $5.67\%$). It is simply more conservative at the fixed threshold.
- **Validation Calibration Failure:** Validation-calibrated thresholds targeting $\le 5\%$ validation FPR failed to generalize, producing test FPRs of $39.65\%$ (CNN) and $33.94\%$ (QCNN) due to large inter-site domain shift.
- **Spatial Independence:** Patches within a site are spatially correlated. The 5 geographic sites are the true replication units. The 3 seeds do not increase geographic diversity.

### 2. Moon Crater Experiment Constraints
- **Class Imbalance:** `none` makes up 68.42% of the test partition. The majority-class baseline gets 68.4% raw accuracy but 33.3% balanced accuracy and 0.271 macro-F1. Never present raw accuracy without balanced accuracy or macro-F1.
- **Corrected CNN Parameter Count:** Display **59,347 trainable parameters** during CNN training. Explain that the 195 recorded in legacy JSON represents only the linear classifier after freezing the encoder.
- **Non-Comparable Latencies:** Classical CNN timing (~4.6 ms/img) includes end-to-end image feature extraction from raw pixels. QCNN timing (~2.8 ms/img) was measured on pre-cached 8D embeddings. **Never claim a quantum speedup or latency advantage.**
- **Test Embedding Computation Disclosure:** In the completed run, test embeddings were computed prior to fitting QCNN/MLP heads (though not used for gradient updates or model selection). State this accurately.
- **Unverified Provenance:** Provenance status is `FILE_FACTS_CONFIRMED_SOURCE_ID_UNVERIFIED`. Source observation IDs, image IDs, and geographic sites are unknown. Do not fabricate lunar site names or map coordinates.
- **Uncertainty Language:** All $\pm$ figures are empirical **sample standard deviations (SD)** across 3 random seeds or 5 sites, not asymptotic confidence bounds.

---

## F. Current Quality and Known Gaps

### 1. Existing Test Evidence
All 27 automated unit and acceptance tests pass in 1.25s:
```
tests\test_m1_data_loading.py::test_mars_metadata_loading PASSED         [  3%]
tests\test_m1_data_loading.py::test_mars_predictions_loading PASSED      [  7%]
tests\test_m1_data_loading.py::test_mars_patch_image_loading PASSED      [ 11%]
tests\test_m1_data_loading.py::test_mars_aggregate_metrics PASSED        [ 14%]
tests\test_m1_data_loading.py::test_moon_results_loading PASSED          [ 18%]
tests\test_m1_data_loading.py::test_moon_image_loading PASSED            [ 22%]
tests\test_m1_data_loading.py::test_moon_predictions_df_and_softmax PASSED [ 25%]
tests\test_m1_data_loading.py::test_moon_aggregate_metrics PASSED        [ 29%]
tests\test_m2_overview.py::test_mars_headline_metrics_match PASSED       [ 33%]
tests\test_m2_overview.py::test_moon_headline_metrics_match PASSED       [ 37%]
tests\test_m3_training_explorer.py::test_mars_site_selection_and_stretching PASSED [ 40%]
tests\test_m3_training_explorer.py::test_mars_fold_mapping PASSED        [ 44%]
tests\test_m3_training_explorer.py::test_moon_provenance_and_images PASSED [ 48%]
tests\test_m4_mars_test_explorer.py::test_mars_fold0_seed42_predictions PASSED [ 51%]
tests\test_m4_mars_test_explorer.py::test_mars_disagreement_filtering PASSED [ 55%]
tests\test_m4_mars_test_explorer.py::test_mars_both_wrong_filtering PASSED [ 59%]
tests\test_m4_mars_test_explorer.py::test_mars_threshold_consistency PASSED [ 62%]
tests\test_m4_mars_test_explorer.py::test_mars_error_sorting PASSED      [ 66%]
tests\test_m5_moon_test_explorer.py::test_moon_seed42_count PASSED       [ 70%]
tests\test_m5_moon_test_explorer.py::test_moon_softmax_sum_and_argmax PASSED [ 74%]
tests\test_m5_moon_test_explorer.py::test_moon_correctness_and_disagreements PASSED [ 77%]
tests\test_m5_moon_test_explorer.py::test_moon_ejecta_class_count PASSED [ 81%]
tests\test_m5_moon_test_explorer.py::test_moon_actual_jpeg_loads PASSED  [ 85%]
tests\test_m6_model_comparison.py::test_mars_paired_differences_cross_zero PASSED [ 88%]
tests\test_m6_model_comparison.py::test_moon_confusion_matrices_shape_and_sums PASSED [ 92%]
tests\test_m6_model_comparison.py::test_chart_builders_return_valid_figures PASSED [ 96%]
tests\test_m7_methodology.py::test_methodology_file_content PASSED       [100%]
============================= 27 passed in 1.25s ==============================
```

### 2. UI/UX Strengths
- **Sub-Second Cold & Warm Starts:** Because no heavy deep learning or quantum libraries are imported, Streamlit pages render instantly.
- **Memory-Mapped Efficiency:** Loading any Mars patch from 374 MB of `.npy` shards takes $< 2$ milliseconds via `mmap_mode='r'`.
- **Honest Visual Encoding:** Color tokens strictly separate classical and quantum models, while safe/unsafe badges make threshold compliance immediately obvious.
- **Zero Fabrication:** Missing lunar coordinates and unavailable quantum gradient heatmaps are explicitly stated in stylized callout boxes.

### 3. Confirmed Usability Deficiencies & Gaps
- **Large Dropdown Selectors:** In `3_Mars_Test_Explorer.py` and `4_Moon_Test_Explorer.py`, picking an individual patch requires scrolling through a `st.selectbox` containing up to 2,195 or 779 text items. This is functional but clumsy.
- **Lack of Intra-Site Spatial Overview:** In `metadata.parquet`, integer grid coordinates `row` and `col` exist for every patch. The app currently displays these as text fields (`Row 128, Col 4096`), missing an opportunity to render a 2D spatial scatter plot of the site showing patch positions colored by error or agreement.
- **No Direct Confusion Matrix Interactivity:** The Plotly confusion matrices on Page 5 are static heatmaps. Clicking a cell (e.g. true `oldcrater` predicted as `none`) does not link to the Moon Explorer to inspect those specific misclassified images.
- **Unused Validation Predictions:** `validation_predictions.parquet` (30 files) and `thresholds_selected_on_validation.csv` exist in the repository but are not currently inspectable in the UI.
- **No Export Options:** Users cannot download filtered data slices as CSV or export publication-quality figures directly from the UI.

---

## G. Recommended Enhancement Opportunities

For an AI planning subsequent UI and feature enhancements, here are prioritized, high-value recommendations:

### 1. Guided Scientific Demo Tour ("From Raw Pixels to Landing Proxy")
- **Concept:** Add a dedicated interactive walkthrough (or tab in Overview) that guides an evaluator step-by-step through a single representative patch:
  1. *Step 1: Raw Sensor Input:* Show channel 0 (I/F), channel 1 (DTM mask), and channel 2 (image mask).
  2. *Step 2: Backbone Feature Representation:* Explain how ResNet-18 maps 64×64 pixels to a 512D embedding, and how the bottleneck compresses it to 8 dimensions.
  3. *Step 3: Classical vs Quantum Head Regressions:* Compare continuous predictions for slope, roughness, and hazard.
  4. *Step 4: Deterministic Decision Boundary:* Show the bivariate threshold rule evaluating the patch, highlighting whether the model produced a true safe, false positive, or false negative.

### 2. Intra-Site Spatial Map (Grid Scatter Plot)
- **Concept:** In `3_Mars_Test_Explorer.py`, use the existing `row` and `col` columns from `metadata.parquet` to plot a 2D scatter plot representing the physical mosaic of the selected landing site.
- **Interaction:**
  - Color points by: Ground truth safety, Model agreement (`CNN == QCNN`), CNN Absolute Slope Error, or QCNN Absolute Slope Error.
  - Clicking a point or hovering over it displays the patch preview and selects it for detailed inspection.
  - **Honesty Rule:** Clearly label the axes as `Intra-Site Grid Row / Col (Native Tile Coordinates)`. Explicitly state that this is an unprojected local mosaic grid, not a global Mars geographic map.

### 3. Confusion Matrix Interactive Cross-Filtering
- **Concept:** On `4_Moon_Test_Explorer.py` (or `5_Model_Comparison.py`), provide a 3×3 clickable button matrix or selectbox matching the confusion matrix cells (e.g. `True: ejecta | Pred: none`).
- **Value:** Instantly filters the 779 test images to only those belonging to that specific error mode, allowing researchers to diagnose morphological features causing classification failure.

### 4. Direct Side-by-Side Dual Patch Comparison
- **Concept:** Allow evaluators in `3_Mars_Test_Explorer.py` to pin two patches side-by-side (e.g., comparing a high-slope crater rim with a flat landing plain, or comparing Seed 42 vs Seed 44 predictions on the same tile).

### 5. Responsive UI Polish & Publication Export
- **Layout:** Optimize containers for both 1920×1080 (desktop presentation) and 1366×768 (laptop display).
- **Exporting:** Add a `Download Filtered Data (CSV)` button on test explorer pages and exportable SVG/PNG buttons on Plotly charts.

---

## H. Safe Implementation Boundaries

Any future development must adhere to the following constraints:

1. **Strict File Isolation:** All code changes, dependencies, and generated assets must remain inside `D:\Mars-QCNN\visualization_app\`. Do not write to, edit, or delete files in `runs\`, `data\`, `src\`, `configs\`, or `tests\`.
2. **Zero Retraining:** Do not trigger model training, fine-tuning, or hyperparameter sweeps. All application features must read pre-computed artifacts.
3. **No Live Quantum Simulation at Runtime:** Do not add PennyLane, Qiskit, PyTorch, or GPU execution to the Streamlit app. The app's sub-second performance depends on reading pre-computed predictions.
4. **Preserve Test Suite Integrity:** The existing 27 acceptance tests in `visualization_app\tests\` must continue to pass (`pytest tests/ -v`). Any new feature should include dedicated acceptance tests.
5. **No Hallucinated Data:** Never synthesize artificial coordinates, fake confidence scores, or simulated gradient attribution maps. If metadata does not exist, provide an honest explanation or limitation box.
6. **Windows Path Resilience:** Always use `pathlib.Path` with forward slashes or raw strings to ensure cross-platform compatibility on Windows.

---

## I. Most Useful Files for Follow-Up Review

If an AI assistant or human reviewer is provided with a limited file budget, they should request these files in order of priority:

1. **`D:\Mars-QCNN\FINAL_READINESS_AUDIT.md`**  
   *Authoritative scientific report documenting experimental design, metrics, parameter corrections, and mandatory reporting limitations.*
2. **`D:\Mars-QCNN\visualization_app\README.md`**  
   *Quick start, launch instructions, and architecture summary.*
3. **`D:\Mars-QCNN\visualization_app\data_loading\paths.py`**  
   *Path definitions and directory structure mapping.*
4. **`D:\Mars-QCNN\visualization_app\data_loading\mars_data.py`**  
   *Mars shard memory-mapping, percentile stretching, and prediction merging logic.*
5. **`D:\Mars-QCNN\visualization_app\data_loading\moon_data.py`**  
   *Moon JSON results loader, softmax probability recovery, and parameter count correction.*
6. **`D:\Mars-QCNN\visualization_app\pages\3_Mars_Test_Explorer.py`**  
   *Implementation of the paired continuous prediction comparison and bivariate threshold explanation.*
7. **`D:\Mars-QCNN\visualization_app\pages\4_Moon_Test_Explorer.py`**  
   *Implementation of multi-model evaluation and softmax confidence visualizations.*
8. **`D:\Mars-QCNN\visualization_app\pages\5_Model_Comparison.py`**  
   *Implementation of aggregate metric comparisons, paired difference t-intervals, and confusion matrix heatmaps.*
9. **`D:\Mars-QCNN\visualization_app\tests\test_m4_mars_test_explorer.py`** & **`test_m5_moon_test_explorer.py`**  
   *Acceptance test specifications validating prediction integrity and filter behaviors.*
