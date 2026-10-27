# UI/UX Audit & Redesign Handoff: Mars-QCNN Research Explorer

**Target System:** Google DeepMind / Gemini Argon (External Planning & Architecture AI)  
**Project Workspace:** `D:\Mars-QCNN`  
**Application Directory:** `D:\Mars-QCNN\visualization_app`  
**Audit Date:** October 20, 2026  
**Auditor:** Antigravity AI (Pair Programming Assistant)  
**Baseline Test Status:** ✅ **42 / 42 Tests Passed (1.90s)**  
**Screenshot Baselines:** ✅ **19 Full-Resolution PNGs Captured (1920×1080 & 1366×768)**  

---

## 1. Executive Summary & UI Architecture Overview

### 1.1 Scientific Purpose & System Context
The **Mars-QCNN Research Explorer** is an interactive scientific dashboard developed to analyze, audit, and demonstrate experimental results from two machine learning pipelines comparing classical Convolutional Neural Networks (CNNs) against Quantum Convolutional Neural Networks (QCNNs):

1. **Mars Terrain Prediction (Continuous Regression & Landing-Safety Proxy):**
   - **Data:** NASA Mars Reconnaissance Orbiter (MRO) HiRISE orbital orthoimagery and calibrated Digital Terrain Models (DTMs). 15,233 surface patches (64×64 pixels at 2 m/pixel resolution).
   - **Task:** Multitask continuous regression of 5 m mean slope [degrees °], surface roughness [metres m], and obstacle hazard fraction [unitless fraction $\in [0, 1]$].
   - **Evaluation Protocol:** 5-fold Leave-One-Site-Out (LOSO) spatial cross-validation across 5 distinct geographic landing sites (Jezero Crater, Gale Crater, Mawrth Vallis, Oxia Planum, Eberswalde Crater) across 3 random seeds (42, 43, 44), yielding 15 repaired paired runs.
   - **Decision Rule:** Bivariate operational proxy for landing feasibility: $\text{Safe} = (\text{Mean Slope} < 8.0^\circ) \land (\text{Hazard Fraction} < 0.05)$.
   - **Models:** End-to-end Classical ResNet-18 (11.24M parameters) vs. Supervised Frozen Representation + 8-qubit Variational QCNN Head (4,152 parameters).

2. **Moon Crater Morphological Classification (LROCNet Benchmark):**
   - **Data:** Lunar Reconnaissance Orbiter Camera (LROC) Narrow Angle Camera (NAC) surface images. 5,000 images cropped to 227×227 RGB.
   - **Task:** 3-class morphological crater classification: `ejecta` (young, high-reflectance crater rays), `oldcrater` (degraded rim), `none` (background lunar terrain).
   - **Evaluation Protocol:** 779 held-out test images evaluated across 3 random seeds (42, 43, 44).
   - **Models:** Classical CNN (59,347 trainable parameters during training), Hybrid CNN-QCNN (568 parameter variational circuit head on frozen 8D embeddings), and Matched Classical Control (Frozen-Embedding MLP with 547 head parameters).

### 1.2 Technology Stack
- **Framework:** Streamlit 1.66.0 (running in headless mode on Python 3.11.15)
- **Plotting & Visualization:** Plotly Graph Objects 7.1.0
- **Data Manipulation & Serialization:** Pandas 3.0.6, NumPy 2.4.6, PyArrow 26.0.0 (Parquet engine), PyYAML 6.0.3
- **Image Processing:** Pillow (PIL) 12.3.0
- **Testing & Verification:** Pytest 9.1.1, Streamlit AppTest (`streamlit.testing.v1`)

### 1.3 UI Directory Structure & Component Organization
```
D:\Mars-QCNN\visualization_app/
├── app.py                      # Main entrypoint, landing page, and system readiness panel
├── style/
│   ├── __init__.py
│   └── theme.py                # Color tokens, badge generators, and injected CUSTOM_CSS
├── components/
│   ├── __init__.py
│   ├── charts.py               # Plotly figure builders (regression bars, safety bars, loss curves, grid scatter)
│   ├── confusion_matrix.py     # Plotly 3x3 heatmap renderer with normalized annotations
│   ├── filters.py              # Pagination and item stepping controls
│   ├── image_cards.py          # Terrain patch and crater image card containers
│   ├── metric_tables.py        # Capacity, parameter count, timing, and paired diff tables
│   └── navigation.py           # Coursework demonstration walkthrough footer buttons
├── data_loading/
│   ├── __init__.py
│   ├── paths.py                # Centralized path resolution and artifact verification
│   ├── mars_data.py            # Cached loaders for parquet, shards, predictions, metrics, epoch logs
│   └── moon_data.py            # Cached loaders for JSON logs, provenance, images, softmax computation
├── pages/
│   ├── 1_Overview.py           # Task framing, Mermaid architectures, verified headline tables
│   ├── 2_Training_Data_Explorer.py # HiRISE and Moon image browsing, metadata inspection
│   ├── 3_Mars_Test_Explorer.py # Held-out test patches, demo presets, error breakdown, decision attribution
│   ├── 4_Moon_Test_Explorer.py # 3-model crater classification, confidence comparison, confusion pairs
│   ├── 5_Model_Comparison.py  # Aggregate metrics, training curves, capacity & timing analysis
│   └── 6_Methodology.py        # Provenance status, quantum caveats, parameter corrections
└── tests/
    ├── test_m1_data_loading.py
    ├── test_m2_overview.py
    ├── test_m3_training_explorer.py
    ├── test_m4_mars_test_explorer.py
    ├── test_m5_moon_test_explorer.py
    ├── test_m6_model_comparison.py
    └── test_m7_methodology.py
```

---

## 2. Inventory of All Six Pages & Entrypoint

The application adheres to Streamlit's native multi-page convention, where `app.py` serves as the entrypoint (`/`) and pages in `pages/` are numbered to establish the navigation order.

| Page Identifier | Source File | Sidebar Label / URL Route | Primary Title Rendered | Core Purpose & Interaction Flow |
| :--- | :--- | :--- | :--- | :--- |
| **Landing** | `app.py` | *(Root entrypoint)* `/` | `🪐 Mars & Moon CNN/QCNN Research Explorer` | System readiness check, verification of critical paths, presentation flow guide, and workspace metadata. |
| **Page 1** | `pages/1_Overview.py` | `1_Overview` `/Overview` | `📊 Mars & Moon CNN/QCNN Research Explorer` | Executive summary, research integrity callouts, 6-step walkthrough accordion, pipeline architecture Mermaid diagrams, verified headline metric comparison tables. |
| **Page 2** | `pages/2_Training_Data_Explorer.py` | `2_Training_Data_Explorer` `/Training_Data_Explorer` | `🖼️ Training & Dataset Explorer` | Dataset browsing across 15,233 Mars HiRISE patches (with site/fold/safe filters) and 5,000 Moon crater images (split/class filters), with 3×4 paginated thumbnail grids and detailed side-by-side patch inspection cards. |
| **Page 3** | `pages/3_Mars_Test_Explorer.py` | `3_Mars_Test_Explorer` `/Mars_Test_Explorer` | `🔴 Mars Held-Out Test Explorer` | Interactive held-out test patch inspection across 5 folds and 3 seeds. Quick-select demo presets, single-patch stepping, side-by-side error comparison cards, intra-site 2D mosaic grid scatter plot, deterministic threshold-based attribution table, and CSV download. |
| **Page 4** | `pages/4_Moon_Test_Explorer.py` | `4_Moon_Test_Explorer` `/Moon_Test_Explorer` | `🌕 Moon Test Explorer` | Evaluation of 779 held-out test images across 3 seeds. Disagreement filtering, confusion pair isolation (e.g. `oldcrater → ejecta`), quick-select presets, 3-model confidence cards (CNN vs Hybrid QCNN vs MLP), softmax probability bar chart, raw logits table, and CSV export. |
| **Page 5** | `pages/5_Model_Comparison.py` | `5_Model_Comparison` `/Model_Comparison` | `📈 Model Comparison & Aggregate Metrics` | 4-tab evaluation: (1) Mars continuous RMSE & safety proxy charts; (2) Moon benchmark bar charts & 3×3 confusion matrix heatmaps; (3) Mars/Moon training epoch loss curves; (4) Trainable parameter verification & timing analysis tables. *(Note: Statistical assessment section was removed).* |
| **Page 6** | `pages/6_Methodology.py` | `6_Methodology` `/Methodology` | `🔬 Scientific Methodology, Provenance & Limitations` | Comprehensive scientific disclosures: HiRISE/LROC provenance, architecture specs, training regime asymmetries, spatial LOSO protocol, EDL proxy caveats, quantum simulation notes, parameter corrections, and Moon test embedding disclosures. |

---

## 3. Existing Design-System Inventory

### 3.1 Color Palette Tokens (`style/theme.py`)

The application enforces a dark UI palette with color-coded semantics:

```python
# Model Identity Palette
COLOR_CNN = "#4FC3F7"        # Light Cyan / Sky Blue (Classical ResNet-18 & CNN backbone)
COLOR_QCNN = "#AB47BC"       # Medium Orchid / Violet Purple (Hybrid Variational QCNN Head)
COLOR_HYBRID = "#AB47BC"     # Alias for QCNN
COLOR_MLP = "#78909C"        # Slate Gray / Cool Steel (Matched Frozen-Embedding MLP Control)
COLOR_BASELINE = "#FFB74D"   # Amber Gold (Train-majority Baseline)

# Semantic & Scientific Status Palette
COLOR_SAFE = "#66BB6A"       # Emerald Green (Safe landing proxy, correct classification, agree)
COLOR_UNSAFE = "#EF5350"     # Crimson Coral (Unsafe landing proxy, incorrect prediction, fail)
COLOR_AGREE = "#66BB6A"      # Green (Models agree)
COLOR_DISAGREE = "#FFA726"   # Warning Amber (CNN != QCNN model disagreement)
COLOR_CORRECT = "#66BB6A"    # Green (Correct prediction vs ground truth)
COLOR_INCORRECT = "#EF5350"  # Red (Incorrect prediction vs ground truth)

# Canvas & Neutral Palette
COLOR_BG_CARD = "#181C24"    # Charcoal Card Surface
COLOR_BORDER = "#2B3240"     # Slate Card Border
COLOR_TEXT = "#E2E8F0"       # Off-white High-Contrast Typography
COLOR_MUTED = "#94A3B8"      # Slate Gray Muted / Secondary Typography
```

### 3.2 Injected CSS Rules (`style/theme.py` `CUSTOM_CSS`)

The global stylesheet injected via `apply_theme()` defines:
1. `.reportview-container`: Background `#0E1117`.
2. `.stMetric`: Background `#181C24`, padding `12px 16px`, border-radius `8px`, border `1px solid #2B3240`.
3. `.metric-card`: Background `#181C24`, border `1px solid #2B3240`, border-radius `8px`, padding `16px`, margin-bottom `12px`.
4. `.badge`: Inline-block, padding `3px 8px`, font-size `0.75rem`, font-weight `600`, border-radius `4px`, uppercase, letter-spacing `0.5px`.
   - `.badge-safe`, `.badge-correct`: Green tint `rgba(102, 187, 106, 0.2)` with `#66BB6A` border.
   - `.badge-unsafe`, `.badge-incorrect`: Red tint `rgba(239, 83, 80, 0.2)` with `#EF5350` border.
   - `.badge-disagree`: Amber tint `rgba(255, 167, 38, 0.2)` with `#FFA726` border.
   - `.badge-model-cnn`: Blue tint `rgba(79, 195, 247, 0.15)` with `#4FC3F7` border.
   - `.badge-model-qcnn`: Purple tint `rgba(171, 71, 188, 0.2)` with `#AB47BC` border.
   - `.badge-model-mlp`: Gray tint `rgba(120, 144, 156, 0.2)` with `#78909C` border.
5. `.callout-box`: Purple accent border `border-left: 4px solid #AB47BC`, background `#181C24`, padding `12px 16px`, border-radius `0 6px 6px 0`, margin `12px 0`.
6. `.callout-warning`: Amber accent border `border-left: 4px solid #FFA726`, background `rgba(255, 167, 38, 0.08)`, padding `12px 16px`, border-radius `0 6px 6px 0`.

### 3.3 Plotly Chart Theming Convention
Every chart generated in `components/charts.py` and `components/confusion_matrix.py` manually configures:
- `paper_bgcolor="#0E1117"`
- `plot_bgcolor="#181C24"`
- `font=dict(color="#E2E8F0", size=12)`
- Gridlines: `gridcolor="#2B3240"`, `zerolinecolor="#2B3240"`
- Margins: Compact margins (`t=50-75`, `b=40-60`, `l=40-60`, `r=20-40`)
- Hover templates: Customized `<extra></extra>` strings with bold titles and formatted numerical precision.

---

## 4. Page-by-Page Visual & Usability Audit

### 4.1 Landing Entrypoint (`app.py`)
- **Visual Structure:** Title with planet emoji, research description, a large purple-bordered 6-step coursework demonstration flow callout box, sidebar workspace path, and a 3-column system readiness metric strip (`col1, col2, col3`).
- **Strengths:** Immediate, clear verification that all repaired Mars and Moon artifacts exist and are loaded in read-only mode.
- **Usability Issues & Flaws:**
  - *Dead End on Main Canvas:* The page displays readiness and instructions but provides no interactive button or call-to-action on the canvas to begin the walkthrough (the user must look to the sidebar).
  - *Hardcoded HTML Callout:* The demonstration guide uses inline HTML `<div class="callout-box">` with an `<ol>` list that feels somewhat crowded on narrow screens.
  - *Excessive Whitespace:* At 1920×1080, the bottom half of the canvas below the readiness metrics is completely blank except for a markdown blockquote.

### 4.2 Page 1: Overview (`pages/1_Overview.py`)
- **Visual Structure:** Title, prominent research integrity callout box, collapsible 6-step walkthrough guide, 2-column executive summary card strip (Mars vs Moon), Mermaid architecture diagrams (tabs), and verified headline results tables (tabs).
- **Strengths:** Thorough scientific framing; clear visual separation of Mars regression and Moon classification.
- **Usability Issues & Flaws:**
  - *Redundant Walkthrough Accordion:* The 6-step walkthrough is displayed both on `app.py` and expanded by default on `1_Overview.py`, leading to repetitive visual clutter.
  - *Table Formatting:* Headline results are rendered via `st.dataframe` with raw strings, lacking visual metric badges, color-coded deltas, or progress indicators.
  - *Mermaid Chart Rendering:* In Streamlit dark mode, raw Mermaid diagrams can render with default bright white boxes or plain styling depending on the client browser theme, causing visual dissonance with `#181C24` card backgrounds.

### 4.3 Page 2: Training Data Explorer (`pages/2_Training_Data_Explorer.py`)
- **Visual Structure:** Two primary tabs ("Mars HiRISE Patches" vs "Moon LROCNet Images"). Left pane: 3×4 image grid (12 images) with status badges and inspect buttons; Right pane: single-patch deep inspection card, criteria check table, and geospatial/sensor metadata expander.
- **Strengths:** Fast loading of 64×64 Mars patches directly from memory-mapped `.npy` shards; clean 2%–98% percentile stretching of HiRISE float16 radiance values.
- **Usability Issues & Flaws:**
  - *Card Clutter & Button Stacking:* Each cell in the 3×4 grid contains an image, a raw HTML badge flexbox, a text caption, and an `Inspect Patch` button. At 1366×768, these 12 buttons create heavy visual noise and vertical stretching.
  - *Inspection State Reset:* Clicking `Inspect Patch` updates `st.session_state["inspected_mars_row"]`, but pagination controls can cause the detail pane to show an item from a different page than the active grid unless synchronized.
  - *Dual Controls:* Perspective fold filtering in the sidebar interacts with split role filters, which can confuse users if they select "Fold 0" and then filter for "Validation".

### 4.4 Page 3: Mars Test Explorer (`pages/3_Mars_Test_Explorer.py`)
- **Visual Structure:** Top 5-metric strip (`total_test`, `filtered`, `agreement`, `cnn_acc`, `qcnn_acc`), horizontal radio bar for Quick-Select Demo Presets, Prev/Next stepping buttons flanking a selectbox dropdown, CSV export button, selected patch header banner, 2-column layout (patch image + ground truth on left; paired ResNet-18 vs QCNN prediction cards + absolute error bar chart on right), collapsible 2D intra-site mosaic grid scatter plot, and a deterministic decision attribution table.
- **Strengths:** Excellent scientific transparency; deterministic mathematical attribution clearly explains the safe/unsafe classification without fabricating gradient maps; intra-site scatter plot correctly displays unprojected tile offsets with inverted row axis.
- **Usability Issues & Flaws:**
  - *Vertical Scrolling Fatigue:* The page is extremely tall (over 2,500px rendered height). The user must scroll through the metric strip, presets, patch selector, image/cards, error bar chart, grid scatter plot, and decision table.
  - *Fragmented Card Styling:* The Classical CNN card has an inline border `#4FC3F7` and the QCNN card has `#AB47BC`, but the internal tables use raw HTML `<table>` elements with hardcoded inline styles (`padding: 3px 0; border-bottom: 1px solid #2B3240`).
  - *Intra-Site Grid Position:* The 2D mosaic scatter plot is tucked inside a closed expander (`expanded=False`), meaning presenters frequently miss it unless they remember to expand it.

### 4.5 Page 4: Moon Test Explorer (`pages/4_Moon_Test_Explorer.py`)
- **Visual Structure:** 5-metric strip, horizontal radio demo preset bar, Prev/Next stepping controls, CSV export button, 2-column inspection layout (lunar image + ground truth on left; 3-column model prediction cards for CNN, Hybrid QCNN, and Frozen MLP + grouped softmax probability bar chart on right), and a raw logits & probabilities table.
- **Strengths:** Direct comparison of the 3 models on identical test samples; clear softmax confidence breakdown across `ejecta`, `oldcrater`, and `none`.
- **Usability Issues & Flaws:**
  - *Horizontal Cramping at 1366×768:* The 3-column model cards (`c_cnn, c_hyb, c_mlp`) each have a title, badge, confidence percentage, and correct/wrong badge. At 1366×768, this 3-column container becomes tight and text can wrap awkwardly.
  - *Logit Table Formatting:* The raw logit table shows 7 columns in an unstyled dataframe. A cleaner visual comparison (e.g. side-by-side gauge or delta bar) would communicate confidence differences more effectively.
  - *Preset Feedback:* When selecting a demo preset (e.g. "Hybrid QCNN Correct / Classical CNN Wrong"), a green `st.success` banner appears, which pushes the stepping controls down further.

### 4.6 Page 5: Model Comparison & Aggregate Metrics (`pages/5_Model_Comparison.py`)
- **Visual Structure:** 4 tabs:
  - Tab 1 (Mars Comparative Evaluation): Operating regime toggle (Fixed vs Validation-Calibrated), continuous RMSE bar chart, landing-safety proxy rate bar chart, expandable full metric tables.
  - Tab 2 (Moon Multi-Model Benchmark): Grouped benchmark bar chart (Accuracy, Balanced Accuracy, Macro-F1), evaluation seed selectbox, 3 side-by-side confusion matrix heatmaps, per-class metrics table.
  - Tab 3 (Training Epoch Curves): Sub-tabs for Mars epoch loss curves (CNN vs QCNN side-by-side) and Moon training loss / validation macro-F1 curves.
  - Tab 4 (Capacity & Timing Analysis): Trainable parameter count table, latency non-comparability warning callout, Moon test embedding disclosure, 2-column timing breakdown (Moon latency & fit duration on left; Mars training wall-clock on right).
- **Strengths:** Dense, authoritative scientific evidence; correctly represents sample standard deviations (SD); provides clear disclosures regarding latency non-comparability.
- **Usability Issues & Flaws:**
  - *Confusion Matrix Squeeze:* Rendering three Plotly heatmaps side-by-side in `col_cm1, col_cm2, col_cm3` makes the 3×3 heatmaps tiny (~300px wide each), causing tick labels (`Pred: oldcrater`) to rotate and cell text (`Count (Pct%)`) to become cramped.
  - *Removed Statistical Section Footprint:* In Tab 1, the paired-site statistical assessment was removed, leaving the continuous RMSE chart and safety chart without an immediate summary table in the main view (metrics are hidden inside expanders).
  - *Sub-Tab Inconsistencies:* Tab 3 contains sub-tabs for Mars and Moon, whereas Tabs 1 and 2 are top-level tabs, creating nested tab navigation.

### 4.7 Page 6: Scientific Methodology & Limitations (`pages/6_Methodology.py`)
- **Visual Structure:** Executive research disclosure callout box, 5 large expandable sections covering dataset provenance, model architectures, training regimes, evaluation protocols, and known scientific limitations.
- **Strengths:** Completely transparent scientific disclosure; meticulously documents all caveats from `FINAL_READINESS_AUDIT.md`.
- **Usability Issues & Flaws:**
  - *Text-Heavy Wall of Expanders:* All 5 expanders are set to `expanded=True` by default, creating a massive single-column wall of text that requires extensive scrolling.
  - *Lack of Visual Aids:* Unlike Page 1 (which has Mermaid diagrams) or Page 5 (which has charts), Page 6 contains no diagrams, callout badge tables, or structural graphics.

---

## 5. Shared Components & Dependencies Map

The application isolates reusable UI patterns inside `visualization_app/components/`. Gemini Argon should review their call sites and dependencies before planning changes:

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                   SHARED UI COMPONENT MAP                                        │
├───────────────────────────────┬───────────────────────────────────┬──────────────────────────────┤
│ Component Module & Functions  │ Used By Pages                     │ Primary Inputs & Return      │
├───────────────────────────────┼───────────────────────────────────┼──────────────────────────────┤
│ components/charts.py          │                                   │                              │
│  - build_mars_regression_chart│ 5_Model_Comparison.py             │ (df, metric_type) -> Figure  │
│  - build_mars_safety_chart    │ 5_Model_Comparison.py             │ (df) -> Figure               │
│  - build_moon_comparison_chart│ 5_Model_Comparison.py             │ (recomputed_df) -> Figure    │
│  - build_loss_curve_chart     │ 5_Model_Comparison.py             │ (epochs, train, val) -> Fig  │
│  - build_mars_grid_scatter    │ 3_Mars_Test_Explorer.py           │ (df_site, color_by, sel_id)  │
├───────────────────────────────┼───────────────────────────────────┼──────────────────────────────┤
│ components/confusion_matrix.py│                                   │                              │
│  - render_confusion_matrix_...│ 5_Model_Comparison.py             │ (cm, class_names, title)     │
├───────────────────────────────┼───────────────────────────────────┼──────────────────────────────┤
│ components/filters.py         │                                   │                              │
│  - render_pagination_controls │ 2_Training_Data_Explorer.py       │ (total, per_page, cur) -> int│
│  - render_prev_next_controls  │ (Available reusable utility)      │ (cur, total, label) -> int   │
├───────────────────────────────┼───────────────────────────────────┼──────────────────────────────┤
│ components/image_cards.py     │                                   │                              │
│  - render_mars_patch_card     │ (Reusable card builder)           │ (img, meta, sel) -> bool     │
│  - render_moon_image_card     │ (Reusable card builder)           │ (img, id, class, sel) -> bool│
├───────────────────────────────┼───────────────────────────────────┼──────────────────────────────┤
│ components/metric_tables.py   │                                   │                              │
│  - render_parameter_count_... │ 5_Model_Comparison.py             │ () -> None (renders table)   │
│  - render_inference_timing_...│ 5_Model_Comparison.py             │ (moon_results) -> None       │
│  - render_mars_training_tim...│ 5_Model_Comparison.py             │ (fold, seed) -> DataFrame    │
│  - render_moon_training_fit...│ 5_Model_Comparison.py             │ (moon_results) -> DataFrame  │
│  - render_mars_paired_diff_...│ (Preserved in module; NOT on p5)  │ (paired_df) -> None          │
├───────────────────────────────┼───────────────────────────────────┼──────────────────────────────┤
│ components/navigation.py      │                                   │                              │
│  - render_walkthrough_footer  │ 1_Overview.py, 2_Training_Data... │ (curr_num, next_path, ...)   │
│                               │ 3_Mars_Test_Explorer.py,          │                              │
│                               │ 4_Moon_Test_Explorer.py,          │                              │
│                               │ 5_Model_Comparison.py,            │                              │
│                               │ 6_Methodology.py                  │                              │
└───────────────────────────────┴───────────────────────────────────┴──────────────────────────────┘
```

---

## 6. Functionality & Scientific-Integrity Preservation Checklist

> [!CAUTION]
> **CRITICAL SCIENTIFIC INTEGRITY RESTRICTIONS FOR GEMINI ARGON**  
> Under no circumstances may any redesign modify, weaken, or omit the audited scientific disclosures, dataset facts, or numerical results. Gemini Argon's redesign plan must adhere strictly to the following requirements:

- [x] **DO NOT RESTORE the Mars Statistical Assessment Section in Model Comparison:**
  - The paired-site statistical assessment UI table (`render_mars_paired_differences_table` displaying nominal 95% t-intervals with *df*=4 crossing zero) was intentionally removed from Tab 1 of `pages/5_Model_Comparison.py` per explicit user instructions.
  - **Rule:** The redesign plan must **NEVER restore** this section to the Mars Comparative Evaluation tab. `test_mars_comparative_evaluation_statistical_assessment_removed` specifically enforces that `render_mars_paired_differences_table` and `3. Paired Site Differences` do not appear in `pages/5_Model_Comparison.py`.
- [x] **Preserve 15 Repaired Mars Runs (5 Folds × 3 Seeds):**
  - All Mars metrics and predictions must come exclusively from `runs/pipeline_repair_audit/retraining_seedset`.
  - Geographic test sites per fold: Fold 0 (`site_01` Jezero), Fold 1 (`site_02` Gale), Fold 2 (`site_03` Mawrth), Fold 3 (`site_04` Oxia), Fold 4 (`site_05` Eberswalde).
- [x] **Preserve Derived Landing-Safety Proxy Wording:**
  - The rule `mean slope < 8.0°` AND `hazard fraction < 0.05` is a simplified terrain feasibility proxy.
  - **Rule:** Never label or describe this rule as "certified landing safety", "operational landing guarantee", or "planetary habitability". All disclosures must emphasize that real Entry, Descent, and Landing (EDL) operations require multi-sensor navigation, hazard detection, and engineering margins.
- [x] **Preserve Parameter Count Corrections (59,347 Trainable Params):**
  - In the Moon crater experiment, the classical CNN trained **59,347 trainable parameters** end-to-end. The legacy JSON log value of 195 represents only the classifier head post-encoder freeze.
  - In Mars, ResNet-18 trained 11.24M parameters end-to-end, while QCNN trained 4,152 head parameters on a frozen representation.
  - **Rule:** The 59,347 parameter audit correction must be prominently displayed wherever model capacities are discussed.
- [x] **Preserve Latency Non-Comparability Caveats:**
  - Recorded Moon inference times (e.g. ~4.6 ms for CNN vs ~2.8 ms for QCNN) cannot be directly compared because CNN includes full 227×227 RGB feature extraction, whereas QCNN and MLP were measured only on pre-cached 8D embeddings.
  - **Rule:** The prominent warning callout stating "Not an apples-to-apples latency comparison; no quantum speedup claimed" must be preserved.
- [x] **Preserve Moon Test Embedding Disclosure:**
  - Test embeddings were computed through the frozen backbone prior to head fitting, but test labels were never used for loss calculation or gradient updates. This must remain explicitly disclosed in Methodology and Model Comparison.
- [x] **Preserve Sample Standard Deviation (SD) Semantics:**
  - All uncertainty bounds across the 5 Mars sites or 3 Moon seeds are empirical **Sample Standard Deviations (SD)**, not asymptotic confidence intervals.
- [x] **Preserve Read-Only Mode & Pre-computed Predictions:**
  - No models may be retrained; no inference pipelines (PyTorch, PennyLane) may be executed in real-time. All data must load from precomputed parquets, JSON logs, and shards.
- [x] **Preserve Demo Presets & Deterministic Filtering:**
  - Mars demo presets (`Model Disagreement`, `CNN False Positive`, `QCNN False Negative`, `Both Models Agree Safe`) must remain functional and deterministically filtered.
  - Moon demo presets (`All 3 Models Agree & Correct`, `Hybrid Beats CNN`, `CNN Beats Hybrid`, `All 3 Models Wrong`) must remain functional.

---

## 7. Relevant Files & Functions for Redesign Opportunities

Below is the concrete mapping of UI improvement opportunities to their exact source locations, identifying what Gemini Argon should target:

| Target Area / Opportunity | Relevant File Path | Functions / Line Ranges | Recommended UI/UX Enhancement for Gemini Argon |
| :--- | :--- | :--- | :--- |
| **Streamlit Deprecation Migration** | All pages & components | Throughout `visualization_app` | Replace all occurrences of `use_container_width=True` with `width="stretch"` (and `use_container_width=False` with `width="content"` or omit) to eliminate Streamlit 1.66.0 deprecation warnings. |
| **Global Design System Consolidation** | `style/theme.py` | Lines 24–131 (`CUSTOM_CSS`, `badge_html`) | Consolidate scattered inline styles (`<div style="background: #181C24; border: ...">`) into reusable CSS utility classes (`.app-card`, `.stat-pill`, `.table-container`, `.hero-banner`). Introduce CSS variables for tokens (`--bg-surface`, `--color-cnn`). |
| **Landing Page Hero & Flow Callout** | `app.py` | Lines 50–93 | Transform the plain HTML demonstration guide into an interactive, visually stunning hero component with direct click-through action cards to "Begin Coursework Demonstration (Step 1)". |
| **Overview Page Visual Polish** | `pages/1_Overview.py` | Lines 64–98, 163–253 | Upgrade the dual Mars/Moon executive summary cards with styled metric badges and subtle glassmorphic styling. Replace raw dataframes with clean styled comparison cards featuring colored difference pills. |
| **Training Explorer Grid Ergonomics** | `pages/2_Training_Data_Explorer.py` | Lines 103–134, 220–248 | Replace the heavy button-per-tile grid with selectable clickable cards (or a streamlined compact list) to reduce visual clutter and prevent vertical button crowding. |
| **Mars Test Explorer Layout Streamlining** | `pages/3_Mars_Test_Explorer.py` | Lines 144–195, 292–413, 417–440 | Re-organize the 2,500px tall vertical stack into a cleaner 2-row layout. Promote the 2D intra-site mosaic grid scatter plot out of the hidden expander into a prominent split-screen tab alongside patch inspection. |
| **Moon Test Explorer Card Layout** | `pages/4_Moon_Test_Explorer.py` | Lines 307–349, 377–412 | Enhance the 3-model confidence cards with mini horizontal confidence bars or radial meters instead of plain percentage text. Redesign the raw logit table with side-by-side comparative bars. |
| **Model Comparison Confusion Matrix Layout** | `pages/5_Model_Comparison.py` | Lines 136–152 | In Tab 2, instead of squeezing 3 heatmaps into a narrow 3-column row, provide a model selector toggle (or 2-column layout) so heatmaps render with generous cell size, clear readable fonts, and un-cramped axis labels. |
| **Methodology Page Readability** | `pages/6_Methodology.py` | Lines 34–143 | Convert the overwhelming stack of 5 fully-expanded accordions into an interactive tabbed or categorized documentation layout with summary callout cards, iconography, and structured specification tables. |
| **Navigation Footer Ergonomics** | `components/navigation.py` | Lines 7–124 (`render_walkthrough_footer`) | Modernize the walkthrough footer with polished step progress pills (e.g., `Step 3 of 6: Mars Test Explorer`) and sleek gradient previous/next buttons. |

---

## 8. Verified Baseline Test Results

Before preparing this handoff, the full automated acceptance test suite was executed against the local environment (`Python 3.11.15`, `pytest 9.1.1`). 

**Execution Command:**
```bash
.\.venv\Scripts\python.exe -m pytest tests/ -v
```

**Results:**
```text
============================= test session starts =============================
platform win32 -- Python 3.11.15, pytest-9.1.1, pluggy-1.7.0
rootdir: D:\Mars-QCNN, configfile: pyproject.toml
collected 42 items

tests\test_m1_data_loading.py::test_mars_metadata_loading PASSED         [  2%]
tests\test_m1_data_loading.py::test_mars_predictions_loading PASSED      [  4%]
tests\test_m1_data_loading.py::test_mars_patch_image_loading PASSED      [  7%]
tests\test_m1_data_loading.py::test_mars_aggregate_metrics PASSED        [  9%]
tests\test_m1_data_loading.py::test_moon_results_loading PASSED          [ 11%]
tests\test_m1_data_loading.py::test_moon_image_loading PASSED            [ 14%]
tests\test_m1_data_loading.py::test_moon_predictions_df_and_softmax PASSED [ 16%]
tests\test_m1_data_loading.py::test_moon_aggregate_metrics PASSED        [ 19%]
tests\test_m1_data_loading.py::test_moon_cross_seed_sample_id_alignment PASSED [ 21%]
tests\test_m1_data_loading.py::test_mars_metadata_and_signed_errors_alignment PASSED [ 23%]
tests\test_m2_overview.py::test_mars_headline_metrics_match PASSED       [ 26%]
tests\test_m2_overview.py::test_moon_headline_metrics_match PASSED       [ 28%]
tests\test_m2_overview.py::test_overview_walkthrough_and_page_execution PASSED [ 30%]
tests\test_m3_training_explorer.py::test_mars_site_selection_and_stretching PASSED [ 33%]
tests\test_m3_training_explorer.py::test_mars_fold_mapping PASSED        [ 35%]
tests\test_m3_training_explorer.py::test_moon_provenance_and_images PASSED [ 38%]
tests\test_m3_training_explorer.py::test_training_explorer_page_execution PASSED [ 40%]
tests\test_m4_mars_test_explorer.py::test_mars_fold0_seed42_predictions PASSED [ 42%]
tests\test_m4_mars_test_explorer.py::test_mars_disagreement_filtering PASSED [ 45%]
tests\test_m4_mars_test_explorer.py::test_mars_both_wrong_filtering PASSED [ 47%]
tests\test_m4_mars_test_explorer.py::test_mars_threshold_consistency PASSED [ 50%]
tests\test_m4_mars_test_explorer.py::test_mars_error_sorting PASSED      [ 52%]
tests\test_m4_mars_test_explorer.py::test_mars_demo_presets PASSED       [ 54%]
tests\test_m4_mars_test_explorer.py::test_mars_test_explorer_page_execution PASSED [ 57%]
tests\test_m4_mars_test_explorer.py::test_mars_grid_scatter_builder PASSED [ 59%]
tests\test_m5_moon_test_explorer.py::test_moon_seed42_count PASSED       [ 61%]
tests\test_m5_moon_test_explorer.py::test_moon_softmax_sum_and_argmax PASSED [ 64%]
tests\test_m5_moon_test_explorer.py::test_moon_correctness_and_disagreements PASSED [ 66%]
tests\test_m5_moon_test_explorer.py::test_moon_ejecta_class_count PASSED [ 69%]
tests\test_m5_moon_test_explorer.py::test_moon_actual_jpeg_loads PASSED  [ 71%]
tests\test_m5_moon_test_explorer.py::test_moon_demo_presets PASSED       [ 73%]
tests\test_m5_moon_test_explorer.py::test_moon_confusion_pair_filtering PASSED [ 76%]
tests\test_m5_moon_test_explorer.py::test_moon_test_explorer_page_execution PASSED [ 78%]
tests\test_m6_model_comparison.py::test_mars_paired_differences_cross_zero PASSED [ 80%]
tests\test_m6_model_comparison.py::test_moon_confusion_matrices_shape_and_sums PASSED [ 83%]
tests\test_m6_model_comparison.py::test_chart_builders_return_valid_figures PASSED [ 85%]
tests\test_m6_model_comparison.py::test_mars_chart_layout_and_units PASSED [ 88%]
tests\test_m6_model_comparison.py::test_capacity_and_timing_tables PASSED [ 90%]
tests\test_m6_model_comparison.py::test_model_comparison_page_execution PASSED [ 92%]
tests\test_m6_model_comparison.py::test_mars_comparative_evaluation_statistical_assessment_removed PASSED [ 95%]
tests\test_m7_methodology.py::test_methodology_file_content PASSED       [ 97%]
tests\test_m7_methodology.py::test_methodology_page_execution PASSED     [100%]

============================= 42 passed in 1.90s ==============================
```

---

## 9. Baseline Screenshot Inventory

All baseline screenshots were captured directly from the live Streamlit instance running locally on `http://localhost:8503` using headless browser automation. The images are stored in `D:\Mars-QCNN\audit_screenshots/`:

| Screenshot File Name | Resolution | Description & Visible UI Components |
| :--- | :--- | :--- |
| `00_landing_1920x1080.png` | 1920×1080 | Welcome page, 6-step walkthrough callout box, 3-metric system readiness strip, sidebar metadata. |
| `00_landing_1366x768.png` | 1366×768 | Laptop viewport rendering of the landing screen, showing tighter padding on readiness metrics. |
| `01_overview_1920x1080.png` | 1920×1080 | Task overview, research integrity notice, 2-column summary cards, and walkthrough expander. |
| `01_overview_1366x768.png` | 1366×768 | Laptop viewport of overview showing vertical stacking of summary cards and header elements. |
| `02_training_data_explorer_1920x1080.png` | 1920×1080 | Mars HiRISE tab: 3×4 thumbnail grid, inspect buttons, right-hand detail inspection pane with criteria check table. |
| `02_training_data_explorer_1366x768.png` | 1366×768 | Laptop viewport showing grid button layout and right-hand metadata pane. |
| `03_mars_test_explorer_top_1920x1080.png` | 1920×1080 | Top section: 5-metric strip, Quick-Select Demo Preset radio bar, patch dropdown selector, and CSV download. |
| `03_mars_test_explorer_scrolled_1920x1080.png` | 1920×1080 | Scrolled view: 64×64 HiRISE patch, Classical ResNet-18 vs QCNN prediction cards, and absolute error bar chart. |
| `03_mars_test_explorer_1366x768.png` | 1366×768 | Laptop viewport showing compact stepping controls, patch header, and prediction error cards. |
| `04_moon_test_explorer_1920x1080.png` | 1920×1080 | Moon test explorer: 5-metric strip, demo presets bar, 227×227 crater image, 3-model confidence cards, and softmax bar chart. |
| `04_moon_test_explorer_1366x768.png` | 1366×768 | Laptop viewport showing 3-column model confidence cards and probability distribution bar chart. |
| `05_model_comparison_tab1_mars_1920x1080.png` | 1920×1080 | Tab 1: Fixed vs Validation-Calibrated operating regime toggle, Continuous RMSE bar chart, and Landing Safety proxy bar chart. |
| `05_model_comparison_tab1_mars_1366x768.png` | 1366×768 | Laptop viewport of Mars comparative evaluation charts. |
| `05_model_comparison_tab2_moon_1920x1080.png` | 1920×1080 | Tab 2: Recomputed metrics grouped bar chart (Accuracy, Balanced Acc, Macro-F1) and seed selector for confusion matrices. |
| `05_model_comparison_tab3_training_fit_1920x1080.png` | 1920×1080 | Tab 3: Sub-tabs showing side-by-side epoch loss progression curves for ResNet-18 vs Hybrid QCNN. |
| `05_model_comparison_tab4_capacity_timing_1920x1080.png` | 1920×1080 | Tab 4: Parameter verification table (11.2M vs 4,152; 59.3k vs 568 vs 547), latency caveat callout, and timing breakdown tables. |
| `05_model_comparison_tab4_capacity_timing_1366x768.png` | 1366×768 | Laptop viewport of capacity and timing analysis tables. |
| `06_methodology_1920x1080.png` | 1920×1080 | Methodology page showing executive disclosure callout and top expanded documentation sections. |
| `06_methodology_1366x768.png` | 1366×768 | Laptop viewport of methodology documentation expanders. |

---

## 10. Outstanding Uncertainties & Audit Boundaries

1. **Unverified Moon Dataset Metadata:**
   - As documented in `FINAL_READINESS_AUDIT.md`, the Moon LROCNet dataset release (`data/raw/moon/lrocnet/provenance_manifest.csv`) has confirmed file hashes, pixel dimensions (227×227 RGB), and split allocations, but lacks verified source observation IDs, sensor geometry, and planetary coordinates. Gemini Argon must maintain the caveat status `FILE_FACTS_CONFIRMED_SOURCE_ID_UNVERIFIED`.
2. **Real-Time Quantum Execution:**
   - PennyLane state-vector quantum simulations were executed during the offline experimental phase. No live quantum circuit simulation or PyTorch backpropagation runs inside Streamlit.
3. **Mobile & Sub-1200px Viewports:**
   - The dashboard is specifically optimized as a coursework presentation and assessment tool for desktop and laptop displays (1366×768 to 1920×1080). Responsive mobile layouts (<768px) were not benchmarked and are considered out of scope.

---

*End of UI/UX Audit Handoff Document. Prepared for Gemini Argon.*
