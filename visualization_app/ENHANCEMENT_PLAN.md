# Enhancement Plan: Mars + Moon CNN/QCNN Research Explorer (`ENHANCEMENT_PLAN.md`)

**Target File Path:** `D:\Mars-QCNN\visualization_app\ENHANCEMENT_PLAN.md`

**Target Application Root:** `D:\Mars-QCNN\visualization_app`

**Primary Reference:** `PROJECT_HANDOFF.md` (Version 1.0) and `D:\Mars-QCNN\FINAL_READINESS_AUDIT.md`

**Scope:** Undergraduate coursework demonstration readiness—prioritizing visual clarity, rapid sample-level inspection, and scientific trustworthiness without architectural overengineering.

---

## 1. Executive Summary

The existing Streamlit application (`D:\Mars-QCNN\visualization_app`) is already a mature, well-tested 6-page dashboard backed by 27 passing unit and acceptance tests (`pytest tests/ -v` running in 1.25s). It already implements core data loaders, memory-mapped HiRISE patch rendering, LROCNet JPEG loading, paired Mars regression/proxy inspection, 3-model Moon classification inspection, aggregate Plotly comparisons, and methodology disclosures.

Because the foundational architecture and core views already exist, **the highest-value coursework enhancement strategy is targeted refinement and reuse—not a rewrite**:

1. **P0 — Targeted Data-Integrity & Scientific Audit Verification:** Confirm code-level enforcement of sample alignment, class ordering (`0: ejecta`, `1: oldcrater`, `2: none`), bivariate proxy boundaries ($\text{mean\_slope\_5m} < 8.0^\circ \land \text{hazard\_fraction} < 0.05$), primary artifact paths, and mandatory scientific caveats, adding lightweight regression tests only where coverage gaps exist.


2. **P1 — Frictionless Sample Selection in Mars & Moon Test Explorers:** Currently, `3_Mars_Test_Explorer.py` and `4_Moon_Test_Explorer.py` rely on large dropdown selectors (`st.selectbox` with up to 2,195 Mars tiles or 779 Moon images), which are clumsy during a live oral presentation. Adding **Quick-Select Demo Presets** (one-click buttons or curated selector pills for representative Agreement, Disagreement, False Positive, and False Negative samples), a **Confusion-Cell Filter** (`True Class → Predicted Class`) on the Moon page, and prev/next sample stepping will make live examination seamless.


3. **P1 — Guided Coursework Presentation Flow & Visual Polish:** Add a lightweight **Coursework Walkthrough Guide** on `app.py` / `1_Overview.py` and concise **"Next in Demo"** navigation footers across pages `1` through `6` so an examiner can follow the logical arc (*Dataset Examples → Model Predictions → Ground Truth → Model Comparison → Limitations*) without rebuilding Streamlit's native multipage router. Audit and polish chart units, error-bar labels (`± 1 Sample SD`), empty-filter states, and mandatory metric disclosures across `5_Model_Comparison.py`.


4. **P2 — Optional High-Value Extras (Strictly Time-Permitting):** An unprojected **Intra-Site Grid Scatter Plot** (`row` vs. `col` from `metadata.parquet`) in `3_Mars_Test_Explorer.py` and CSV export buttons for filtered test tables.



---

## 2. Current-State Assumptions

### 2.1 Verified Facts Documented in `PROJECT_HANDOFF.md`

* **Runtime & Environment:** The app runs in an isolated Python 3.11 virtual environment (`D:\Mars-QCNN\visualization_app\.venv`) using Streamlit (`>= 1.37.0`), Plotly (`>= 5.22.0`), Pandas (`>= 2.1.0`), NumPy (`>= 1.26.0`), PyArrow (`>= 14.0.0`), Pillow (`>= 10.0.0`), Scikit-learn (`>= 1.3.0`), PyYAML (`>= 6.0.1`), and Pytest (`>= 8.0.0`). PyTorch and PennyLane are intentionally excluded at runtime for sub-second loading.


* **Existing Application Structure:**
* Entry point: `app.py` (wide layout, CSS injection, `verify_critical_paths()` readiness dashboard).


* Data loaders: `data_loading/paths.py`, `data_loading/mars_data.py`, `data_loading/moon_data.py`.


* Styling & components: `style/theme.py`, `components/image_cards.py`, `components/filters.py`, `components/charts.py`, `components/confusion_matrix.py`, `components/metric_tables.py`.


* Multipage views: `pages/1_Overview.py` through `pages/6_Methodology.py`.


* Test suite: `tests/test_m1_data_loading.py` through `tests/test_m7_methodology.py` (27 passing tests).




* **Mars Artifacts & Schema:**
* Processed metadata: `D:\Mars-QCNN\runs\controlled_evaluation\processed\metadata.parquet` (15,233 rows × 31 columns, including `tile_id`, `image_id`, `site_id`, `row`, `col`, `valid_coverage`, `dtm_valid_coverage`, `image_valid_coverage`, `mean_slope_5m`, `roughness_m`, `hazard_fraction`, `safe_label`, `shard_path`, `shard_idx`, `fold`) and `dataset_metadata.json`.


* Memory-mapped image shards: `shards/site_{01-05}_tiles.npy` with shape `(N, 3, 64, 64)` in `float16` (Channel 0: calibrated I/F, Channel 1: DTM validity mask, Channel 2: image validity mask).


* Primary evaluation source: `D:\Mars-QCNN\runs\pipeline_repair_audit\retraining_seedset` across 5 leave-one-site-out folds (0–4) and 3 seeds (42, 43, 44) = 15 runs per model (`model1_resnet18` with 11,242,563 trainable params; `model2_qcnn` with 4,152 trainable head params).


* Paired predictions: `test_predictions.parquet` (14 columns: `tile_id`, `image_id`, `fold`, `seed`, `model_name`, `variant`, `pred_mean_slope_5m`, `true_mean_slope_5m`, `pred_roughness_m`, `true_roughness_m`, `pred_hazard_fraction`, `true_hazard_fraction`, `pred_safe`, `true_safe`) verified to have identical ordered `tile_id` records across all 15 fold/seed pairs (45,699 total paired rows).




* **Moon Artifacts & Schema:**
* Raw data: `D:\Mars-QCNN\data\raw\moon\lrocnet` containing 5,000 RGB JPEGs (227×227) split into `train` (3,608), `val` (613), and `test` (779: `ejecta` = 89, `oldcrater` = 157, `none` = 533), plus `provenance_manifest.csv` (where `source_observation_id`, `source_image_id`, and `site_id` are confirmed `NaN`/empty, and `provenance_status` is `FILE_FACTS_CONFIRMED_SOURCE_ID_UNVERIFIED`).


* Primary evaluation source: `D:\Mars-QCNN\data\raw\moon\lrocnet\experiment_lunar_cnn_qcnn` with per-seed logs at `runs/logs/seed_{42,43,44}/results.json`, `history.csv`, and verified metrics at `runs/results_verification/recomputed_metrics.csv`.


* Models & parameters: `cnn` (59,347 trainable parameters during training; legacy JSON records 195 post-freezing), `hybrid_qcnn` (568 trainable head parameters), `frozen_embedding_mlp` (547 trainable head parameters), and `majority_train_baseline`.





### 2.2 Reasonable Recommendations Based on Verified Facts

* **Reuse Existing Inspector Views:** `3_Mars_Test_Explorer.py` and `4_Moon_Test_Explorer.py` already render images, ground truth, multi-model predictions, errors/probabilities, and threshold/logit breakdowns. Enhancements should focus on **sample discovery ergonomics** (curated presets, prev/next buttons, confusion-pair filtering, metadata header clarity) rather than rewriting the inspection cards.


* **Lightweight Demo Walkthrough:** Use existing pages in numerical order (`1` → `6`) and add a compact **"Presentation Walkthrough Guide"** expander on `1_Overview.py` (and/or sidebar indicator) with step-by-step talking points and recommended filters for live demonstration.
* **Explicit Visual Polish:** Standardize chart axis titles with physical units (`°`, `m`, fraction, `%`), ensure error bars explicitly state `± 1 Sample SD`, and verify clean empty states when strict filters return zero samples.

### 2.3 Questions & Uncertainties Flash Must Verify Before Editing

Before modifying any code, Gemini Flash must inspect the local files in `D:\Mars-QCNN\visualization_app` to resolve the following questions:

1. **Mars Metadata Join in Test Explorer:** Does `data_loading/mars_data.py` merge `metadata.parquet` fields (`site_id`, `row`, `col`, `valid_coverage`, `dtm_valid_coverage`, `image_valid_coverage`) onto `test_predictions.parquet` by `tile_id` and `image_id`, and does `3_Mars_Test_Explorer.py` display `site_id`, `image_id`, and `tile_id` prominently alongside `row` and `col`?


2. **Mars Proxy Inequality & Floating-Point Boundaries:** Does the UI recompute `pred_safe` / `true_safe` using strict inequality (`< 8.0` and `< 0.05`) or read the saved boolean columns from `test_predictions.parquet`, and do both match 100% across all 15 runs?


3. **Moon Sample ID & Path Resolution:** How does `data_loading/moon_data.py` resolve `test.sample_ids` (e.g., `test/ejecta/M1122721870LC_pyr-1052.jpg`) against `D:\Mars-QCNN\data\raw\moon\lrocnet`, and are `cnn`, `hybrid_qcnn`, and `frozen_embedding_mlp` predictions/logits indexed strictly by the same `sample_ids` array order across seeds 42, 43, and 44?


4. **Moon Methodological Disclosures:** Where in `1_Overview.py`, `5_Model_Comparison.py`, and `6_Methodology.py` is the disclosure regarding **pre-computed test embeddings** (computed before fitting QCNN/MLP heads, though test labels were never used for training, model selection, or threshold selection) currently rendered? Is it clearly visible on both `5_Model_Comparison.py` and `6_Methodology.py`?


5. **Existing Pagination Component Reuse:** How is `render_pagination_controls` in `components/filters.py` implemented, and can it (or a companion helper) be reused to step through filtered test samples (`Prev Sample` / `Next Sample`) in `3_Mars_Test_Explorer.py` and `4_Moon_Test_Explorer.py`?



---

## 3. Prioritized Enhancements (P0 / P1 / P2)

| Priority | Enhancement Area | Existing Functionality to Reuse | What to Add / Refine | Implementation Value | Risks & Mitigations | Acceptance Criteria |
| --- | --- | --- | --- | --- | --- | --- |
| **P0** | **Correctness, Data Integrity & Constraint Verification** | `data_loading/paths.py`, `mars_data.py`, `moon_data.py`, and existing `tests/test_m1_*.py` to `test_m7_*.py` (27 tests).

 | Verify code paths for sample-to-image alignment, target/class order, proxy formula ($< 8.0^\circ \land < 0.05$), primary artifact paths, parameter counts (`59,347` vs `195`), and test-embedding disclosure. Add targeted unit tests for any unverified invariant.

 | Guarantees 100% scientific defensibility under examiner questioning. | *Risk:* Redundant auditing slows work. *Mitigation:* Inspect loaders and existing tests first; only add tests for missing checks. | All 27 existing tests + new integrity tests pass; zero historical/uncalibrated Mars paths referenced.

 |
| **P1** | **Mars Terrain Patch Inspector Ergonomics (`3_Mars_Test_Explorer.py`)** | Existing fold/seed selectors, 8 outcome filters, 6 sort modes, 256px patch viewer, paired ResNet-18 vs QCNN cards, signed errors ($\Delta$), absolute error bar chart, and bivariate threshold table.

 | 1. Add **Quick-Select Demo Presets** (buttons/pills that auto-select a deterministic example for: *Both Agree Safe*, *Models Disagree (`CNN != QCNN`)*, *CNN False Positive*, *QCNN False Negative*) in the active fold/seed.<br>

<br>2. Add **Prev / Next Patch** stepping buttons next to the `tile_id` selector.<br>

<br>3. Ensure header clearly displays `site_id`, `image_id`, `tile_id`, `fold`, `seed`, and local grid (`row`, `col`), plus explicit caveat that the proxy is not verified landing safety or habitability.

 | Eliminates scrolling through 2,195 dropdown items during a live demo.

 | *Risk:* Hardcoded `tile_id` fails when fold changes. *Mitigation:* Dynamically select the top-ranked matching sample from the active fold/seed dataframe for each preset category. | Lecturer/student can click any preset or Prev/Next button to inspect a verified patch with full metadata, errors, and threshold breakdown in $<1\text{s}$. |
| **P1** | **Moon Test-Image Inspector Ergonomics (`4_Moon_Test_Explorer.py`)** | Existing seed selector, class/outcome filters, confidence sorters, 227×227 JPEG viewer, 3 model cards (CNN, Hybrid QCNN, Frozen MLP), softmax chart, and raw logits table.

 | 1. Add **Quick-Select Demo Presets** (*All 3 Agree Correct*, *Hybrid QCNN Beats CNN*, *CNN Beats Hybrid QCNN*, *High-Confidence Error*).<br>

<br>2. Add a **Confusion-Cell Filter** (`True Class → Predicted Class` for a selected model, e.g., `True: oldcrater → Pred: none`).

<br>

<br>3. Add **Prev / Next Image** stepping buttons alongside the `sample_id` selector. | Replaces tedious 779-item dropdown scrolling with instant diagnostic filtering and error-mode inspection.

 | *Risk:* Filter combination returns 0 images and breaks UI. *Mitigation:* Render a clean `st.info` empty-state banner with a "Reset Filters" hint when filtered count is 0. | User can jump directly to agreement/disagreement cases or filter by specific confusion cell without index errors. |
| **P1** | **Guided Demonstration Flow (`app.py`, `1_Overview.py` – `6_Methodology.py`)** | Existing 6-page structure (`app.py` + `pages/1_Overview.py` to `6_Methodology.py`) and Mermaid diagrams.

 | 1. Add a collapsible **"Coursework Demo Walkthrough (6 Steps)"** card on `app.py` and `1_Overview.py` mapping the required narrative (*Dataset Examples → Model Predictions → Ground Truth → Model Comparison → Limitations*) to exact pages.<br>

<br>2. Add a subtle footer or sidebar guide on each page indicating current walkthrough stage and next page link (`st.page_link` if supported in Streamlit `>=1.37.0` or clear navigation hint).

 | Gives the presenter a built-in script and keeps the examiner oriented without altering the app architecture. | *Risk:* Cluttering the UI with heavy tutorial popups. *Mitigation:* Keep it to a single clean expander on `1_Overview.py` and a compact 1-line navigation cue per page. | Examiner can follow the 6-step narrative end-to-end with clear transitions between data, sample predictions, aggregate metrics, and caveats. |
| **P1** | **Aggregate Results & Visual Polish (`5_Model_Comparison.py` & `components/`)** | Existing 4-tab layout (Mars evaluation, Moon benchmark, Epoch curves, Capacity & Timing), Plotly builders, and paired difference tables.

 | 1. Verify/polish explicit units on all chart axes (`°`, `m`, hazard fraction, `%`).<br>

<br>2. Ensure error bar legends state `± 1 Sample SD (n=5 sites)` for Mars and `± 1 Sample SD (n=3 seeds)` for Moon.

<br>

<br>3. Verify prominent placement of all mandatory disclosures: Mars $df=4$ zero-crossing 95% t-intervals and validation calibration domain shift (39.65% CNN / 33.94% QCNN test FPR); Moon 59,347 vs 195 CNN params, cached-embedding latency caveat (~2.8 ms vs ~4.6 ms), and pre-computed test embeddings disclosure.

 | Prevents misinterpretation of statistical noise, parameter counts, or inference speed during examination. | *Risk:* Text walls overwhelm charts. *Mitigation:* Use concise, structured callout boxes and table footnotes in `components/metric_tables.py`.

 | All charts have unambiguous labels/units; all mandatory scientific constraints are visible on relevant tabs; layout works cleanly at 1920×1080 and 1366×768.

 |
| **P2** | **Optional Extras: Intra-Site Grid Map & CSV Export** | `row` and `col` columns in `metadata.parquet`; filtered dataframes in `3_Mars_Test_Explorer.py` and `4_Moon_Test_Explorer.py`.

 | 1. Optional collapsible **Intra-Site 2D Grid Scatter Plot** (`col` vs `row`) in `3_Mars_Test_Explorer.py` colored by Ground Truth Safe, Model Agreement, or Absolute Slope Error, explicitly labeled as *Local Unprojected Tile Grid Coordinates (Not Planetary Lat/Lon)*.

<br>

<br>2. Optional `st.download_button` for exporting the currently filtered test table as CSV.

 | Shows spatial clustering of terrain errors within a landing site mosaic.

 | *Risk:* Misleading examiner into thinking it is a global planetary map. *Mitigation:* Explicit axis labels and disclaimer; strictly optional after P0/P1 pass. | Scatter plot renders accurately from `row`/`col` without fabricating geographic coordinates; CSV download matches filtered dataframe.

 |

---

## 4. Data-Integrity Requirements

Gemini Flash must verify and enforce the following non-negotiable data-integrity rules across `data_loading/` and `pages/`:

1. **Sample-to-Image & Ground-Truth Alignment:**
* **Mars:** Every displayed patch in `2_Training_Data_Explorer.py` and `3_Mars_Test_Explorer.py` must be fetched from `shard_path` at `shard_idx` matching the exact `tile_id` in `metadata.parquet`. In `3_Mars_Test_Explorer.py`, paired ResNet-18 (`model1_resnet18`) and QCNN (`model2_qcnn`) rows for a given `(fold, seed)` must assert exact `tile_id` and `image_id` equality, and `true_mean_slope_5m`, `true_roughness_m`, `true_hazard_fraction`, and `true_safe` must match between both models and `metadata.parquet`.


* **Moon:** In `4_Moon_Test_Explorer.py`, for each seed (`42`, `43`, `44`), the image loaded from `D:\Mars-QCNN\data\raw\moon\lrocnet\<sample_id>` must correspond to the exact index $i \in [0, 778]$ in `results.json["test"]["sample_ids"]`. Furthermore, Flash must verify that all three models (`cnn`, `hybrid_qcnn`, `frozen_embedding_mlp`) are evaluated on the exact same 779 held-out `test.sample_ids` and `test.ground_truth` array.




2. **Target Ordering, Class Ordering & Mathematical Definitions:**
* **Mars Targets & Units:**
* `mean_slope_5m`: degrees (`°`), threshold $< 8.0^\circ$.


* `roughness_m`: meters (`m`), no threshold in bivariate rule (continuous regression target only).


* `hazard_fraction`: unitless fraction $\in [0, 1]$, threshold $< 0.05$.


* **Low-Risk Terrain Proxy:** $\text{Safe} \iff (\text{mean\_slope\_5m} < 8.0^\circ) \land (\text{hazard\_fraction} < 0.05)$. Both signed error ($\Delta = \text{pred} - \text{true}$) and absolute error ($\vert{}\text{pred} - \text{true}\vert{}$) must be clearly labeled.




* **Moon Classes & Probabilities:**
* Integer-to-label mapping must strictly enforce `0: "ejecta"`, `1: "oldcrater"`, `2: "none"`.


* Softmax probabilities must be computed deterministically from `test.logits` using numerically stable log-sum-exp / shifted exponential so $\sum_{c=0}^{2} p_c = 1.0$, and $\arg\max_c(p_c)$ must match the saved integer prediction in `test.predictions`. Never fabricate confidence scores where logits do not exist.






3. **Authoritative Experiment Sources:**
* **Mars Primary Source:** Strictly load from `D:\Mars-QCNN\runs\pipeline_repair_audit\retraining_seedset` (5 folds × 3 seeds = 15 runs per model). Processed tile metadata and `.npy` shards are read from `D:\Mars-QCNN\runs\controlled_evaluation\processed`. Never display historical single-seed predictions from `runs\controlled_evaluation` or uncalibrated `runs\real_v1` as primary metrics.


* **Moon Primary Source:** Strictly load from `D:\Mars-QCNN\data\raw\moon\lrocnet\experiment_lunar_cnn_qcnn` (`runs/logs/seed_{42,43,44}/results.json`, `history.csv`, and `runs/results_verification/recomputed_metrics.csv`).




4. **Defensive Error Handling & Empty States:**
* If a user selects a combination of filters (e.g., class filter + outcome filter in `3_Mars_Test_Explorer.py` or `4_Moon_Test_Explorer.py`) that produces `0` matching samples, the page must display a polite informational banner (`st.info("No samples match the current filter combination. Try broadening your selection.")`) and stop gracefully rather than throwing an `IndexError` or `KeyError`.
* Missing lunar spatial coordinates (`source_observation_id`, `source_image_id`, `site_id` are `NaN` in `provenance_manifest.csv`) must continue to be explicitly disclosed as `FILE_FACTS_CONFIRMED_SOURCE_ID_UNVERIFIED`—never replaced with synthetic coordinates.





---

## 5. Recommended UI Flow for Coursework Demonstration

Instead of rebuilding navigation, preserve the existing 6-page structure and guide the lecturer/examiner through a clean 6-step narrative arc:

| Demo Step | Narrative Goal | Target Page | Key Visuals & Interactive Actions to Highlight |
| --- | --- | --- | --- |
| **Step 1: Task & Architecture Framing** | Introduce the two planetary tasks, dataset scale, and classical vs. hybrid QCNN pipelines. | `app.py` → `pages/1_Overview.py`<br> | Confirm all artifacts show **Verified** on `app.py`. On `1_Overview.py`, walk through the **Coursework Demo Walkthrough** card, Mermaid architecture diagrams (Mars ResNet-18 11.2M vs 8-qubit QCNN head 4,152 params; Moon CNN 59,347 vs QCNN head 568 vs MLP head 547 params), and headline summary tables.

 |
| **Step 2: Raw Sensor Data & Ground Truth** | Show real Mars HiRISE tiles and Moon LROC crater images alongside their ground-truth targets/labels. | `pages/2_Training_Data_Explorer.py`<br> | **Mars Tab:** Show how Leave-One-Site-Out (LOSO) folds split the 5 landing sites into Train (3), Val (1), and Test (1). Inspect a 64×64 HiRISE patch (2nd–98th percentile stretch) and its continuous slope, roughness, and hazard targets.

<br>

<br>**Moon Tab:** Browse `ejecta`, `oldcrater`, and `none` 227×227 JPEGs, noting the 68.4% `none` class imbalance and unverified source observation IDs.

 |
| **Step 3: Mars Sample-Level Prediction & Proxy Inspection** | Inspect a held-out Mars test patch, compare CNN vs. CNN→QCNN regression outputs against ground truth, and evaluate the low-risk proxy. | `pages/3_Mars_Test_Explorer.py`<br> | Click the new **Quick-Select Demo Presets** (*Both Agree Safe*, *Models Disagree*, *CNN False Positive*, *QCNN False Negative*). Examine the paired prediction cards, absolute error bar chart, and Bivariate Threshold Rule Table ($\text{Slope} < 8.0^\circ \land \text{Hazard} < 0.05$), emphasizing that the rule is a simplified research proxy, not flight-qualified landing safety.

 |
| **Step 4: Moon Sample-Level Multi-Model Inspection** | Inspect held-out lunar test images across CNN, Hybrid QCNN, and Frozen MLP with derived softmax probabilities. | `pages/4_Moon_Test_Explorer.py`<br> | Use the **Quick-Select Demo Presets** or **Confusion-Cell Filter** (e.g., *Hybrid QCNN Beats CNN* or `True: oldcrater → Pred: none`). Compare the 3 model cards, horizontal softmax distribution chart, and raw logits table.

 |
| **Step 5: Aggregate Statistical Comparison** | Transition from single-sample intuition to rigorous multi-fold/multi-seed statistical findings. | `pages/5_Model_Comparison.py`<br> | **Tab 1 (Mars):** Show lower mean QCNN RMSE, but highlight the Paired Differences Table where all 95% t-intervals ($df=4$) **cross zero**, plus the safety trade-off (CNN higher recall/coverage; QCNN lower FPR) and validation-calibration domain shift.

<br>

<br>**Tab 2 (Moon):** Compare Macro-F1 and Balanced Accuracy vs. Majority Baseline (68.4% accuracy = 33.3% balanced accuracy) and inspect 3×3 confusion matrices.

<br>

<br>**Tab 3 & 4:** Inspect epoch loss curves, parameter counts (`59,347` CNN training params), and non-comparable cached-embedding timing.

 |
| **Step 6: Scientific Limitations & Defense Q&A** | Conclude with transparent methodological caveats and experimental provenance. | `pages/6_Methodology.py`<br> | Review the 5 academic expanders: spatial correlation within the 5 Mars sites ($df=4$), pipeline asymmetry, Moon pre-computed test embeddings disclosure, unverified LROCNet geographic coordinates, and why these preliminary results do not claim quantum speedup or landing safety.

 |

---

## 6. Implementation Phases

To keep modifications safe, easily verifiable, and minimal in prompt expenditure, the work is divided into **3 focused phases** (2 core P0/P1 phases + 1 optional P2/polish phase):

### Phase 1: P0 Data-Integrity Verification & P1 Sample Inspector Ergonomics (Mars & Moon)

* **Goal:** Verify all P0 data/scientific invariants in `data_loading/` and upgrade `pages/3_Mars_Test_Explorer.py` and `pages/4_Moon_Test_Explorer.py` with Quick-Select Demo Presets, Prev/Next sample stepping, Moon Confusion-Cell filtering, complete sample metadata headers, and safe empty states.


* **Files in Scope:** `data_loading/mars_data.py`, `data_loading/moon_data.py`, `components/filters.py`, `pages/3_Mars_Test_Explorer.py`, `pages/4_Moon_Test_Explorer.py`, `tests/test_m1_data_loading.py`, `tests/test_m4_mars_test_explorer.py`, `tests/test_m5_moon_test_explorer.py`.


* **Dependency:** First phase; establishes verified data invariants and interactive sample selection before guided walkthrough links are added.

### Phase 2: P1 Guided Coursework Walkthrough & Aggregate Comparison Polish

* **Goal:** Add the 6-step **Coursework Demo Walkthrough** guide to `app.py` / `pages/1_Overview.py` with step-by-step navigation cues across pages `1`–`6`, and polish `pages/5_Model_Comparison.py`, `pages/6_Methodology.py`, `components/charts.py`, and `components/metric_tables.py` so every chart unit, error bar label (`± 1 Sample SD`), parameter count (`59,347` vs `195`), timing caveat, and test-embedding disclosure is unambiguous.


* **Files in Scope:** `app.py`, `pages/1_Overview.py`, `pages/2_Training_Data_Explorer.py`, `pages/5_Model_Comparison.py`, `pages/6_Methodology.py`, `components/charts.py`, `components/metric_tables.py`, `tests/test_m2_overview.py`, `tests/test_m6_model_comparison.py`, `tests/test_m7_methodology.py`.


* **Dependency:** Builds on Phase 1 preset names and filter controls so the walkthrough guide can reference exact buttons/presets on Pages 3 and 4.

### Phase 3 (Optional P2): Intra-Site Grid Map, CSV Export & Final Visual QA

* **Goal:** If desired after Phase 1 and Phase 2 pass, add the collapsible **Intra-Site 2D Grid Scatter Plot** (`col` vs `row` from `metadata.parquet`) in `pages/3_Mars_Test_Explorer.py`, add CSV download buttons for filtered test tables on Pages 3 and 4, and execute full regression and viewport visual QA (`1920×1080` and `1366×768`).


* **Files in Scope:** `pages/3_Mars_Test_Explorer.py`, `pages/4_Moon_Test_Explorer.py`, `components/charts.py`, `tests/test_m4_mars_test_explorer.py`, `tests/test_m5_moon_test_explorer.py`.


* **Dependency:** Strictly optional; only execute after Phases 1 and 2 are complete and verified.

---

## 7. Copy-Paste-Ready Gemini Flash Prompts

### Prompt 1 — Phase 1: P0 Verification & P1 Mars/Moon Sample Inspector Enhancements

```markdown
You are implementing **Phase 1** of `D:\Mars-QCNN\visualization_app\ENHANCEMENT_PLAN.md` (P0 Data-Integrity Verification & P1 Mars/Moon Test Explorer Ergonomics).

### 1. Mandatory Boundaries (Strictly Enforced)
- Read `D:\Mars-QCNN\visualization_app\ENHANCEMENT_PLAN.md` and inspect the actual files before editing.
- Keep ALL file edits strictly inside `D:\Mars-QCNN\visualization_app\`.
- NEVER modify, move, or delete datasets, `.npy` shards, parquet files, JSON logs, checkpoints, or experiment code outside `visualization_app`.
- NEVER retrain models, run PyTorch/PennyLane inference, or add new package dependencies.
- NEVER fabricate sample IDs, coordinates, predictions, or confidence scores.
- Preserve existing working features and ensure all 27 existing pytest tests continue to pass.

### 2. Inspection & Verification Steps (Do This First)
Inspect `data_loading/paths.py`, `data_loading/mars_data.py`, `data_loading/moon_data.py`, `components/filters.py`, `pages/3_Mars_Test_Explorer.py`, `pages/4_Moon_Test_Explorer.py`, and `tests/test_m1_data_loading.py`, `tests/test_m4_mars_test_explorer.py`, `tests/test_m5_moon_test_explorer.py`.
Verify and confirm:
1. Mars predictions load exclusively from `D:\Mars-QCNN\runs\pipeline_repair_audit\retraining_seedset` (5 folds × 3 seeds = 15 runs), and `tile_id` + `image_id` + ground-truth columns (`true_mean_slope_5m`, `true_roughness_m`, `true_hazard_fraction`, `true_safe`) align 100% between `model1_resnet18`, `model2_qcnn`, and `metadata.parquet`.
2. Mars low-risk proxy strictly evaluates `(mean_slope_5m < 8.0) and (hazard_fraction < 0.05)` and matches saved `pred_safe` / `true_safe` booleans.
3. Moon results load exclusively from `D:\Mars-QCNN\data\raw\moon\lrocnet\experiment_lunar_cnn_qcnn\runs\logs\seed_{42,43,44}\results.json`, class indices map strictly to `0: ejecta`, `1: oldcrater`, `2: none`, all 3 models (`cnn`, `hybrid_qcnn`, `frozen_embedding_mlp`) share identical `test.sample_ids` (779 images) and `test.ground_truth`, and softmax probabilities sum to 1.0 with argmax matching `test.predictions`.

### 3. Implementation Scope
1. **`pages/3_Mars_Test_Explorer.py` (and helper functions in `data_loading/mars_data.py` or `components/filters.py` if appropriate):**
   - Add a **Quick-Select Demo Case** control bar (buttons or horizontal radio/pills) above or beside the patch selector that dynamically picks a deterministic representative patch from the active `(fold, seed)` for:
     - *Manual / Filter Selection* (default)
     - *Both Models Agree Safe (True Positive)*
     - *Model Disagreement (`CNN != QCNN`)*
     - *CNN False Positive (Unsafe predicted Safe)*
     - *QCNN False Negative (Safe predicted Unsafe)*
     (If a category has 0 patches in a given fold/seed, display a polite `st.caption` / `st.info` fallback rather than crashing.)
   - Add **Prev Patch / Next Patch** stepping buttons beside the patch selector dropdown so the presenter does not have to scroll through 2,195 items.
   - Ensure the selected patch metadata header clearly displays `site_id`, `image_id`, `tile_id`, `fold`, `seed`, and local intra-site grid coordinates (`row`, `col`), plus a clear note that `mean slope < 8° AND hazard fraction < 0.05` is a simplified terrain proxy, not verified landing safety or habitability.
   - Ensure both signed error ($\Delta = \text{pred} - \text{true}$) and absolute error ($\vert{}\text{pred} - \text{true}\vert{}$) are clearly labeled with units (`°` for slope, `m` for roughness, unitless fraction for hazard).

2. **`pages/4_Moon_Test_Explorer.py` (and helper functions if appropriate):**
   - Add a **Quick-Select Demo Case** control bar that dynamically selects a deterministic representative test image from the active seed for:
     - *Manual / Filter Selection* (default)
     - *All 3 Models Agree & Correct*
     - *Hybrid QCNN Correct / Classical CNN Wrong*
     - *Classical CNN Correct / Hybrid QCNN Wrong*
     - *All 3 Models Wrong (Hard Crater Example)*
   - Add an optional **Confusion Pair Filter** (`All Pairs` or specific `True Class → Model Predicted Class` combinations) or ensure existing class + outcome filters allow 1-click isolation of specific error modes.
   - Add **Prev Image / Next Image** stepping buttons beside the image dropdown so the presenter does not have to scroll through 779 items.
   - Ensure empty filter combinations render a clean `st.info` message without raising `IndexError`.

3. **Automated Tests (`tests/test_m1_data_loading.py`, `tests/test_m4_mars_test_explorer.py`, `tests/test_m5_moon_test_explorer.py`):**
   - Add targeted unit/acceptance tests verifying:
     - Cross-seed and cross-model `sample_ids` alignment (all 779 Moon test samples identical across seeds 42, 43, 44).
     - Preset case selector helper logic for Mars and Moon returns valid sample rows (or handles empty categories gracefully).
     - Mars metadata (`site_id`, `image_id`, `row`, `col`) aligns with paired test predictions.

### 4. Verification & Required Completion Report
Run `.\.venv\Scripts\pytest tests/ -v` from `D:\Mars-QCNN\visualization_app` and verify the app launches cleanly.
Provide a structured completion report listing:
1. Answers to the Phase 1 inspection checks (verified facts found in code).
2. Exact files modified and functions/controls added.
3. Full `pytest` output showing all existing + new tests passing.
4. Any checks that could not be performed.

```

---

### Prompt 2 — Phase 2: P1 Guided Coursework Walkthrough & Aggregate Comparison Polish

```markdown
You are implementing **Phase 2** of `D:\Mars-QCNN\visualization_app\ENHANCEMENT_PLAN.md` (P1 Guided Presentation Flow, Aggregate Results Polish & Scientific Disclosures).

### 1. Mandatory Boundaries (Strictly Enforced)
- Read `D:\Mars-QCNN\visualization_app\ENHANCEMENT_PLAN.md` and inspect all target files before editing.
- Keep ALL file edits strictly inside `D:\Mars-QCNN\visualization_app\`.
- NEVER modify datasets, saved predictions, metrics, or experiment artifacts outside `visualization_app`.
- NEVER retrain models or add external dependencies.
- Preserve all existing functionality and tests from Phase 1.

### 2. Inspection & Verification Steps (Do This First)
Inspect `app.py`, `pages/1_Overview.py`, `pages/2_Training_Data_Explorer.py`, `pages/5_Model_Comparison.py`, `pages/6_Methodology.py`, `components/charts.py`, and `components/metric_tables.py`.
Check:
1. Whether every chart in `components/charts.py` and `pages/5_Model_Comparison.py` displays explicit physical units (`°`, `m`, fraction, `%`) and states whether error bars represent `± 1 Sample SD across 5 sites` (Mars) or `± 1 Sample SD across 3 seeds` (Moon).
2. Whether `5_Model_Comparison.py` and `6_Methodology.py` clearly state all mandatory scientific constraints:
   - Mars: Primary evaluation is the 15-run repaired seedset (`retraining_seedset`); QCNN has lower mean RMSE, but paired site-level 95% t-intervals ($df=4$, 5 sites) cross zero for all targets; CNN has higher safe recall ($5.56\%$ vs $2.77\%$) and coverage ($5.67\%$ vs $3.02\%$) while QCNN has lower FPR ($2.78\%$ vs $5.27\%$); validation-calibrated thresholds suffer domain shift (39.65% CNN / 33.94% QCNN test FPR).
   - Moon: Hybrid QCNN has higher mean balanced accuracy and macro-F1 while raw accuracy is virtually tied (preliminary, not proven quantum advantage); majority baseline achieves 68.4% raw accuracy but 33.3% balanced accuracy; CNN trained with **59,347** trainable parameters (vs 195 post-freeze linear classifier in legacy JSON), Hybrid QCNN head has **568**, and Frozen MLP head has **547**; QCNN head timing (~2.8 ms/img) uses cached 8D embeddings and excludes encoder execution vs end-to-end CNN (~4.6 ms/img); completed Moon run computed test embeddings before fitting heads, though test labels were never used for training, model selection, or threshold selection.

### 3. Implementation Scope
1. **Guided Demonstration Flow (`app.py` and `pages/1_Overview.py` through `pages/6_Methodology.py`):**
   - Add a clean, prominent **"🎓 Coursework Demonstration Walkthrough (6-Step Guide)"** expander or card on `1_Overview.py` (and a brief reference on `app.py`) outlining the exact presentation flow:
     1. *Task & Architecture (`1_Overview`)*
     2. *Dataset & Ground Truth (`2_Training_Data_Explorer`)*
     3. *Mars Sample Inspection & Low-Risk Proxy (`3_Mars_Test_Explorer`)*
     4. *Moon Multi-Model Sample Inspection (`4_Moon_Test_Explorer`)*
     5. *Aggregate Metrics, Error Bars & Trade-offs (`5_Model_Comparison`)*
     6. *Methodological Caveats & Provenance (`6_Methodology`)*
   - At the bottom of each page (`1` through `6`), add a compact **"Next in Demonstration Walkthrough"** navigation footer (using `st.page_link` if supported by the installed Streamlit version, or a styled callout banner with the next page name and what to highlight next). Do NOT replace Streamlit's native sidebar navigation.

2. **Aggregate Results & Visual Polish (`pages/5_Model_Comparison.py`, `pages/6_Methodology.py`, `components/charts.py`, `components/metric_tables.py`):**
   - Refine chart titles, axis labels, and hover templates in `components/charts.py` and `pages/5_Model_Comparison.py` so units (`°`, `m`, fraction, `%`) and error-bar definitions (`± 1 Sample SD`) are explicit on every figure.
   - Ensure `pages/5_Model_Comparison.py` Tab 4 (Capacity & Timing Analysis) and `pages/6_Methodology.py` explicitly include the **Moon Test Embedding Computation Disclosure** (*test embeddings were computed prior to fitting QCNN/MLP heads in the completed run, but test labels were never used for gradient updates, model selection, or threshold selection*) alongside the `59,347` vs `195` parameter clarification and cached-embedding latency disclosure.
   - Ensure Mars Tab 1 in `pages/5_Model_Comparison.py` clearly explains both the fixed-threshold trade-off (QCNN lower FPR vs CNN higher safe recall/coverage) and the validation-calibrated threshold generalization gap across sites.

3. **Automated Tests (`tests/test_m2_overview.py`, `tests/test_m6_model_comparison.py`, `tests/test_m7_methodology.py`):**
   - Add or update tests verifying that the mandatory disclosures (59,347 parameters, cached-embedding latency non-comparability, test-embedding computation disclosure, zero-crossing $df=4$ intervals) and chart titles/labels are present and accurate.

### 4. Verification & Required Completion Report
Run `.\.venv\Scripts\pytest tests/ -v` from `D:\Mars-QCNN\visualization_app` and verify the app launches without errors.
Provide a structured completion report listing:
1. Summary of disclosures and chart labels verified/updated.
2. Exact files modified.
3. Full `pytest` output.
4. Any checks that could not be performed.

```

---

### Prompt 3 — Phase 3 (Optional P2): Intra-Site Grid Scatter Plot, CSV Export & Visual QA

```markdown
You are implementing **Phase 3 (Optional P2 Extras & Final QA)** of `D:\Mars-QCNN\visualization_app\ENHANCEMENT_PLAN.md`. Only proceed if Phases 1 and 2 are complete and all tests pass.

### 1. Mandatory Boundaries (Strictly Enforced)
- Keep ALL file edits strictly inside `D:\Mars-QCNN\visualization_app\`.
- NEVER modify datasets, predictions, metrics, or files outside `visualization_app`.
- NEVER invent geographic coordinates (lat/lon) or planetary maps for Mars or Moon.
- Preserve all existing features and tests.

### 2. Implementation Scope
1. **Mars Intra-Site Local Grid Overview (`pages/3_Mars_Test_Explorer.py` & `components/charts.py`):**
   - Inspect how `row` and `col` from `metadata.parquet` are available in the `3_Mars_Test_Explorer.py` test dataframe.
   - Add a collapsible expander (`"🗺️ Intra-Site Tile Grid Overview (Local Mosaic Coordinates)"`) that renders a 2D Plotly scatter plot of `col` (x-axis) vs `row` (y-axis, inverted so row 0 is at the top) for the active test site (`fold`/`seed`), with a selector to color points by:
     - *Ground Truth Safe Proxy (`true_safe`)*
     - *Model Agreement (`CNN == QCNN` vs `CNN != QCNN`)*
     - *CNN vs QCNN Absolute Slope Error*
   - Highlight the currently selected `tile_id` on the scatter plot with a distinct marker outline/star.
   - **Mandatory Honesty Rule:** Title the axes `Intra-Site Grid Col (px/tile offset)` and `Intra-Site Grid Row (px/tile offset)` and include an explicit caption stating: *"Unprojected local HiRISE mosaic grid coordinates (`row`, `col` from `metadata.parquet`), not global Martian latitude/longitude."*

2. **Filtered Table CSV Export (`pages/3_Mars_Test_Explorer.py` & `pages/4_Moon_Test_Explorer.py`):**
   - Add a compact `st.download_button` (`"📥 Download Filtered Predictions (CSV)"`) on both test explorer pages allowing the user to export the currently filtered test dataframe slice.

3. **Automated Tests & Final QA:**
   - Add a test in `tests/test_m4_mars_test_explorer.py` confirming the intra-site grid chart builder returns a valid `plotly.graph_objects.Figure` with non-empty traces and accurate `row`/`col` bounds.
   - Run the complete test suite: `.\.venv\Scripts\pytest tests/ -v`.
   - If browser inspection is available in your environment, inspect the live app at `1920×1080` and `1366×768` across all 6 pages and verify zero layout clipping, overlapping badges, or horizontal overflow. If browser access is unavailable, explicitly report that visual viewport checks must be performed manually by the user.

### 3. Required Completion Report
Provide a final report detailing:
1. Files modified and P2 features added.
2. Full `pytest` test execution output.
3. Status of each item in the Regression and Visual QA Checklist (distinguishing automated checks from manual browser checks).

```

---

## 8. Regression and Visual QA Checklist

Use this checklist after implementation to verify the application before the coursework demonstration:

### 8.1 Environment & Automated Regression Checks

```powershell
cd D:\Mars-QCNN\visualization_app
.\.venv\Scripts\activate
.\.venv\Scripts\pytest tests/ -v
.\.venv\Scripts\streamlit run app.py

```

* [ ] All 27 baseline tests + newly added Phase 1–3 tests pass in `pytest` with zero warnings/failures regarding missing artifacts.


* [ ] `git status` (or file modification check) confirms zero files outside `D:\Mars-QCNN\visualization_app` were touched.


* [ ] `app.py` launches in `< 2` seconds and all artifact status indicators show **Verified**.



### 8.2 Sample-Level Data-Integrity Checks

* [ ] **Mars Patch Inspector (`3_Mars_Test_Explorer.py`):**
* [ ] Select Fold 0 (`site_01`), Seed 42 (2,195 test patches).


* [ ] Click each **Quick-Select Demo Preset** (*Both Agree Safe*, *Model Disagreement*, *CNN False Positive*, *QCNN False Negative*) and confirm the displayed patch, `site_id`, `image_id`, `tile_id`, `row`, `col`, ground-truth values, and model predictions update immediately.
* [ ] Confirm that for any selected patch, $\text{Safe} \iff (\text{Mean Slope} < 8.0^\circ) \land (\text{Hazard Fraction} < 0.05)$ holds identically in the Bivariate Threshold Rule Table for Ground Truth, ResNet-18, and QCNN.




* [ ] **Moon Test Inspector (`4_Moon_Test_Explorer.py`):**
* [ ] Select Seed 42 (779 test images: 89 `ejecta`, 157 `oldcrater`, 533 `none`).


* [ ] Click each **Quick-Select Demo Preset** (*All 3 Agree Correct*, *Hybrid Beats CNN*, *CNN Beats Hybrid*, *All 3 Wrong*) and confirm the 227×227 JPEG, ground-truth badge, 3 model cards, softmax bar chart, and raw logits table update cleanly.


* [ ] Verify that displayed softmax percentages sum to $100.0\%$ ($\sum p_i = 1.0$) and the highest-probability class matches each model's displayed prediction.





### 8.3 Manual Visual QA (Browser Viewport Checks)

*(Note: Do not claim these visual checks have occurred until inspected in an actual browser window at both resolutions.)*

* [ ] **Desktop Presentation (`1920×1080`):**
* [ ] `1_Overview.py`: System readiness cards, Mermaid flowcharts, and headline tables render side-by-side without horizontal scrollbars.


* [ ] `3_Mars_Test_Explorer.py` & `4_Moon_Test_Explorer.py`: 256px/280px image cards and side-by-side model prediction cards align cleanly with readable badge contrast (`#4FC3F7` CNN Blue, `#AB47BC` QCNN Purple, `#78909C` MLP Gray, `#66BB6A` Safe Green, `#EF5350` Unsafe Red).


* [ ] `5_Model_Comparison.py`: 3×3 Moon confusion matrix heatmaps and Mars grouped bar charts display complete axis titles, units, and legends without text clipping.




* [ ] **Laptop Display (`1366×768`):**
* [ ] With the Streamlit sidebar expanded, verify that 3-column and 4-column metric/prediction cards wrap or scale legibly without truncating numbers or badge labels.
* [ ] Verify Plotly legends do not overlap chart bars or titles at narrower container widths.



---

## 9. Final Acceptance Criteria

The enhanced Mars + Moon CNN/QCNN Research Explorer is ready for coursework demonstration when all of the following measurable conditions are met:

1. **Zero Artifact Mutation:** All files outside `D:\Mars-QCNN\visualization_app` remain completely untouched, and runtime requires zero live PyTorch or PennyLane execution.


2. **100% Test Suite Pass Rate:** `.\.venv\Scripts\pytest tests/ -v` passes all 27 original acceptance tests plus all new integrity/preset tests.


3. **One-Click Sample Storytelling:** On both `3_Mars_Test_Explorer.py` and `4_Moon_Test_Explorer.py`, the presenter can switch between representative agreement, disagreement, false-positive, and false-negative examples in a single click (via Quick-Select Demo Presets or Prev/Next buttons) without hunting through 2,195 or 779 dropdown entries.


4. **End-to-End Walkthrough Cohesion:** An examiner can follow the 6-step walkthrough from `1_Overview.py` through `6_Methodology.py` covering *Dataset Examples → Model Predictions → Ground Truth → Model Comparison → Limitations* with clear visual hierarchy and explicit unit labels.
5. **Full Compliance with Scientific Reporting Constraints:**
* Mars metrics strictly reflect the 15-run `retraining_seedset`, explicitly show $df=4$ zero-crossing 95% t-intervals, highlight the fixed-threshold recall/coverage vs. FPR trade-off and validation calibration shift, and state that $\text{slope} < 8^\circ \land \text{hazard} < 0.05$ is a simplified terrain proxy rather than flight-qualified landing safety.


* Moon metrics strictly reflect the 3-seed LROCNet benchmark alongside the majority baseline (68.4% raw accuracy vs. 33.3% balanced accuracy), report the corrected `59,347` trainable CNN parameters alongside `568` (QCNN head) and `547` (MLP head), disclose the pre-computed test embeddings workflow and non-comparable cached-embedding head latency, and make zero claims of proven quantum advantage or unverified lunar coordinates.