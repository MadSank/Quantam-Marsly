# Mars + Moon CNN/QCNN Research Explorer

An interactive scientific visualization platform for paired classical convolutional neural networks (CNNs) and variational quantum convolutional neural networks (QCNNs) applied to planetary surface analysis:

1. **Mars Terrain Prediction (Continuous Regression & Landing-Safety Proxy):**
   - High-resolution HiRISE orbital imagery and DTM patches (64×64 @ 2 m/pixel, 15,233 total patches).
   - Predicting 5 m mean slope (°), roughness (m), and hazard fraction.
   - Bivariate safe-proxy decision rule: `mean slope < 8.0°` AND `hazard fraction < 0.05`.
   - Leave-one-site-out (LOSO) spatial cross-validation across 5 geological landing sites.
   - Primary evaluation: 15 paired runs (5 folds × 3 random seeds: 42, 43, 44).

2. **Moon Crater Classification (LROCNet):**
   - Multi-class lunar surface structure classification: `ejecta` (0), `oldcrater` (1), `none` (2).
   - 5,000 images (3,608 train, 613 validation, 779 test) from the Lunar Reconnaissance Orbiter Camera (LROC).
   - Multi-model benchmark: End-to-end CNN, Hybrid CNN-QCNN (8-qubit variational circuit), and matched Classical Frozen-Embedding MLP control across seeds 42, 43, 44.

---

## 🚀 Quick Start & Setup

### Prerequisites
- Python 3.11 (or 3.10+)
- Windows, macOS, or Linux

### 1. Environment Setup
The visualization application runs in an isolated virtual environment located inside `visualization_app/`:

```powershell
# Navigate to application directory
cd D:\Mars-QCNN\visualization_app

# Create virtual environment (using uv or standard python venv)
uv venv .venv --python 3.11
# or: python -m venv .venv

# Activate the virtual environment
.\.venv\Scripts\activate

# Install application dependencies
uv pip install -r requirements.txt
# or: pip install -r requirements.txt
```

### 2. Launch the Application
Launch Streamlit using the dedicated virtual environment:

```powershell
cd D:\Mars-QCNN\visualization_app
.\.venv\Scripts\streamlit run app.py
```

The application will launch and open in your default browser at:
**`http://localhost:8501`**

To specify a custom port or run headless:
```powershell
.\.venv\Scripts\streamlit run app.py --server.port 8505 --server.headless true
```

---

## 🗺️ Application Architecture & Page Guide

The app follows Streamlit's multipage architecture under `visualization_app/pages/`:

| Page | Description | Source Artifacts |
|---|---|---|
| **`app.py`** | Application landing entry point, project overview, and artifact readiness status check. | System paths |
| **`1_Overview.py`** | Executive summary, headline metrics matching `FINAL_READINESS_AUDIT.md`, and pipeline architecture workflow diagrams. | `fixed_threshold_aggregate.csv`, `recomputed_metrics.csv` |
| **`2_Training_Data_Explorer.py`** | Interactive browser for 15,233 Mars HiRISE patches (with per-tile 2nd–98th percentile stretch) and 5,000 Moon crater images with verified provenance manifests. | `shards/*.npy`, `metadata.parquet`, `provenance_manifest.csv` |
| **`3_Mars_Test_Explorer.py`** | Held-out test-patch inspector for all 15 fold/seed runs (e.g. 2,195 patches in Fold 0). Displays paired ResNet-18 vs QCNN predictions, bivariate threshold decision breakdown table, error sorting, and model agreement filters. | `test_predictions.parquet` (CNN & QCNN) |
| **`4_Moon_Test_Explorer.py`** | Inspection of all 779 test images across seeds 42, 43, 44. Displays Classical CNN, Hybrid QCNN, and Frozen MLP predictions with derived softmax confidence distributions and disagreement filters. | `results.json`, `lrocnet/test/*/*.jpg` |
| **`5_Model_Comparison.py`** | Comprehensive comparative analysis: grouped bar charts with sample SD error bars across 5 sites / 3 seeds, paired difference t-intervals (crossing zero), confusion matrix heatmaps, epoch loss curves, parameter counts, and timing analysis. | `fixed_threshold_aggregate.csv`, `history.csv`, `epoch_log.csv` |
| **`6_Methodology.py`** | Scientific documentation covering data provenance, model architectures, training asymmetry (end-to-end vs frozen representations), spatial statistics, and essential research disclaimers. | `FINAL_READINESS_AUDIT.md` |

---

## 🔬 Scientific Honesty Rules & Disclaimers

This application operates strictly under the research integrity guidelines established in `FINAL_READINESS_AUDIT.md`:

1. **No Live Retraining:** Checkpoints, predictions, and metrics are loaded from pre-computed experiment artifacts. No models are trained or retrained.
2. **Landing Safety Proxy:** The "safe" label is a derived mathematical proxy (`mean slope < 8.0°` AND `hazard fraction < 0.05`). It is **not** an operational landing safety system or flight-certified hazard avoidance algorithm.
3. **Corrected Moon Parameter Count:** The Classical CNN contains **59,347 trainable parameters during training**. The 195 recorded in legacy JSON reflects only the classifier head after freezing the backbone for hybrid training.
4. **Non-Comparable Latencies:** In Moon logs, Classical CNN timing represents full end-to-end inference (227×227 RGB -> prediction), whereas QCNN and MLP timings were measured only on pre-cached 8D embeddings. They are **not** apples-to-apples latency comparisons.
5. **No Quantum Speedup Claimed:** All quantum circuits were simulated numerically on classical hardware. No quantum advantage or superiority is claimed.
6. **Lunar Data Imbalance:** The Moon test set is 68.4% `none`. A naive majority-class classifier attains 68.4% raw accuracy with 33.3% balanced accuracy and 0.271 macro-F1. Evaluators must reference balanced accuracy and macro-F1.
7. **Lunar Provenance:** Provenance status is `FILE_FACTS_CONFIRMED_SOURCE_ID_UNVERIFIED`. Source observation IDs and geographic site coordinates remain unverified.
8. **Uncertainty Reporting:** All error bounds and intervals are reported as empirical sample standard deviations (SD) or nominal 5-site paired t-intervals; they are not asymptotic confidence bounds.

---

## 🛠️ Testing & Verification

Acceptance tests are implemented using `pytest` under `visualization_app/tests/`:

```powershell
cd D:\Mars-QCNN\visualization_app
.\.venv\Scripts\pytest tests/ -v
```

All 27 acceptance tests verify:
- Shard memory-mapping, float16 loading, and per-tile display stretching.
- Parquet schema consistency and paired tile ID matching.
- Exact headline metric parity with `FINAL_READINESS_AUDIT.md`.
- Softmax probability normalization ($\sum p_i = 1.0$) and argmax prediction consistency.
- Leave-one-site-out fold-to-site mapping.
- Provenance manifest integrity.
- Paired differences zero-crossing verification.

---

## ⚙️ Path Configuration

All paths are resolved automatically relative to the workspace root `D:\Mars-QCNN`. To override the root directory (e.g. in CI or on another machine), set the environment variable:

```powershell
$env:MARS_QCNN_ROOT = "D:\Mars-QCNN"
```
