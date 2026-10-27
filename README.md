# Mars Terrain QCNN vs. Classical Deep Learning

Benchmarking Hybrid Quantum-Classical Convolutional Neural Networks against rigorous classical controls for physical terrain property prediction from Mars HiRISE orbital imagery.

---

## 1. Architectural Overview

| Model | Architecture | Encoding / Input | Trainable Head / Total Params | Controls |
|---|---|---|---|---|
| **Model 1** | ResNet-18 Baseline | Grayscale 1-ch Conv + Sun angles | 512-d -> MLP (128) -> 3 outputs | Standard CNN baseline |
| **Model 2** | Bottleneck QCNN | Frozen ResNet-18 512-d -> 8-d bottleneck -> 8-qubit QCNN | 48 head params (4,176 total) | Linear (27 head), Matched MLP (51 head, within 6.25%), No-entanglement QCNN, Permuted-qubits QCNN |
| **Model 3** | Quanvolutional CNN | 4-qubit 2x2 stride-2 quanv filter (16x16x4 maps) -> 3-block CNN | 3-block CNN + MLP head | Seed-matched Random Classical Filter, Plain CNN |
| **Model 4** | Pure Amplitude QCNN | 16x16 L2-normalized image -> 8/10-qubit Amplitude QCNN | 48 head params (48 total) | Tiny Classical CNN, 256-d MLP control |

### Key Scientific Safeguards
1. **Zero Spatial / Label Leakage:** 5-fold cross-validation is strictly grouped by `image_id` with hard runtime disjoint assertions. `TargetScaler` (`log1p` + z-score) is fitted exclusively on training folds.
2. **Artifact-Resistant Preprocessing:** Masks missing/PDS nodata (`<= -1e30`), sensor saturation plateaus, and CCD stitching seams (> 6x median gradient). Calculates 5 m and 15 m slope via Horn's weighted stencil.
3. **Strict Quantum Batching:** PennyLane QNodes utilize `default.qubit`, `interface="torch"`, `diff_method="backprop"`, and 2D tensor parameter broadcasting over the batch dimension, running without Python per-sample loops.
4. **Matched Classical Controls:** Model 2's MLP control has 51 parameters (within 6.25% of the 48-parameter QCNN head and 0.07% of the 4,176 total parameters).

---

## 2. Quickstart & Smoke Verification (<10 Minutes)

### Step 1: Environment Setup
```bash
# Create Python 3.11 virtual environment and install dependencies
uv venv --python 3.11 .venv
# On Windows PowerShell:
.\.venv\Scripts\activate
# On Linux / macOS:
source .venv/bin/activate

pip install --upgrade pip
pip install -e .
```

### Step 2: Automated Pytest Suite
Verifies analytical tilted-plane geometry, zero group leakage across folds, train-only standardization, PennyLane batched execution vs. loop equivalence, parameter count matching (<10%), and full smoke training of all 12 models and controls under 10 minutes:
```bash
pytest -v
```

### Step 3: Standalone Smoke Pipeline
Generates 200 synthetic tiles across 5 sites, trains all 12 model and control variants for 2 epochs, computes test-fold physical-unit metrics, and evaluates paired confidence intervals:
```bash
# Train all models and controls on synthetic smoke dataset
python -m src.train --config configs/default.yaml --smoke --all

# Evaluate test-fold predictions and render summary tables and plots
python -m src.evaluate --config configs/default.yaml --smoke

# View performance summary
cat results/smoke/summary_table.md
```

---

## 3. Full Real-Data Pipeline (HiRISE Orbitals & DTMs)

Once `data/manifest.csv` is populated with HiRISE orthoimage and DTM paths or URLs:

```bash
# 1. Download/link rasters, verify with GDAL, and extract solar angles
python -m src.data.download --config configs/default.yaml

# 2. Co-register to 0.5 m/px, mask seams/nodata, compute Horn slopes, and tile
python -m src.data.preprocess --config configs/default.yaml

# 3. Train Model 1 (ResNet-18) across all 5 folds and seeds [0, 1, 2]
python -m src.train --config configs/default.yaml --model model1 --fold all --seed all

# 4. Train Model 2 (8-qubit Bottleneck QCNN + 4 controls)
python -m src.train --config configs/default.yaml --model model2 --variant all --fold all --seed all

# 5. Train Model 3 (4-qubit Quanvolution CNN + 2 controls)
python -m src.train --config configs/default.yaml --model model3 --variant all --fold all --seed all

# 6. Train Model 4 (8/10-qubit Amplitude QCNN + 2 controls)
python -m src.train --config configs/default.yaml --model model4 --variant all --fold all --seed all

# 7. Aggregate physical metrics, calculate paired 95% CIs, and plot
python -m src.evaluate --config configs/default.yaml
cat results/summary_table.md
```

---

## 4. Repository Structure

```text
.
├── configs/
│   └── default.yaml                     # Central YAML configuration
├── data/
│   ├── manifest.csv                     # Ingestion manifest (site_id, ortho, dtm)
│   ├── raw/                             # Downloaded/linked GeoTIFFs & PDS products
│   ├── processed/                       # Float16 shards and metadata.parquet
│   └── cache/                           # Cached embeddings and quanvolution maps
├── src/
│   ├── __init__.py                      # Package root marker
│   ├── utils.py                         # Config loader, CLI overrides, seeding, atomic saving
│   ├── train.py                         # Resumable multi-model trainer
│   ├── evaluate.py                      # Original-unit metrics, paired CIs, summary CSV, plots
│   ├── data/
│   │   ├── download.py                  # Raster ingestion & GDAL verification
│   │   ├── preprocess.py                # Co-registration, seam/nodata masking, Horn slope
│   │   ├── synthetic.py                 # Analytical tilted planes & synthetic terrains
│   │   ├── splits.py                    # Grouped 5-fold CV by image_id
│   │   └── dataset.py                   # PyTorch Dataset, TargetScaler, sun-consistent augs
│   ├── quantum/
│   │   └── circuits.py                  # Batched PennyLane QCNN & Quanvolution QNodes
│   └── models/
│       ├── __init__.py                  # Model factory (build_model)
│       ├── cnn.py                       # Model 1: 1-channel ResNet-18
│       ├── bottleneck_qcnn.py           # Model 2: Bottleneck QCNN + 4 controls
│       ├── quanv_cnn.py                 # Model 3: Quanv CNN + 2 controls
│       └── pure_qcnn.py                 # Model 4: Amplitude QCNN + 2 controls
├── tests/
│   ├── conftest.py                      # Shared pytest fixtures & synthetic workspace
│   ├── test_preprocess_and_synthetic.py # Geometry, Horn slope & masking tests
│   ├── test_splits_and_leakage.py       # Grouped CV leakage & TargetScaler tests
│   ├── test_quantum_and_models.py       # QCNN batching vs loop & param matching tests
│   └── test_smoke_pipeline.py           # End-to-end 200-tile, 2-epoch benchmark
├── pyproject.toml                       # Python package specification & dependencies
└── README.md                            # Reproduction guide & documentation
```
