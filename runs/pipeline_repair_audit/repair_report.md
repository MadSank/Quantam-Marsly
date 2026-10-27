# Pipeline Repair Audit — Normalization and ResNet Stem

**Workspace:** `D:\Mars-QCNN`  
**Status:** Minimal repairs implemented; preprocessing-only verification completed.  
**Scope:** No training, inference, threshold tuning, checkpoint reads/writes, model performance evaluation, data regeneration, or production manifest/config changes.

## Findings and root causes

1. **Image contrast loss:** `fit_training_image_bounds` in `src/controlled_evaluation.py` computes training-fold-only 2nd/98th percentile bounds from valid image pixels. `make_fold_config` stores those bounds in one fold config, and that same config is passed to train, validation, and test datasets. `MarsTerrainDataset.__getitem__` in `src/data/dataset.py` affine-scales calibrated I/F and previously always clipped it to `[0,1]`. That made valid held-out values outside the training percentile interval indistinguishable: below-bound values became 0 and above-bound values became 1. Site_03 had 57.9647% high-bound clipping and site_04 had 100% low-bound clipping under the exact historical inclusive-bound count.
2. **Three-channel random stem:** `Model1ResNet18` in `src/models/cnn.py` always replaced ResNet18's `conv1`. The grayscale adaptation zeroed the replacement, then copied averaged RGB weights only into channel 0. This adaptation was applied even with `pretrained: false` and `input_channels: 3`, so two of the three randomly initialized input channels began exactly zero.

## Repairs made

- **Normalization:** New fold-fitted bounds explicitly serialize `clip: false` and a description of the affine/no-clipping application. Valid values outside the training percentile bounds now remain below 0 or above 1, retaining their ordering and contrast. Invalid image pixels are explicitly reset to zero after scaling using the existing image-valid mask. DTM-valid and image-valid channels, calibrated I/F values, upstream product/band handling, authoritative georeferencing, scale/offset, and CORE_NULL processing were not changed.
- **Compatibility:** Existing fold configs without a `clip` key retain the historical clipped behavior. This permits old artifacts/configs to remain reproducible; newly fitted folds receive `clip: false`. The no-clipping behavior is the smallest correction supported by the diagnosis: fitting remains strictly on training data and held-out distributions are not used to choose bounds.
- **ResNet stem:** Added `make_resnet_input_stem`. Three-channel stems are retained intact for pretrained and randomly initialized ResNet18. A pretrained one-channel stem is initialized from the mean of pretrained RGB filters. A non-pretrained one-channel stem receives a fresh Conv2d initialization. Unsupported pretrained channel adaptations fail explicitly rather than silently discarding the pretrained initialization.

## Files changed

- `src/data/dataset.py` — optional clipping behavior, legacy-default compatibility, explicit invalid-pixel zeroing retained.
- `src/controlled_evaluation.py` — new training-fold normalization parameters set `clip: false`.
- `src/models/cnn.py` — correct channel-aware ResNet stem initialization.
- `tests/test_controlled_evaluation.py` — no-clipping, invalid mask, train-only fitting, and YAML serialization/reload regression checks.
- `tests/test_quantum_and_models.py` — per-channel initialization/contribution and pretrained-vs-random stem tests.

Code-path references: normalization fitting and serialization are `src/controlled_evaluation.py:47-78` and `:93-94`; dataset application and invalid masking are `src/data/dataset.py:145-153` and `:193-203`. The ResNet replacement defect was `src/models/cnn.py`'s constructor path at the prior `conv1` adaptation; repaired helper and constructor call are now `src/models/cnn.py:11-43` and `:76-78`.

## Preprocessing-only audit

The audit used the already processed shards and metadata at `runs/controlled_evaluation/processed`; it did not regenerate data. Each test site was evaluated using its existing grouped fold. For each fold, bounds were refitted in memory from only the three training sites and training rows, then the same config object was applied to that fold's test rows. In the normal controlled run, this config is also the shared config passed to training and validation datasets. No validation or test pixels contributed to fitting.

| Site | Fold | Training sites used for fit | Train patches / valid pixels | Bounds (I/F) | Valid image pixels | Invalid fraction | Old clip fraction* | New valid values outside [0,1]** | New valid I/F min to max | New I/F SD | Invalid outputs nonzero |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| site_01 | 0 | 03, 04, 05 | 11,153 / 45,676,558 | 0.04312134–0.18457031 | 8,989,296 | 0.0158% | 0.00% | 0.00% | 0.170–0.939 | 0.1321 | 0 |
| site_02 | 1 | 01, 04, 05 | 10,864 / 44,492,631 | 0.04299927–0.16162109 | 7,718,576 | 0.0309% | 0.00% | 0.00% | 0.198–0.524 | 0.0660 | 0 |
| site_03 | 2 | 01, 02, 05 | 11,770 / 48,203,057 | 0.07482910–0.16125488 | 10,173,223 | 0.0122% | 57.9647% | 57.7806% | 0.508–1.596 | 0.2044 | 0 |
| site_04 | 3 | 01, 02, 03 | 6,564 / 26,881,095 | 0.07269287–0.18676758 | 4,008,150 | 0.0457% | 100.00% | 99.99998% | -0.373–0.000 | 0.0562 | 0 |
| site_05 | 4 | 02, 03, 04 | 5,348 / 21,899,949 | 0.04016113–0.18750000 | 31,495,185 | 0.0097% | 0.00% | 0.00% | 0.308–0.537 | 0.0321 | 0 |

* Historical fraction counts valid pixels at or beyond the fitted percentile bounds, which become exact 0/1 after clipping.  
** Strictly outside the normalized interval; values exactly equal to a bound map to exactly 0 or 1 without being clipped. The small difference for site_03 is due to values exactly at the high bound.

Channel 1 (DTM-valid fraction) remained in `[0,1]`, with means 0.99990, 0.99995, 0.99992, 0.99999, and 0.99994 for sites 01–05. Channel 2 (image-valid fraction) remained in `[0,1]`, with means 0.99982, 0.99969, 0.99986, 0.99948, and 0.99989. Invalid image pixels were zero after normalization for every site. Full per-site/channel statistics are in `preprocessing_audit.csv`.

The new preview shows site_03 texture remains varied rather than pinned at white, and site_04 retains its dark-terrain contrast rather than becoming uniformly black. The per-chip q02–q98 scaling is **display-only**; the numerical input still uses the saved fold affine parameters without clipping. This correction preserves signal but does not make held-out values fall within the training range, and it is not evidence of improved model performance.

## ResNet verification

The focused test confirms the non-pretrained 3-channel stem is retained with nonzero weights on all three channels. Activating each channel separately produces finite, nonzero stem output. Tests also confirm pretrained 3-channel identity, pretrained grayscale RGB averaging, and fresh non-pretrained grayscale initialization. No checkpoint or pretrained network download is needed by these tests.

## Tests and commands

Executed successfully:

```powershell
.venv-gpu\Scripts\python.exe -m pytest tests\test_controlled_evaluation.py tests\test_quantum_and_models.py tests\test_real_preprocessing.py tests\test_preprocess_and_synthetic.py tests\test_splits_and_leakage.py -q
```

Result: **20 passed**. These include synthetic terrain preprocessing, real-data mask/derivative helper tests, grouped split tests, new normalization tests, and stem initialization checks. Only existing Rasterio affine deprecation warnings were emitted.

Preprocessing audit command:

```powershell
.venv-gpu\Scripts\python.exe runs\pipeline_repair_audit\run_preprocessing_audit.py
```

It reads the existing parquet/shards and writes only inside this new audit directory. The serialized fold parameters are at `fold_normalization_parameters.yaml`.

## Artifacts

- `preprocessing_audit.csv` — numeric per-site/channel and clipping audit.
- `fold_normalization_parameters.yaml` — training-only fit parameters for each fold.
- `preprocessing_previews.png` — raw calibrated I/F, corrected normalized I/F, and validity masks for all five sites.
- `run_preprocessing_audit.py` — reproducible audit script.
- `retraining_config.yaml` — isolated future-run config derived from the controlled baseline; its output root is `runs/pipeline_repair_audit/retraining_seedset`, separate from existing results.

## Limitations and handoff

- Site_05 still has the diagnosed domain/target shift despite no clipping. This patch does not diagnose or claim to solve that behavior.
- No model inference or performance claim was made. No safety conclusion follows from the image-preprocessing preview.
- Existing clipping behavior remains when loading older configs that omit `clip`. A new run must regenerate fold configs from the repaired `fit_training_image_bounds` function.
- Valid normalized test inputs can lie outside `[0,1]`; model behavior on those inputs remains untested here.
- Terrain derivatives, their native-resolution computation, nodata handling, and the safety proxy definitions were not changed. This task did not evaluate derivative bias around nodata boundaries.

**Confirmation:** No training, calibration, threshold tuning, checkpoint modification, checkpoint-based inference, prediction generation, or model performance evaluation was run. Existing datasets, checkpoints, predictions, logs, and prior reports were preserved.

**Recommendation:** Review this isolated repair and audit, then run a new controlled retraining with the repaired code in the isolated output root. Keep the old results unchanged and compare only after new training has completed under the previously approved protocol.

Exact next command (not run):

```powershell
.venv-gpu\Scripts\python.exe -m src.controlled_evaluation --config runs\pipeline_repair_audit\retraining_config.yaml --mode run --folds all --seeds 42,43,44
```
