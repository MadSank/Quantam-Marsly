# Controlled all-site CNN vs CNN→QCNN evaluation — results audit

**Status:** Complete for one predeclared seed (42), all five leave-one-site-out folds. This report and its artifacts are isolated under `runs/controlled_evaluation`; the prior experiment results and production pipeline were not edited. No model training was launched after completion of this evaluation.

## Protocol and integrity checks

- The controlled dataset contains 15,233 coverage-eligible patches with no per-class cap. Site counts were 2,195 (site_01), 1,885 (site_02), 2,484 (site_03), 979 (site_04), and 7,690 (site_05). The test membership was determined by site/coverage, not by class selection.
- In every fold, CNN and CNN→QCNN prediction files have identical ordered tile IDs and identical `true_safe` labels. Fold counts and safe/unsafe counts match: site_01 212/1,983; site_02 750/1,135; site_03 808/1,676; site_04 151/828; site_05 4,925/2,765. No prediction table contained missing values.
- Splits hold out one full site for testing and a separate full site for validation; three sites train. Validation and test tile IDs are checked disjoint. Checkpoint selection/early stopping uses validation loss only. Fold radiometric I/F normalization bounds are fitted on valid pixels from training-site patches only. The test imagery is run for prediction after the model is selected, but its labels are not used for fitting or threshold selection.
- Seed 42 was used for all five folds. Three seeds were not run; this is an initial controlled cross-site comparison, not a multi-seed estimate.
- Both models use 2 m pixels, 64×64 tiles (128 m square), the same three regression targets (5 m mean slope, roughness, hazard fraction), log1p target scaling, safety definition (mean slope <8° and hazard fraction <0.05), objective, fold plan, seed, optimizer schedule and validation selection rule. Augmentation is disabled for both. Input I/F normalization is fit fold-wise on training data.
- CNN has 11,242,563 trainable parameters and is trained end-to-end from randomly initialized ResNet-18 (`pretrained: false`). CNN→QCNN obtains fold-specific embeddings from that fold’s best CNN checkpoint; the backbone is then frozen and embeddings are cached. The bottleneck/quantum head has 4,152 trainable parameters (48 in the final head). Thus the experiment compares an end-to-end CNN head with a small head on a trained frozen CNN representation; it does not compare equal trainable parameter counts or two independently trained end-to-end backbones. The QCNN simulator ran on CPU; CNN portions used CUDA.

## Safety audit

The fixed definition is `safe_label == 1` when the ground-truth 5 m mean slope is below 8° **and** the hazard fraction is below 0.05. A predicted safe tile is a positive. Therefore TP = predicted-safe/actually-safe, FP = predicted-safe/actually-unsafe, TN = predicted-unsafe/actually-unsafe, FN = predicted-unsafe/actually-safe. `false_positive_rate = FP/(FP+TN)` measures unsafe tiles incorrectly declared safe; `false_discovery_rate = FP/(TP+FP)` is the unsafe share among declared-safe tiles. Empty denominators are NaN.

**Default fixed safety threshold (risk < 1):**

| Held-out site | Model | TP | FP | TN | FN | Safe recall | Unsafe FPR | Safe precision | Predicted safe |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| site_01 | CNN | 0 | 6 | 1,977 | 212 | 0.0% | 0.30% | 0.0% | 6 (0.27%) |
| site_01 | CNN→QCNN | 0 | 18 | 1,965 | 212 | 0.0% | 0.91% | 0.0% | 18 (0.82%) |
| site_02 | CNN | 0 | 0 | 1,135 | 750 | 0.0% | 0.0% | NaN | 0 (0.00%) |
| site_02 | CNN→QCNN | 9 | 9 | 1,126 | 741 | 1.20% | 0.79% | 50.0% | 18 (0.95%) |
| site_03 | CNN | 0 | 0 | 1,676 | 808 | 0.0% | 0.0% | NaN | 0 (0.00%) |
| site_03 | CNN→QCNN | 0 | 0 | 1,676 | 808 | 0.0% | 0.0% | NaN | 0 (0.00%) |
| site_04 | CNN | 0 | 0 | 828 | 151 | 0.0% | 0.0% | NaN | 0 (0.00%) |
| site_04 | CNN→QCNN | 0 | 0 | 828 | 151 | 0.0% | 0.0% | NaN | 0 (0.00%) |
| site_05 | CNN | 4,920 | 2,758 | 7 | 5 | 99.90% | 99.75% | 64.08% | 7,678 (99.84%) |
| site_05 | CNN→QCNN | 0 | 0 | 2,765 | 4,925 | 0.0% | 0.0% | NaN | 0 (0.00%) |

Macro-averaged across sites (each site weighted equally), fixed-threshold safe recall/FPR/predicted-safe coverage are **20.0% / 20.0% / 20.0% for CNN** and **0.24% / 0.34% / 0.35% for CNN→QCNN**. The CNN’s apparent average is driven by site_05, where it calls almost every tile safe but has 99.75% unsafe FPR; the QCNN is extremely conservative at this threshold and misses nearly all actually-safe tiles. These are not operationally acceptable safety results.

A second, explicitly exploratory operating point chooses one risk threshold per fold using only that fold’s validation site, targeting validation FPR ≤5%. On the held-out test sites, macro safe recall/FPR/coverage were **34.1% / 37.8% / 36.8% for CNN** and **24.2% / 24.5% / 24.4% for CNN→QCNN**. The nominal validation constraint did not transfer to test sites: CNN test FPR exceeded 5% in sites 01, 02, and 05; QCNN exceeded it in sites 01 and 02, and was just under in the others. Validation-selected thresholds are not a safety guarantee. They should be interpreted as a diagnostic of cross-site calibration failure, not as a recommended deployment threshold.

## Regression audit

Regression metrics were recomputed from saved inverse-transformed predictions against the same saved ground-truth values, separately per held-out site. All 15,233 predictions per model had finite values for all three targets. Targets are interpreted in degrees (mean slope), meters (roughness), and unitless hazard fraction, respectively; prediction generation applies the saved `TargetScaler.inverse_transform` before writing parquet. The controlled run uses log1p targets; no missing-value exclusions were needed.

Equal-site mean test metrics (RMSE / MAE):

| Target | CNN | CNN→QCNN |
|---|---:|---:|
| 5 m mean slope (degrees) | 8.61 / 7.25 | 8.44 / 7.08 |
| Roughness (meters) | 1.49 / 1.10 | 1.39 / 1.01 |
| Hazard fraction | 0.403 / 0.336 | 0.393 / 0.333 |

These small macro regression differences favor QCNN numerically in this single-seed run, but do not establish a reliable or practically meaningful advantage. Aggregate tables average site metrics (equal site weighting), rather than pooling all tile errors; the per-site tables preserve the site-specific results.

## Fairness, limitations, and conclusion

The data, sample IDs, targets, scaler treatment, split, seed, loss and model-selection rule are matched. The representation is also matched in the sense that QCNN consumes embeddings from the fold’s trained CNN, but the models have different trainable capacity and optimization: CNN is end-to-end with ~11.24M trainable parameters; QCNN trains a 4,152-parameter head over the frozen fold CNN. The comparison therefore tests whether that particular frozen-embedding quantum head works against the CNN’s own end-to-end regression head; it cannot support a broad claim that quantum models outperform classical CNNs. There is one seed, only five site groups, substantial site-dependent prevalence/behavior, and no external independent validation. The nominal t intervals in the generated aggregate CSVs are descriptive only and are particularly weak with five heterogeneous sites and one seed; unbounded intervals for rates should not be read as probabilities.

**Decision:** No-go for claiming a QCNN performance advantage or starting a production/training phase that treats these safety predictions as validated. The next defensible step is a reviewed evaluation protocol with at least three seeds, repeated site-held-out folds, class-prevalence-aware reporting, calibrated thresholds fixed from training/validation only, and explicit acceptance criteria for unsafe-as-safe errors. First inspect per-site prediction failures and confirm model-head/target behavior; do not tune thresholds against test sites. Keep production preprocessing and training configuration unchanged until that protocol is approved.

## Artifact index

- Per-site confusion matrices and regression metrics: `runs/controlled_evaluation/metrics/fixed_threshold_per_site_seed.csv`
- Validation-calibrated test metrics: `runs/controlled_evaluation/metrics/validation_calibrated_test_per_site_seed.csv`
- Validation-selected thresholds: `runs/controlled_evaluation/metrics/thresholds_selected_on_validation.csv`
- Paired site differences: `runs/controlled_evaluation/metrics/fixed_threshold_paired_site_differences.csv` and `runs/controlled_evaluation/metrics/validation_calibrated_test_paired_site_differences.csv`
- Diagnostic plots: `runs/controlled_evaluation/metrics/fpr_vs_safe_recall.png` and `runs/controlled_evaluation/metrics/fpr_vs_safe_coverage.png`
- Run configuration: `runs/controlled_evaluation/base_config.yaml`; fold-specific configurations and checkpoints are under `runs/controlled_evaluation/configs` and `runs/controlled_evaluation/runs`.
- Progress: all five folds for seed 42 are marked complete in `runs/controlled_evaluation/progress.json`. Focused tests for normalization, threshold logic, metrics, and RNG state passed (16 passed; existing dependency deprecation warnings were emitted).
