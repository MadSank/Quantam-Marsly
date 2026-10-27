# Cross-Site Prediction Collapse — Read-Only Diagnosis

**Workspace:** `D:\Mars-QCNN`  
**Scope:** Existing controlled seed-42, five-fold artifacts only. No training, tuning, test-label threshold selection, production edits, or changes to existing results were performed. New diagnostics are isolated in this directory.

## Executive finding

The predictions are reproducible from the saved best checkpoints and have no class inversion or binary-classifier probability bug. The most direct, confirmed input failure is fold-fitted image normalization with clipping: **100% of valid site_04 image pixels are below the train-only lower bound and become exactly 0; 57.9647% of valid site_03 pixels exceed the training upper bound and become exactly 1.** The separate DTM/image-valid channels remain near 1, so these valid scenes become an effectively constant image channel rather than nodata. The diagnostic montage shows site_04 as black and site_03 strongly clipped. This directly accounts for severe representation collapse on those sites.

Site_05 has no such clipping (0% below/above its fold bounds), yet its CNN predicts almost identical safe-like terrain values for actually safe and actually unsafe tiles. That is a cross-site feature/target generalization failure, alongside a strong label and target distribution shift. QCNN instead outputs near-constant unsafe-like regression values across site_05. No evidence points to a broken QCNN gradient path or wrong checkpoint; its regression head is simply poorly discriminative at the safety cutoffs, and its best checkpoint was the first epoch in three of five folds.

## 1. Decision and metric implementation

- There is **no safe-class probability** in either model or saved prediction table. The CNN head and QCNN readout each emit three unconstrained regression scores: mean slope, roughness, and hazard fraction. There is no sigmoid, softmax, or class-index output. The report’s phrase “predicted-safe probability” would be inaccurate; the saved deterministic safe decision is the only binary output.
- Fold-fitted target scaling is `log1p` then train-only z-score. Predictions are inverse-transformed with `expm1`, physical clamps (slope/roughness ≥0, hazard in [0,1]), then safety is derived as `(slope < 8°) AND (hazard_fraction < 0.05)`. The safe class is label 1. Recomputing this exact predicate from every saved prediction reproduces every stored `pred_safe`; confusion matrices recomputed from the saved values match the existing tables. There is no threshold or class inversion defect.
- For diagnostic sweeps only, `risk = max(predicted_slope/8, predicted_hazard_fraction/0.05)` and safe is `risk < threshold`. The default decision is equivalent to `risk < 1` because inverse-transformed slope and hazard are nonnegative. Coverage is the share with risk below the stated threshold. Risk is a threshold score, **not a probability**.
- Metadata safety labels match the same strict inequalities and source terrain target construction. The five test sets are paired: identical IDs/labels per model, no missing outputs, total 15,233 tiles. Site ID is used for grouped partitioning, not supplied as a model feature. Fold-specific I/F bounds and target scalers use training data only; no test labels enter fitting or checkpoint choice.
- The DTM labels/GDAL expose float32 elevation in meters with scale 1 and offset 0 in the inspected products. Thus missing DTM scale/offset application is not supported as the cause. Existing slope/roughness targets are generated in native-DTM physical coordinates; no error was found in the inverse target scaler from checkpoint replay.

Relevant implementation: [dataset.py](/D:/Mars-QCNN/src/data/dataset.py), [train.py](/D:/Mars-QCNN/src/train.py), [controlled_evaluation.py](/D:/Mars-QCNN/src/controlled_evaluation.py), [preprocess_real.py](/D:/Mars-QCNN/src/data/preprocess_real.py).

## 2. Site shift and input-mask evidence

All retained sites have mean joint valid coverage between **99.948% and 99.989%**; DTM and image valid coverage are each above 99.94% on average. Therefore missing-data coverage is not the explanation for the model-wide collapse. The image-valid mask stays available separately from intensity, which is why the below-range valid site_04 image becomes black rather than being confused with a missing-data mask.

| Site | Safe prevalence | Median slope (°) | Median roughness (m) | Median hazard | Mean valid coverage | Fold test I/F below low bound | Above high bound |
|---|---:|---:|---:|---:|---:|---:|---:|
| site_01 | 9.66% | 21.44 | 1.10 | 0.991 | 99.982% | 0.00% | 0.00% |
| site_02 | 39.79% | 8.77 | 1.00 | 0.016 | 99.969% | 0.00% | 0.00% |
| site_03 | 32.53% | 8.59 | 2.42 | 0.181 | 99.986% | 0.00% | **57.96%** |
| site_04 | 15.42% | 14.60 | 1.54 | 0.430 | 99.948% | **100.00%** | 0.00% |
| site_05 | 64.04% | 5.82 | 1.28 | 0.015 | 99.989% | 0.00% | 0.00% |

The values in the clipping columns are exact counts over valid stored image pixels, applying the precise per-fold bounds and `np.clip` logic used by the dataset. The listed I/F quantiles in the CSV are a reproducible sample of 64 valid pixels per patch (fixed RNG seed); clipping fractions are not sampled. All products are the selected RED filter; Coprates uses verified RED band 2 while the other products use RED band 1. Product-specific radiometric scales/offsets were applied into I/F, but differing acquisition, filter response, illumination, and surface reflectance remain potential domain differences. The current evidence does not establish that radiometrically calibrated I/F is distributionally interchangeable across observations.

The fold-level label mix is also uneven between training, validation and held-out site:

| Test fold/site | Train safe fraction | Validation site / safe fraction | Test safe fraction |
|---|---:|---|---:|
| 0 / site_01 | 52.76% | site_02 / 39.79% | 9.66% |
| 1 / site_02 | 48.67% | site_03 / 32.53% | 39.79% |
| 2 / site_03 | 50.02% | site_04 / 15.42% | 32.53% |
| 3 / site_04 | 26.97% | site_05 / 64.04% | 15.42% |
| 4 / site_05 | 31.96% | site_01 / 9.66% | 64.04% |

This is not label-based sampling or leakage; it is a small, heterogeneous set of whole sites. But it means the one validation site per fold can be very unlike the corresponding held-out site.

**Site_04 normalization evidence:** fold 3 training-only range is [0.07269287, 0.18676758] I/F. Site_04’s sampled valid-pixel q01/median/q99 are [0.03241, 0.04791, 0.06281], and its exact maximum is below the lower bound: all valid image pixels clip to 0. CNN risk has SD 0.0162 (median 3.1322); QCNN risk SD is 0.0529 (median 4.1166). The QCNN embedding SD across coordinates is 0.0056, and bottleneck-angle per-coordinate sample SD averages 0.0036. This is direct evidence of image/representation collapse under that fold’s preprocessing.

**Site_03 normalization evidence:** fold 2 training-only range is [0.07482910, 0.16125488] I/F. Its valid image median is 0.16504 and 57.9647% of valid pixels clip to 1. CNN risk median/SD are 9.5243/1.0545; QCNN risk median/SD are 10.7723/0.1847. This is substantial loss of bright-terrain image contrast.

**Site_05 evidence:** its fold-4 bounds are [0.04016113, 0.1875], and its raw I/F q01/median/q99 are [0.09131, 0.10309, 0.11316]; no valid pixels clip. However, the fold-4 training safe prevalence is 31.96%, validation site_01 is 9.66%, and test site_05 is 64.04%. Training vs test target means are slope 10.93° vs 6.46°, roughness 2.12 m vs 1.48 m, and hazard fraction 0.263 vs 0.0737. Among the 2,765 actually unsafe site_05 tiles, 176 fail slope only, 694 hazard only, and 1,895 both; the 4,925 safe tiles satisfy both criteria.

For site_05, CNN predictions barely separate the true classes: actual-unsafe vs actual-safe mean predicted slope is **6.240° vs 6.181°**, hazard **0.0305 vs 0.0281**, risk median **0.7758 vs 0.7656**. It predicts 99.75% of unsafe and 99.90% of safe tiles safe. QCNN does the opposite with nearly equal scores across classes: unsafe vs safe mean slope **9.767° vs 9.754°**, hazard **0.1236 vs 0.1209**, risk median **2.4547 vs 2.4022**. Its learned outputs are unsafe-like for both classes, yielding zero safe predictions. Thus prevalence/target shift alone is insufficient to explain the divergent models; weak within-site discrimination and different learned mappings are directly visible.

## 3. Scores, coverage sweep, and calibration transfer

Saved scores are nonconstant and finite, but several site/model risk distributions are tightly clustered. The complete min/max/mean/SD/q01/q05/q25/median/q75/q95/q99 tables are split by model, site, and ground-truth class in `score_distributions_by_site_and_label.csv`. Safe coverage and confusion counts across the fixed, label-blind threshold grid are in `risk_threshold_sweep.csv`; no threshold was selected from held-out labels.

At risk threshold 1, predicted-safe coverage is CNN: site_01 0.27%, site_02 0%, site_03 0%, site_04 0%, site_05 99.84%; CNN→QCNN: 0.82%, 0.95%, 0%, 0%, 0%. Median risk by site is CNN [2.496, 6.237, 9.524, 3.132, 0.769] and QCNN [2.601, 4.593, 10.772, 4.117, 2.417]. This explains why a single fixed cut produces both all-safe and all-unsafe site behavior.

The existing validation calibration independently selected a risk threshold per model/fold on the single validation site to target FPR ≤5%. Those thresholds are site/model-specific, not a globally calibrated probability cutoff. For fold 4, CNN chose 0.864 on site_01 validation (FPR 4.99%, coverage 4.51%) and then on site_05 test had FPR **97.65%**, safe recall 99.39%, coverage 98.76%. QCNN chose 1.973 on site_01 validation (FPR 4.99%, coverage 4.78%); on site_05 test its FPR was 0.036%, safe recall 0%, coverage 0.013%. The opposite test outcomes arise because the regressors’ within-site score distributions and target mapping do not transfer, not because the `<` comparator is reversed. Applying a post hoc site_05 threshold would use test information and was not done.

## 4. QCNN path and checkpoint audit

- The model output interface is continuous regression: the QCNN’s three Pauli expectation values are stacked, passed through learned `out_scale`/`out_bias`, then treated as standardized terrain regression outputs. There is no softmax. The 8-d bottleneck is `Linear(512,8)` followed by `tanh` mapped to angles [0,π]. The circuit readout is two Pauli-Z expectations plus their product, for the three target dimensions.
- Cached embeddings are produced by the corresponding fold’s selected CNN best checkpoint in eval mode with the same fold-normalized image input. QCNN inference on 24 deterministic fold-4 tiles replayed from its matching best checkpoint and cached embeddings agrees with the saved output to max absolute differences [9.6e-7 slope, 3.6e-7 roughness, 8.2e-8 hazard]; CNN replay is also within 7.5e-4 / 2.0e-4 / 2.7e-5. Derived safe classes match. This rules out a fold-4 checkpoint/prediction mismatch; the saved completion metadata also records identical model test/validation IDs.
- QCNN’s best validation checkpoint was epoch 0 for folds 0, 1, and 4, and epoch 3 for folds 2 and 3 (epochs are zero-indexed). Its saved last-checkpoint Adam states contain nonzero first/second moments for bottleneck, convolution, pooling, and readout parameters, so the gradient path and optimizer updates were active. A dead circuit or frozen head is not supported. However, best-epoch-at-start in three folds and near-constant site_03/04/05 scores support inadequate learning/discriminative capacity or an embedding/head mismatch as hypotheses. The existing run is too small (8 epoch cap, one seed) to distinguish them.
- QCNN training uses ordinary MSE on the three standardized regression targets, no class weighting, no safe-class sampling, and no safety-specific loss. It therefore has no objective that forces high safe recall; if its conditional terrain predictions sit above either safety cutoff, the deterministic rule labels every tile unsafe. The training splits are not uniformly safe-majority: safe fractions range 26.97%–52.76%; fold 4 is 31.96% safe. Class imbalance is a plausible influence but not a confirmed sole cause.
- Recovering the three circuit expectation inputs from the inverse-scaled saved QCNN scores and the learned affine readout gives values within [-0.856, 0.971], consistent with valid Pauli expectations in [-1,1]. No output overflow or anomalous probability transform is indicated. Direct un-affined expectation traces were not persisted, so this is an inverse recovery from saved regression outputs; the per-site latent/readout summaries are labeled accordingly.

## 5. CNN input stem implementation defect

The CNN config has `input_channels: 3` and `pretrained: false`. `Model1ResNet18` creates a three-channel random ResNet stem, then applies the pretrained-grayscale adaptation path: it zeros all new `conv1` weights and copies the average of the old random RGB filters into channel 0 only. Channels 1 and 2 (DTM-valid and image-valid coverage) start exactly zero rather than with normal three-channel random initialization. They are trainable and did acquire nonzero weights, so this is not a permanently disabled-channel bug; final channel L2 norms are still much smaller than channel 0 (roughly 0.07–0.17 vs 0.83–0.87 in the saved fold checkpoints). This is a **confirmed initialization defect** in the non-pretrained, 3-channel path. Its contribution to test collapse is unquantified and is not a sole explanation.

## 6. Prioritized causes

| Classification | Finding | Evidence / interpretation | Smallest corrective experiment (not run) |
|---|---|---|---|
| **Confirmed defect** | Train-only 2–98% image bounds are clipped to [0,1] without an out-of-range diagnostic/guard. | Site_04 has exactly 100% low clipping; site_03 57.9647% high clipping; masks remain valid. | Before training, compare a fixed, physically documented I/F transform that preserves values outside train quantiles (e.g. non-clipping affine plus explicit OOD flag) on stored patches; verify per-site image variance and clipping counts. Do not fit bounds on test sites. |
| **Confirmed defect** | Non-pretrained 3-channel ResNet input stem is initialized by a pretrained grayscale adaptation that zeros channels 1/2. | Code path and saved stem norms confirm the initialization; training later updates them but far less than channel 0. | Add an isolated initialization check and compare standard 3-channel Kaiming initialization in one controlled fold after review; keep all else fixed. |
| **Evidence-supported hypothesis** | Large site-level covariate and target shift; validation site is not representative of held-out site. | Site_05 prevalence 64.04% vs fold-4 training 31.96% and validation 9.66%; site_03/04 input clipping and target distributions also differ. | After fixing input handling, run a no-training site-wise feature/target overlap audit; then validate with multiple training-only calibration sites or a nested site protocol. |
| **Evidence-supported hypothesis** | QCNN frozen embedding + small 8-d bottleneck/quantum regressor underfits or collapses within-site class cues. | Outputs vary very little on sites 03–05; best checkpoint is epoch 0 in three folds, though gradients are active. | On the same saved CNN embeddings and split, first compare the existing QCNN to a linear and small MLP head using identical MSE/checkpoint rules; this isolates head capacity. Do not tune against test labels. |
| **Evidence-supported hypothesis** | MSE terrain regression is misaligned with safety recall; continuous estimates near/above cutoffs yield unsafe decisions despite useful regression behavior. | No classifier probability/loss exists; QCNN slope/hazard estimates often remain above 8°/0.05. | Report regression residuals and safety operating curves on validation only; later define safety acceptance criteria before any new training. |
| **Unresolved hypothesis** | Remaining cross-site differences reflect real I/F spectral/illumination/terrain appearance or product pairing beyond radiometric calibration. | All selected filters are labeled RED (Coprates band 2, others band 1), but products/observations differ and site_05 does not clip. | Review product-specific PDS/radiometric metadata and representative chips; compare a physically normalized image feature audit before modeling. |
| **Not supported as a cause** | Class inversion, incorrect binary comparator, broken QCNN gradients, DTM scale/offset omission, explicit site-ID leakage, or fold-4 checkpoint mismatch. | Predicate replay, label recount, optimizer moments, GDAL/PDS scale=1/offset=0, grouped splits, no site feature, and checkpoint replay support the contrary. | No corrective model experiment indicated; retain the checks as regression tests. |

## Recommendation / stopping point

**Next step: data/pipeline correction first, followed by a no-training verification—not calibration tuning or another training run yet.** The clipping destroys all image intensity information at site_04 and most bright-side detail at site_03. Correct the normalization design and the non-pretrained CNN stem initialization in a reviewed isolated change; verify transformed pixel distributions, masks, and channel initialization across all five sites. Then run a small inference/gradient smoke only after that review. Once inputs pass, run a new multi-seed site-held-out comparison with matched classical head controls and validation-only calibration. Do not use a threshold selected from these held-out labels. Existing production preprocessing, configs, model code, dataset, checkpoints, predictions, and reports remain unchanged.

## Reproducibility and artifacts

Run the read-only recomputation from the project root:

```powershell
.venv-gpu\Scripts\python.exe runs\prediction_diagnosis\reproduce_core_checks.py
```

It prints exact per-site clipping fractions and recomputed fixed-threshold confusion matrices/risk quantiles from stored shards and predictions; it writes no files. Supporting tables:

- `site_target_and_coverage_distributions.csv` — safe prevalence, terrain target quantiles, validity, selected filter/band, calibration scale/offset.
- `input_normalization_by_site.csv` — per-fold bounds, sampled calibrated-I/F quantiles, exact low/high clipping fractions, coverage.
- `score_distributions_by_site_and_label.csv` — min/max/mean/SD and q01/q05/q25/median/q75/q95/q99 for each saved regression score and derived risk, split by actual safe label.
- `risk_threshold_sweep.csv` — coverage and confusion/recall/FPR/FDR at fixed risk thresholds; no test-based selection.
- `prediction_site_summary.csv` — fixed-threshold direct-recomputed site confusion and risk summaries.
- `qcnn_intermediates_by_site.csv` — cached embedding/bottleneck/angle/readout summaries, selected epoch, and optimizer-state evidence.
- `representative_input_tiles.png` and `preview_patch_manifest.csv` — one terrain-median representative unsafe/safe patch per site, displayed after that site’s fold-specific normalization. The black Coprates rows and bright clipped site_03 preview are visible diagnostics.
- `input_and_qcnn_intermediate_diagnostics.csv` — original combined intermediate table; separate normalized views above are provided for easier reading.
