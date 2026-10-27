# Mars-QCNN Post-Training Results Audit

**Audit date:** 2026-10-19  
**Scope:** read-only audit of the saved fold-0 CNN and CNN→QCNN runs. Metrics below were recomputed from the saved prediction parquet files; no training was launched and no prior artifact was changed.

## 1. Artifacts and test-set identity

- CNN checkpoint and run metadata: `runs/real_v1/model1_resnet18/fold_0_seed_42/{best_checkpoint.pt,last_checkpoint.pt,config_snapshot.yaml,epoch_log.csv,param_counts.json}`.
- CNN predictions and metric outputs: `runs/real_v1/model1_resnet18/fold_0_seed_42/test_predictions.parquet`; `results/real_v1/{summary.csv,summary_table.md,paired_comparison.csv,metric_bars.png,predicted_vs_measured.png}`.
- QCNN checkpoint and run metadata: `runs/real_v1_qcnn/model2_qcnn/fold_0_seed_42/{best_checkpoint.pt,last_checkpoint.pt,config_snapshot.yaml,epoch_log.csv,param_counts.json}`.
- QCNN predictions and metric outputs: `runs/real_v1_qcnn/model2_qcnn/fold_0_seed_42/test_predictions.parquet`; `results/real_v1_qcnn/{summary.csv,summary_table.md,paired_comparison.csv,metric_bars.png,predicted_vs_measured.png}`.
- Ground-truth feature/label table: `data/processed/real_v1/metadata.parquet`; row-level preprocessing provenance: `data/processed/real_v1/patch_manifest.csv`; dataset summary/splits: `data/processed/real_v1/dataset_metadata.json`. The targets are derived terrain proxies from the PDS DTM rasters, not observed landing outcomes. The exact ortho labels used are in `%TEMP%/MarsQCNN_PhaseB1/`; DTM labels are embedded in the `.IMG` products under `data/raw/`.

Both prediction files contain the same 962 rows in the same order, with exactly the same `tile_id`, `image_id`, `true_safe`, and three `true_*` values. Every `image_id` is `site_01` (Kaiser test site). The safe label distribution is 212 safe and 750 unsafe. Sample IDs are the full matching set in the `tile_id` column; they use the `site_01_r{row}_c{column}` convention. There are no missing or non-finite prediction or truth values in either file. Fold 0 uses sites 02/03/05 for training, site 04 (Coprates) for validation, and site 01 for test; counts are 4,500 / 901 / 962.

**Feature extractor and trainable parameters:** the CNN run trains all ResNet-18 backbone and regression-head parameters from random initialization (11,242,563 trainable; 0 frozen). The QCNN run consumes a saved `(6363, 512)` float32 embedding array at `data/cache/real_v1_qcnn/embeddings_fold_0_seed_42.npy`. Source code shows embeddings are generated in eval mode by `Model1ResNet18.extract_embeddings`; the QCNN model then trains its 512→8 bottleneck and QCNN/readout head (4,152 trainable parameters). The CNN backbone is therefore frozen operationally and is not part of the QCNN checkpoint/parameter count. The QCNN embeddings came from the CNN fold-0 best checkpoint copied to its cache-loader location; the files are present. The cache itself does not contain a checkpoint hash/provenance record.

## 2. Safety classification (positive class = safe)

The saved preprocessing code defines a true safe tile as `mean_slope_5m < 8° AND hazard_fraction < 0.05`. `hazard_fraction` is the fraction of valid 5 m-baseline slope samples exceeding 15°. Predicted-safe uses the same strict thresholds applied to inverse-transformed model slope and hazard predictions. Thus TP = safe correctly predicted, FP = unsafe predicted safe, TN = unsafe correctly predicted, FN = safe predicted unsafe.

| Model | TP | FP | TN | FN | Safe precision | Safe recall | Specificity | FPR among unsafe | FDR among predicted-safe | Predicted safe |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CNN | 67 | 245 | 505 | 145 | 21.47% | 31.60% | 67.33% | 32.67% | 78.53% | 312/962 (32.43%) |
| CNN→QCNN | 48 | 65 | 685 | 164 | 42.48% | 22.64% | 91.33% | 8.67% | 57.52% | 113/962 (11.75%) |

The existing `78.5%` and `57.5%` values are **false-discovery rates** (also called the existing code's `false_safe_rate`): `FP / (TP + FP)`, the fraction of model-predicted-safe tiles that are actually unsafe. They are not false-positive rates, whose denominator is all actually unsafe tiles (`FP / (FP + TN)`). Zero denominators would make the corresponding rate undefined; none of these denominators is zero.

## 3. Regression audit

Saved predictions are already in physical target units: the code fits log1p plus standardization using only the training split, then applies the inverse transform (including nonnegative/0–1 clipping) before writing predictions. Truth columns are unscaled source labels. The PDS DTM label states elevations in meters above the Mars 2000 equipotential surface, `SCALING_FACTOR=1`, `OFFSET=0`, with a missing constant; GDAL's float32/nodata view is consistent. Slope is in degrees, roughness is detrended RMS elevation residual in meters, and hazard fraction is dimensionless. The PDS labels and the saved Phase B.1 report at `%TEMP%/MarsQCNN_PhaseB1/outputs/Phase_B1_report.md` establish the corresponding DTM interpretation; no extra elevation conversion was applied in this audit.

| Target | Units / meaning | CNN RMSE | CNN MAE | CNN→QCNN RMSE | CNN→QCNN MAE |
|---|---|---:|---:|---:|---:|
| Mean 5 m-baseline slope | degrees | 14.0741 | 11.5108 | 14.7107 | 12.0230 |
| Roughness | meters | 0.9411 | 0.5923 | 0.9872 | 0.6696 |
| Hazard fraction | fraction [0,1] | 0.6462 | 0.5284 | 0.6422 | 0.5281 |

Each metric uses all 962 paired samples; no values were dropped. These recomputations agree with the existing one-run summaries to displayed precision.

## 4. Comparison fairness and selection

**Verified common protocol:** same processed metadata/shards, input representation (normalized image plus DTM-valid and image-valid mask channels), selected RED band at every site (Coprates band 2, per its PDS label), target definitions, safe thresholds, split seed 42/fold 0, random seed 42, batch size 16, AdamW at 1e-4 with weight decay 1e-4, and MSE on training-fold log1p-standardized targets. Both configs disable sun features and use the same nominal photometric augmentation settings. QCNN receives the same CNN feature backbone as the fold-0 CNN run, then freezes its embeddings.

**Validation-only selection verified:** `get_fold_indices` assigns fold 0 (site 01) as test and fold 1 (site 04) as validation; neither is in training. The training loop selects `best_checkpoint.pt` by validation loss, loads that checkpoint before test prediction, and only then writes test predictions. The CNN best checkpoint is epoch 4 (0-based epoch 3, validation loss 1.5725); QCNN best is epoch 5 (0-based epoch 4, validation loss 1.9714). The test site is not used for early stopping or checkpoint selection.

**Differences/confounds:**

1. These are not equal-capacity or equal-training-budget models: CNN trains 11.24M parameters end-to-end; QCNN trains a 4,152-parameter head on a CNN encoder trained separately for the same supervised target. The QCNN encoder's fitting cost/parameters are omitted from its run count. This is a hybrid transfer experiment, not a comparison of CNN and QCNN from equivalent initialization/training.
2. CNN allows 12 epochs/patience 4 and stops after 8; QCNN allows 6/patience 3 and completes 6. Their cosine schedules therefore differ.
3. Training augmentation is not actually matched. The embedding cache was made from unaugmented eval-mode images. During QCNN dataset loading, image photometric augmentation may run, but `BottleneckQCNNModel.forward` selects the cached `embedding` and ignores the altered image. The CNN uses the augmented image directly.
4. The CNN checkpoint resumed after three CPU epochs under PyTorch 2.14.1+cpu into a run logged as PyTorch 2.9.1+cu128 on CUDA. QCNN used PyTorch 2.9.1+cu128 on CPU. The optimizer/scheduler state was restored, but this cross-version/device continuation is an additional reproducibility confound. Checkpoints do not save/restore RNG states, so resumed shuffling and photometric augmentation did not continue the original random stream.
5. Per-site image stretch uses percentiles calculated over each whole product, including validation/test-site imagery. It uses no labels, but is transductive site-level input normalization and should be fixed/prespecified for a strict deployment-style test.
6. The available `paired_comparison.csv` files have no cross-model paired comparison: each evaluation was run separately and each results folder contains one model. No multi-seed or across-site uncertainty is present.

## 5. Limitations, next experiments, and decision

This is one held-out geographical site and one seed. Tile windows within one image are spatially related, so 962 tiles are not 962 independent site-level replications. Site-specific geology, coverage and radiometry can dominate the estimate. The proxy labels are not a validated landing-safety outcome. One test site cannot support a generalization claim or statistically defensible model ranking.

On this single site, CNN has lower slope and roughness RMSE and higher safe recall; QCNN has slightly lower hazard RMSE, higher safe precision and specificity, and fewer unsafe tiles declared safe. Neither dominates. The apparent QCNN false-safe improvement accompanies a much lower safe-prediction rate and safe recall. The current evidence supports **no overall performance advantage** for either model.

**Smallest defensible next comparison (proposal only):** first make the QCNN augmentation behavior explicit and pair it with a classical head/control on exactly the same frozen embeddings; align the epoch/early-stopping and model-selection budget; record encoder/checkpoint hashes and software/device versions. Then evaluate the paired models over all five site-held-out folds with identical split-specific preprocessing and validation-only selection, preferably at least three seeds. Report site-level results separately and avoid treating within-site tiles as independent replicates. Add geographically independent sites before claiming deployment generalization. No such experiment was run in this audit.

**Recommendation: NO-GO for further tuning or a performance claim from the current two runs. GO for a controlled, matched comparison phase after the protocol differences above are resolved and reviewed.**
