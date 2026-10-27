# Retraining Preflight — Read Only

**Proposed command audited (not executed):**

```powershell
.venv-gpu\Scripts\python.exe -m src.controlled_evaluation --config runs\pipeline_repair_audit\retraining_config.yaml --mode run --folds all --seeds 42,43,44
```

**Scope:** Configuration, path, cache, model, checkpoint, comparison, and CLI source review only. No training, cache generation, checkpoint load, prediction, or evaluation was launched. This report is the only new file written for this preflight.

## Results

| Check | Status | Evidence |
|---|---|---|
| 1. Repaired preprocessing and ResNet stem in all folds | **PASS** | `make_fold_config` refits normalization per fold from `splits["train"]` and adds `clip: false`; one resulting config is passed to train/validation/test datasets and to CNN embedding generation. The config specifies `pretrained: false`, `input_channels: 3`; repaired `make_resnet_input_stem` retains ResNet18's initialized three-channel stem. |
| 2. Cache validity / no defective embedding reuse | **PASS** | Processed shards are the existing calibrated I/F plus DTM/image-valid channels (`dataset_metadata.json` documents radiometric scaling and `sitewise_image_stretch: false`); fold affine normalization is applied later by the repaired dataset loader. They are appropriate to reuse. Embeddings are written to a new fold/seed cache path below the isolated root. That root, and therefore its cache directory, does not currently exist. `prepare_run_caches` therefore cannot load an old embedding file on this proposed first run; it will generate embeddings from that fold/seed's newly trained repaired CNN checkpoint. |
| 3. Per-fold/seed config and checkpoint isolation | **PASS** | CLI parses comma-separated seeds as integers, loops every fold × seed, and calls `make_fold_config` each time; it writes each fold/seed into `training.seeds` and `controlled_run`. The output root is currently absent, so there are no `last_checkpoint.pt`, `best_checkpoint.pt`, or prediction files to resume/load. |
| 4. Outputs do not overwrite prior runs | **PASS** | `controlled_root` is `D:/Mars-QCNN/runs/pipeline_repair_audit/retraining_seedset`, which is currently absent. `make_fold_config` routes run directories, configs, caches, and results/metrics beneath that root. Existing controlled artifacts under `runs/controlled_evaluation` are the read-only processed input only; new logs, checkpoints, predictions, protocol records, and metrics go to the isolated root. |
| 5. Identical test IDs and labels | **PASS** | Both models use the same processed metadata, `get_fold_indices`, fold and seed. `run_one` explicitly compares CNN and CNN→QCNN `tile_id` order and the saved safety and three terrain labels for both test and validation outputs, raising an assertion on any mismatch. |
| 6. Protocol and comparison limitations documented | **PASS** | Summary below. The design is controlled but is not an architecture-only comparison because the CNN trains end-to-end while the hybrid trains a bottleneck/QCNN on frozen CNN embeddings. |
| 7. Folds and seeds supported as written | **PASS** | `--folds all` expands to `[0,1,2,3,4]`; `--seeds 42,43,44` parses to three integers. The execution loops all 15 fold/seed combinations. |

## Protocol resolved from configuration and code

- **Fold design:** five grouped folds with split seed 42 and site order site_01 through site_05. For each test fold, the next fold is validation and the other three sites are training data. Normalization and target scaling are fitted on training rows only.
- **Preprocessing and input:** nominal 2 m/px, 64×64 tiles (128 m footprint), three channels (calibrated image, DTM-valid fraction, image-valid fraction), no sitewise image stretch, no photometric or geometric augmentation. The new fold normalization is training-only 2nd/98th percentile affine I/F with `clip: false`; image-invalid cells are reset to zero. Product-specific band mapping and radiometric scale/offset are already embodied in the stored calibrated-I/F patches.
- **CNN:** random, non-pretrained ResNet18, three input channels, 128-wide regression head, dropout 0.1. The corrected three-channel random stem is left intact. The CNN backbone and regression head are trained.
- **CNN→QCNN:** the run first trains the fold/seed CNN, then obtains embeddings using its validation-selected best checkpoint. Embeddings are cached separately per fold and seed, and the QCNN model consumes them as fixed inputs. The QCNN trainable path is an 8-dimensional bottleneck plus an 8-qubit angle-encoded QCNN readout; it does not fine-tune the CNN backbone. The PennyLane default.qubit execution is CPU-backed. Its regression target/loss path is the same three standardized terrain targets with MSE.
- **Optimization and selection:** both models use AdamW, learning rate `1e-4`, weight decay `1e-4`, batch size 16, maximum 8 epochs, and early stopping patience 3. The checkpoint with lowest validation MSE is retained. The test fold is evaluated after selection; source flow does not use test labels for checkpoint selection.
- **Safety decision/calibration:** fixed safe-label definition is mean slope `<8°` and hazard fraction `<0.05`. The validation FPR target is 5%; `collect_results` selects any reported risk threshold from validation predictions and then applies it to test predictions. The fixed safety thresholds are not changed by this command.
- **Comparison limitation:** data, split, targets, losses, and selection rules are aligned, and ID/label parity is asserted. This is still not an architecture-only test: the CNN is a fully trained image regressor; CNN→QCNN uses the CNN's frozen representation and trains a much smaller bottleneck/quantum head in a separate stage. A difference cannot be attributed solely to replacing a classical head with a quantum head.

## Important implementation caveats

- Resume behavior exists in `train_single_run`: if a run's `last_checkpoint.pt` exists, it loads that checkpoint and resumes; if the run is complete and predictions exist, it can return early. That is safe for this proposed **first** run because the configured output root is absent. A later rerun against a partially populated root would resume rather than guarantee a clean restart.
- The embedding cache is existence-based (`if emb_cache_path.exists(): np.load(...)`); it does not validate a model/config fingerprint. It cannot reuse a defective cache on this invocation because the isolated cache root is absent. Preserve that isolation: if the output root exists by launch time, inspect its contents and use a new empty output root for a fresh post-repair run rather than assuming cached embeddings are valid.
- Cached raw patches are intentionally reused. They are not normalized/clipped model inputs: their metadata records calibrated I/F and masks, and the repaired normalization is applied on dataset load. No patch regeneration is needed for the code repair being audited.

## Recommendation

**Preflight: PASS for the proposed first run in the current filesystem state.** All seven requested checks pass; no critical failure or unresolved item blocks the command. The output root is currently empty/nonexistent, which is essential to the no-stale-checkpoint/cache conclusion. Do not treat a rerun against a subsequently populated output root as a fresh run. This audit did not execute the command or start training.

## Reproducible read-only checks

From the workspace root, verify the output root remains absent before a future first run:

```powershell
Test-Path runs\pipeline_repair_audit\retraining_seedset
```

The CLI and output/cache/resume behavior can be reviewed at `src/controlled_evaluation.py` (`make_fold_config`, `run_one`, `main`) and `src/train.py` (`prepare_run_caches`, `train_single_run`). Do not run the proposed command as part of preflight.
