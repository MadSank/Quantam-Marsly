# LROCNet lunar CNN versus hybrid CNN–QCNN

The code lives in `experiments/moon/`; the dataset and all run artifacts stay under the gitignored `data/raw/moon/lrocnet/`. It does not modify Mars source, configs, data, or results (it imports the shared `src/quantum/circuits.py::BatchedQCNNHead`). It classifies the supplied folder labels (`ejecta=0`, `oldcrater=1`, `none=2`); it is not a landing-safety assessment. The supplied train/validation/test split is retained. Its generation procedure and geographic independence are not known; filename-prefix group separation is not proof of site independence.

## Models and comparison

- **CNN:** grayscale input (the audit found identical RGB channels), values scaled to `[0,1]`, three convolution blocks, 64-dimensional pooled embedding, and a three-logit linear classifier. AdamW, inverse-frequency class weights fitted from training labels only, validation macro-F1 checkpoint selection, patience 3, maximum 12 epochs.
- **Hybrid CNN–QCNN:** uses the per-seed CNN encoder selected with train/validation only, then freezes that encoder. A learned `64→8` bottleneck maps through `tanh` to angle features in `[0,π]`; the existing PennyLane `BatchedQCNNHead` runs an 8-qubit `default.qubit` circuit (angle encoding, convolution/pooling schedule 8→4→2, three shared convolution parameter groups, two pooling groups). It measures `Z`, `Z`, and `ZZ` on the two remaining wires; the head's learned affine readout yields three logits. The quantum/head parameters are trained with PyTorch gradients, and a nonzero gradient is asserted before fitting.
- **Frozen-embedding classical control:** same frozen CNN embeddings and 8-value bottleneck dimensions, with a 64→8→3 MLP (547 trainable parameters), to distinguish a quantum contribution from simply using a frozen encoder. The hybrid head has 8 bottleneck parameters plus the circuit and affine readout parameters. The CNN is end-to-end trainable; hybrid/control are frozen-encoder readout comparisons, so CNN-versus-hybrid changes both architecture and optimization regime.
- **Majority baseline:** predicts the most common training class (computed from train labels only).

All model selection uses validation macro-F1. Test images are not instantiated by `--pilot`; the full command evaluates the supplied test split once after each seed's choices are fixed. Test metrics include accuracy, balanced accuracy, macro/per-class precision/recall/F1 and confusion matrices. Simulator timing is reported separately from GPU CNN inference; no quantum speedup is claimed. Multiple seeds are configured (42, 43, 44). The saved predictions include identical ordered sample paths and labels for all model variants.

## Environment and commands

Verified with Python 3.11.15, PyTorch 2.9.1+cu128, torchvision 0.24.1+cu128, PennyLane installed, CUDA 12.8, NVIDIA RTX 4070 Laptop GPU. No packages were installed. From `D:\Mars-QCNN` in PowerShell:

```powershell
.\.venv-gpu\Scripts\python.exe experiments\moon\train.py --pilot
.\.venv-gpu\Scripts\python.exe experiments\moon\train.py
```

The pilot is train/validation only, class-balanced, bounded to 192 training and 96 validation images, and never opens test. Full outputs are placed in the configured `runs` directory (`data/raw/moon/lrocnet/experiment_lunar_cnn_qcnn/runs`; paths in `config.yaml` are relative to the repository root); the program refuses to overwrite it. Pilot outputs go to a timestamped `pilot_*` folder next to it. For an interrupted run, `--resume` continues from per-stage, per-epoch model/optimizer checkpoints and incrementally appended JSONL epoch logs. It writes per-seed JSON and CSV histories, metrics, sample IDs, labels, logits/predictions and checkpoints. Do not start a new run into an existing output directory; use a separate config/output path for a repeat.

## Completed run

Seeds 42/43/44 completed in 334 seconds (5.6 minutes) on the RTX 4070 Laptop GPU. Fitting times by seed were CNN 29.6/30.2/30.3 s and QCNN head 53.7/77.0/85.0 s. CNN test inference including its image encoder measured about 0.82–0.86 ms/image; QCNN-head inference measured about 0.78–0.82 ms/image on already computed frozen embeddings, so these numbers are not directly comparable and hybrid end-to-end inference also requires the encoder. Peak VRAM was not captured by the training process. The earlier pilot-based estimate of 12–20 minutes was conservative; measured runtime is the authoritative figure for this run.

Independent metric recomputation from saved logits and labels is in `runs/results_verification/results_audit.md` and `results_audit.json`. Mean ± sample SD over seeds on the supplied 779-image test partition:

| Model | Accuracy | Balanced accuracy | Macro-F1 |
|---|---:|---:|---:|
| CNN | 0.530 ± 0.035 | 0.558 ± 0.053 | 0.465 ± 0.024 |
| Hybrid CNN–QCNN | 0.529 ± 0.039 | 0.608 ± 0.006 | 0.497 ± 0.026 |
| Frozen-embedding MLP control | 0.451 ± 0.037 | 0.587 ± 0.021 | 0.440 ± 0.028 |
| Train-majority baseline | 0.684 ± 0.000 | 0.333 ± 0.000 | 0.271 ± 0.000 |

The hybrid has higher mean balanced accuracy and macro-F1 than the end-to-end CNN in this run, while accuracy is similar; three seeds on one supplied split do not establish a general advantage. The CNN and hybrid use different training regimes (end-to-end versus frozen encoder), the test split's geographic independence is unverified, and the majority baseline's greater accuracy reflects the dominant `none` class. Do not claim quantum speedup or interpret these labels as terrain/landing safety.

The completed 334-second run was performed before per-epoch recovery checkpoints were added to the trainer, so its artifacts contain final per-seed weights and histories, not optimizer-state resume snapshots. The current trainer writes per-epoch stage checkpoints (including optimizer and RNG states) and JSONL logs and supports `--resume` for a future run; this updated recovery path was smoke-tested against an uninterrupted synthetic run. Test dataset construction and test embedding generation have been moved to the final per-seed evaluation block in the current source so future runs keep test inputs locked through fitting and validation selection.

Run the read-only saved-result check with:

```powershell
.\.venv-gpu\Scripts\python.exe experiments\moon\verify_results.py
```

It refuses to overwrite its `runs/results_verification` directory.

## Limitations

Class definitions/annotation criteria and the original split-generation method remain undocumented in the local provenance audit. Candidate filename prefixes are unverified. Results describe only the supplied split and do not establish source/site generalization. CNN versus hybrid is not architecture-only because the hybrid encoder is frozen after CNN validation selection. This is a modest coursework simulator experiment on crater/ejecta labels, not direct safety classification; simulator runtime is not evidence of hardware quantum advantage.

## Location and credits

This folder holds the code that was originally written under `data/raw/moon/lrocnet/experiment_lunar_cnn_qcnn/` (`train.py`, `smoke_test.py`, `verify_results.py`, `config.yaml`, this README) and `data/raw/moon/lrocnet/` (`audit/audit_lrocnet.py`, `audit/provenance_audit.py`). It was moved here on 2026-10-27 so it is version-controlled. The only edits were path changes so it runs from the new location. `FINAL_READINESS_AUDIT.md` still cites the old paths, because it is a record of that time.

All code here was written for this project. None of it is derived from code distributed with the LROCNet dataset; the eight dataset archives contain only JPEG images and macOS metadata files.

The images, class labels and train/validation/test split come from the LROCNet lunar crater dataset. Its licence and citation are still to be confirmed (tracked in `website/PLAN.md`, build step 10).


## Code history

**What changed after the reported run.**
- The reported run (seeds 42/43/44) wrote its checkpoints between 11:50 and 11:55 on 2026-10-20, and its `summary.json` and `metrics.csv` at 11:55.
- `train.py` was last modified at 11:59 and this README at 12:00, so the current `train.py` is **not** the exact source that produced the reported numbers.
- That earlier source was never under version control, and no copy survives. What changed is reconstructed from three pieces of evidence:
  1. This README's earlier text and `FINAL_READINESS_AUDIT.md` §2–3.
  2. The run's own artifacts.
  3. The current source.

| Behaviour | Reported run (old trainer) | Current `train.py` | Evidence |
|---|---|---|---|
| When test data is touched | The test dataset was built, and test images passed through the frozen CNN encoder to get embeddings, **before** the hybrid and MLP heads were fitted. | `test_ds` stays unset until a seed's CNN, hybrid and MLP are all fitted and selected. Only then are test images loaded, embedded and scored. | Audit §2 (Moon) and §3.5; the README text of the time; the current code (`test_ds=None`, the final evaluation block). |
| Resuming an interrupted run | Final per-seed weights only. | Per-epoch stage checkpoints with optimizer and RNG state, JSONL epoch logs, and `--resume`. | The run has no `*_epochs.jsonl` or `*_training.pt` files. |
| Recorded parameter counts | One `cnn` entry, counted after the encoder was frozen (`trainable: 195`). | `cnn_during_fit` (59,347) and `cnn_classifier_after_encoder_freeze` (195), recorded separately. | `runs/logs/seed_*/results.json`, compared with the current code. |

**Could test data have influenced training, early stopping, checkpoint selection or calibration?** The answer is **can't tell with certainty**, because the old source is gone. The surviving evidence all points to no:

- **Gradient updates:** class weights come from the training labels only: `class_counts_train` is `[358, 594, 2656]`, which matches the training folders. No logged quantity involves test data. The audit states the test embeddings "were not used for gradient updates, validation selection, or threshold selection". That statement can't be checked against the old source now.
- **Early stopping:** in all 9 fits (3 models × 3 seeds), training stopped exactly 3 epochs after the epoch with the best **validation** macro-F1. That matches the validation-only patience rule (patience 3) recorded in `history.csv`. A stopping rule driven by test data would be unlikely to coincide with it in every fit.
- **Checkpoint selection:** the logged selection metric is validation macro-F1. Whether the saved weights are the best-validation epoch hasn't been re-verified from the checkpoints (see the re-checks below).
- **Calibration:** none exists. Predictions are the argmax of the logits, and there's no threshold or temperature. The saved predictions equal the argmax of the saved logits (`runs/results_verification/results_audit.md`).
- **The one known difference:** test embeddings were computed early. On its own, that can't change any result, as long as the encoder was frozen and in evaluation mode (no BatchNorm statistics updated) while embedding. Whether the old code embedded test data in evaluation mode can't be confirmed without the source.

**Are the published Moon numbers valid as stated?** The arithmetic is verified: every reported metric was independently recomputed from the saved logits and labels. Two re-checks were planned to test whether the results are free of test influence; both were run on 2026-10-27 (results below).

1. **Read-only checkpoint replay** (CPU, a few minutes, no training).
   - Load each seed's `cnn`, `hybrid` and `frozen_mlp` checkpoints.
   - Recompute the test logits; they must match the saved logits within floating-point tolerance. This shows the reported predictions come from the saved weights, with no BatchNorm change after saving.
   - Recompute each model's validation macro-F1; it must equal the best `val_macro_f1` for that fit in `history.csv`. This shows the saved weights are the validation-selected epochs.
   - Limit: this does not rule out test influence during training itself.
2. **Locked-test re-run** (decisive; about 6 minutes on the RTX 4070).
   - Run the current `train.py`, which by construction keeps test data locked until the end, for seeds 42/43/44 into a **new** output directory.
   - Compare its metrics with the reported ones. Identical numbers settle the question.
   - Any difference should be judged against the seed-to-seed spread, because GPU and simulator nondeterminism can cause small changes.

### Re-check results (2026-10-27)

No published number, export or test was changed because of these results.

**1. Checkpoint replay** (`replay_checkpoints.py`, CPU, read-only): **passed** for all 9 model × seed fits.
- Recomputed test predictions are identical to the saved ones. Logits differ by at most 4.5 × 10⁻⁴, which is GPU-vs-CPU floating-point noise.
- Recomputed validation macro-F1 equals the best logged epoch exactly.
- So the reported predictions come from the saved, validation-selected weights.

**2. Locked-test re-run** (current `train.py`, config `configs/rerun_locked_test.yaml`, output `data/raw/moon/lrocnet/experiment_lunar_cnn_qcnn/rerun_locked_test_20261027/`, same GPU and environment). The existing `runs/` was not touched.
- **CNN:** bit-identical to the published run for all three seeds: the same logits, epochs and metrics. CUDA training here was therefore deterministic. (I had expected small differences from non-deterministic CUDA kernels; there were none.)
- **Hybrid QCNN head and frozen-embedding MLP:** **not** reproduced. Both heads are trained on CPU from the identical frozen CNN embeddings, yet they already differ before training: the initial QCNN gradient norm is 0.198 / 0.378 / 0.209 against the published 0.413 / 0.523 / 0.381. Their training then takes different numbers of epochs.

Mean ± sample SD over seeds 42/43/44 on the same 779-image test set:

| Model | Metric | Published | Re-run | Difference | Re-run mean inside published seed range? |
|---|---|---:|---:|---:|---|
| CNN | accuracy | 0.530 ± 0.035 | 0.530 ± 0.035 | 0 | yes (identical) |
| CNN | balanced accuracy | 0.558 ± 0.053 | 0.558 ± 0.053 | 0 | yes (identical) |
| CNN | macro-F1 | 0.465 ± 0.024 | 0.465 ± 0.024 | 0 | yes (identical) |
| Hybrid CNN–QCNN | accuracy | 0.529 ± 0.039 | 0.451 ± 0.041 | −0.078 | **no** (published range 0.488–0.565; difference is 2.0 SD) |
| Hybrid CNN–QCNN | balanced accuracy | 0.608 ± 0.006 | 0.587 ± 0.032 | −0.021 | **no** (published range 0.604–0.615) |
| Hybrid CNN–QCNN | macro-F1 | 0.497 ± 0.026 | 0.439 ± 0.035 | −0.058 | **no** (published range 0.469–0.519; difference is 2.2 SD) |
| Frozen-embedding MLP | accuracy | 0.451 ± 0.037 | 0.502 ± 0.026 | +0.051 | no (published range 0.409–0.481; within 2 SD) |
| Frozen-embedding MLP | balanced accuracy | 0.587 ± 0.021 | 0.606 ± 0.022 | +0.019 | yes |
| Frozen-embedding MLP | macro-F1 | 0.440 ± 0.028 | 0.479 ± 0.019 | +0.039 | no (published range 0.408–0.463; within 2 SD) |
| Majority baseline | all three | 0.684 / 0.333 / 0.271 | identical | 0 | yes |

**What this means.**
- **Most likely cause:** the CNN and its embeddings are identical, so the head results differ because the random-number state at the moment the heads are initialised differs. A `torch.utils.data.DataLoader` draws one value from PyTorch's global random generator every time it is iterated, even without shuffling. The old trainer iterated test-data loaders before fitting the heads (the change described above). That would shift the generator, and with it the head initialisation and the shuffling order.
- **This is a random-state difference, not a data leak.** The numbers drawn are random, not test data. It is consistent with the replay result and with the absence of any test influence on training or selection. It has not yet been confirmed directly.
- **The published hybrid and MLP figures are valid for the run that produced them** (re-check 1). They are not robust to an equivalent re-initialisation, and three seeds understate the variability of the frozen-embedding heads.
- **The relative ordering is not stable.** Published balanced accuracy: hybrid 0.608 > MLP 0.587 > CNN 0.558. In the re-run: MLP 0.606 > hybrid 0.587 > CNN 0.558. On macro-F1 the re-run hybrid (0.439) falls below the CNN (0.465). Any statement that the hybrid head scores higher than the CNN or the MLP control is therefore not supported beyond the original three seeds.

**Not yet done:**
1. Confirm the random-state explanation. Re-create the old order by iterating the test loaders the same number of times before head initialisation, and check that the published initial gradient norms (0.4127 / 0.5228 / 0.3807) and head metrics are reproduced exactly.
2. Decide whether the published Moon figures, the website caveats and `FINAL_READINESS_AUDIT.md` need an added disclosure or more seeds. That decision belongs to the project owner; nothing has been changed.

The website shows this note next to the Moon code excerpts and keeps the audit's test-embedding caveat.

**Summary for readers of the results.** The hybrid QCNN head and frozen-embedding MLP results did not reproduce exactly in a locked-test re-run, most likely because of a different random initialisation, so the ranking between hybrid, MLP and CNN on the Moon should not be treated as robust.

### Deterministic trainer (2026-10-28)

`train.py` is now deterministic, so a future run can be repeated exactly. Changes:

- **Deterministic kernels:** `torch.use_deterministic_algorithms(True)` is on, with `CUBLAS_WORKSPACE_CONFIG=:4096:8` set before CUDA starts. The CNN's `nn.AdaptiveAvgPool2d(1)` is replaced by `GlobalMeanPool`, which gives the same output, has no parameters and keeps the same `state_dict`, but has a deterministic CUDA backward pass. The reported checkpoints still load.
- **Named random streams:** each model initialisation (`cnn_init`, `hybrid_init`, `mlp_init`) and each data loader gets its own seed, derived from (seed, stream name).
  - Initialisation runs inside `torch.random.fork_rng`.
  - Every `DataLoader` owns a generator, so iterating one loader (for example a test loader) can no longer shift anything else. This removes the cause identified above.
- **Matched head batches:** the hybrid and MLP heads read the same `head_train` / `head_val` streams through separate loaders, so both see the same batch order. Before, the MLP reused the hybrid's loader, so its order depended on how long the hybrid trained.
- **The gradient check** draws its batch from its own stream, so it no longer advances the training loader.
- **`summary.json` records** the determinism settings, library versions, a SHA-256 of `train.py` and every stream seed.

Supporting tools:
- `tests/test_determinism.py` (8 tests) covers the mechanics. Initialisation is unaffected by earlier random draws and loader iterations. The global generator is restored after initialisation. Loader order depends only on its own stream. `GlobalMeanPool` equals `AdaptiveAvgPool2d(1)`, and the reported checkpoints still load.
- `compare_runs.py` compares two run folders bit for bit, excluding wall-clock timings.
- `verify_results.py` takes `--runs <folder>` and checks the seed list from that run's own config, instead of assuming three seeds.

Known limit: `--resume` restores the global generator but not the loaders' own generators, so a resumed run is not guaranteed bit-identical to an uninterrupted one. Run without `--resume` when exact reproduction matters.

**Check A, passed:** seed 42 was run twice with `configs/determinism_check_a.yaml` and `configs/determinism_check_b.yaml` (outputs `data/raw/.../determinism_check_{a,b}_20261028/`, 83 s each on the RTX 4070). `compare_runs.py` reports them **identical**: every logit, prediction, metric, loss and validation-F1 history, and every checkpoint tensor. As a control, it reports 131 differences against the earlier locked-test re-run.

**10-seed re-run: planned, not yet done.** `configs/deterministic_10seed.yaml` (seeds 42–51) and `configs/deterministic_repeat.yaml` (seeds 42 and 43, for a bit-for-bit repeat) are ready. The first attempt was stopped on request before it wrote any output. Expected time is about 14 minutes for 10 seeds plus 3 minutes for the repeat. Until it is run, the published three-seed Moon figures stand, with the caveat above.
