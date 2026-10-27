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

