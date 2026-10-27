# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Two loosely coupled halves sharing one workspace:

1. **Research pipeline** (`src/`, `tests/`, `configs/`): classical CNN (ResNet-18) vs. hybrid quantum-classical CNN (PennyLane) for predicting Mars terrain proxies (5 m mean slope, roughness, hazard fraction) from HiRISE orthoimage/DTM patches. Package name `mars-qcnn-terrain`; run modules as `python -m src.…` from the repo root.
2. **Visualization app** (`visualization_app/`): a Streamlit "Mission Console" dashboard that reads **pre-computed** Mars and Moon (LROCNet crater classification) experiment artifacts. It never trains models and has no torch/PennyLane dependency. The current branch (`website-redesign`) is focused on this app.

## Environments (Windows; three separate venvs)

- `.venv` — pipeline (CPU). `pip install -e .` from root.
- `.venv-gpu` — pipeline with GPU torch.
- `visualization_app/.venv` — app only; install from `visualization_app/requirements.txt`.

Do not mix them: the app venv lacks torch, the pipeline venvs lack streamlit/plotly.

## Commands

Pipeline (from repo root, `.venv` active):
```bash
pytest                                            # all pipeline tests (testpaths=tests, -v --tb=short via pyproject)
pytest tests/test_splits_and_leakage.py::test_name  # single test
python -m src.train --config configs/default.yaml --smoke --all     # synthetic smoke training
python -m src.evaluate --config configs/default.yaml --smoke        # -> results/smoke/summary_table.md
python -m src.train --config <cfg> --model model2 --variant all --fold all --seed all --override key=value
python -m src.data.preprocess_real --config configs/real_data.yaml   # real HiRISE preprocessing
python -m src.controlled_evaluation --config <cfg> --mode {prepare,smoke,run,metrics} --folds all --seeds 42,43,44
```

App (from `visualization_app/`):
```bash
.\.venv\Scripts\streamlit run app.py              # http://localhost:8501
.\.venv\Scripts\pytest tests/ -v                  # acceptance tests (tests/test_m1..m7_*.py, one per milestone/page)
```
Set `MARS_QCNN_ROOT` to override the workspace root the app resolves artifacts from.

## Pipeline architecture

- `src/utils.py` loads YAML config + `--override key=value` CLI overrides, seeding, atomic saves. All paths (processed data, cache, runs, results) come from the config's `paths:` block, so each config writes to its own `runs/<name>` / `results/<name>` tree.
- `src/data/`: `preprocess.py` (original/synthetic) and `preprocess_real.py` (real PDS DTMs: nodata/seam masking, Horn slope at 5 m/15 m baselines, 64×64 tiles at 2 m/px) write float16 `.npy` shards + `metadata.parquet`. `dataset.py` holds `MarsTerrainDataset`, `TargetScaler` (log1p + z-score, **fit on training fold only**) and `derive_safe_decisions`. `splits.py` produces grouped folds; tests assert no group leakage.
- `src/models/__init__.py::build_model` is the factory for model1 (ResNet-18), model2 (frozen ResNet → 8-d bottleneck → 8-qubit QCNN, plus matched classical controls), model3 (quanvolution), model4 (amplitude QCNN). Quantum circuits live in `src/quantum/circuits.py` and must stay batched (parameter broadcasting, `backprop`), not per-sample loops — a test checks batch-vs-loop equivalence.
- `src/train.py::train_single_run` is resumable (RNG state capture/restore); per-run outputs go to `runs/.../<model>/fold_<k>_seed_<s>/` (`config_snapshot.yaml`, `epoch_log.csv`, `param_counts.json`, `test_predictions.parquet`, `validation_predictions.parquet`).
- `src/controlled_evaluation.py` is the leave-one-site-out driver (3 train / 1 val / 1 test site) that reuses `train_single_run`, tracks `progress.json`, and writes fixed-threshold and validation-calibrated safety metrics under `controlled_root`.

## App architecture

- `data_loading/paths.py` is the single source of artifact locations. The **primary Mars results** are `runs/pipeline_repair_audit/retraining_seedset/` (15 runs = 5 LOSO folds × seeds 42/43/44); Mars image shards come from `runs/controlled_evaluation/processed/`; Moon data is under `data/raw/moon/lrocnet/`. Much of this is gitignored (shards, `data/raw/`), so the app and its tests only work on a machine that has the artifacts.
- `app.py` + `pages/1_…6_*.py` use Streamlit native multipage routing. Shared UI lives in `components/` (`layout.page_header/section_header/note_block/kv_table`, charts, filters, image cards) and `style/theme.py` (`apply_theme()` CSS tokens `--mq-*`, `finalize_figure()` for Plotly). New pages/widgets should use these instead of ad-hoc styling; theme colors are mirrored in `.streamlit/config.toml`.
- Use `width="stretch"`, not the deprecated `use_container_width=True`.

## Scientific-integrity constraints (enforced by app tests)

`FINAL_READINESS_AUDIT.md` is authoritative; app tests check headline metrics match it exactly. When editing app text or numbers:
- Never alter predictions, metrics, labels or splits; the app only displays saved artifacts.
- "Safe" is a derived proxy (`mean slope < 8.0°` AND `hazard fraction < 0.05`), not an operational landing-safety system — keep that caveat.
- The Mars comparison is end-to-end CNN vs. a small head on a frozen supervised representation (pipeline comparison, not equal capacity). Paired t-intervals cross zero: no overall winner, no quantum advantage claimed (circuits are simulated).
- Moon: Classical CNN has 59,347 trainable params (not the legacy 195); CNN vs QCNN/MLP latencies are not comparable; test set is 68.4% `none`, so report balanced accuracy and macro-F1.
- Uncertainty is sample SD or nominal 5-site paired t-intervals.
- The paired-site statistical assessment section removed from Model Comparison Tab 1 must not be restored.

## Website

Website work follows website/PLAN.md.

## Reference docs

`README.md` (pipeline quickstart; its repo tree predates `controlled_evaluation.py`/`preprocess_real.py`), `visualization_app/README.md` and `PROJECT_HANDOFF.md` (app spec and artifact schemas), `UI_REDESIGN_PLAN.md` / `UI_REDESIGN_IMPLEMENTATION_REPORT.md` (design system), `audit_screenshots/` vs `redesign_screenshots/` (before/after at 1920×1080 and 1366×768).
