"""Leakage-checked all-site CNN versus frozen-backbone CNN→QCNN evaluation.

All outputs are written below runs/controlled_evaluation. Preprocess first with
the companion config's unlimited class cap and raw calibrated I/F mode.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import platform
import time
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import stats
from torch.utils.data import DataLoader
import yaml

from src.data.dataset import MarsTerrainDataset, TargetScaler, derive_safe_decisions
from src.data.splits import get_fold_indices
from src.models import build_model
from src.train import load_or_generate_metadata, prepare_run_caches, train_single_run
from src.utils import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

TARGETS = ["mean_slope_5m", "roughness_m", "hazard_fraction"]


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)


def fit_training_image_bounds(metadata: pd.DataFrame, train_df: pd.DataFrame) -> dict[str, Any]:
    """Fit calibrated-I/F 2–98% bounds using valid pixels from training rows only."""
    shards: dict[str, np.ndarray] = {}
    pieces: list[np.ndarray] = []
    n_pixels = 0
    for shard_path, rows in train_df.groupby("shard_path", sort=True):
        shard = np.load(str(shard_path), mmap_mode="r")
        for start in range(0, len(rows), 128):
            sub = rows.iloc[start : start + 128]
            idx = sub.shard_idx.to_numpy(dtype=np.int64)
            patches = np.asarray(shard[idx, 0], dtype=np.float32)
            valid = np.asarray(shard[idx, 2] > 0, dtype=bool)
            values = patches[valid]
            if values.size:
                pieces.append(values)
                n_pixels += int(values.size)
    if not pieces:
        raise ValueError("No valid training image pixels available to fit radiometric bounds")
    values = np.concatenate(pieces)
    low, high = np.percentile(values, [2.0, 98.0]).astype(float)
    if not np.isfinite(low + high) or high <= low:
        raise ValueError(f"Invalid training-only radiometric bounds: {low}, {high}")
    return {
        "if_low": float(low),
        "if_high": float(high),
        # Preserve out-of-fold calibrated-I/F contrast instead of mapping
        # valid distribution shift to constant black/white pixels.
        "clip": False,
        "application": "training-fold affine transform; no clipping; invalid image pixels set to zero",
        "fit_percentiles": [2.0, 98.0],
        "fit_scope": "valid pixels from training-fold patches only",
        "fit_sites": sorted(train_df.site_id.astype(str).unique().tolist()),
        "fit_patch_count": int(len(train_df)),
        "fit_valid_pixel_count": n_pixels,
    }


def make_fold_config(base: dict[str, Any], fold: int, seed: int, *, smoke: bool = False) -> tuple[dict[str, Any], dict[str, Any]]:
    cfg = copy.deepcopy(base)
    metadata = pd.read_parquet(cfg["paths"]["processed_dir"] + "/metadata.parquet")
    splits = get_fold_indices(metadata, test_fold=fold, n_splits=5)
    train_sites = set(splits["train"].site_id.astype(str))
    val_sites = set(splits["val"].site_id.astype(str))
    test_sites = set(splits["test"].site_id.astype(str))
    if len(val_sites) != 1 or len(test_sites) != 1 or train_sites & val_sites or train_sites & test_sites or val_sites & test_sites:
        raise AssertionError("Fold must have disjoint site-level train/validation/test groups")
    bounds = fit_training_image_bounds(metadata, splits["train"])
    cfg["input_normalization"] = bounds
    cfg["training"]["seeds"] = [int(seed)]
    cfg["controlled_run"] = {
        "fold": int(fold),
        "seed": int(seed),
        "test_site": sorted(test_sites)[0],
        "validation_site": sorted(val_sites)[0],
        "training_sites": sorted(train_sites),
        "normalization_fit_scope": bounds["fit_scope"],
    }
    if smoke:
        cfg["training"]["epochs"] = 1
        cfg["training"]["early_stopping_patience"] = 1
        out = Path(base["controlled_root"]) / "smoke"
        cfg["paths"]["runs_dir"] = str(out / "runs")
        cfg["paths"]["cache_dir"] = str(out / "cache" / f"fold_{fold}_seed_{seed}")
        cfg["paths"]["results_dir"] = str(out / "results")
    else:
        out = Path(base["controlled_root"])
        cfg["paths"]["runs_dir"] = str(out / "runs")
        cfg["paths"]["cache_dir"] = str(out / "cache" / f"fold_{fold}_seed_{seed}")
        cfg["paths"]["results_dir"] = str(out / "results")
    cfg["paths"]["cache_dir"] = str(Path(cfg["paths"]["cache_dir"]).resolve())
    meta = {
        **cfg["controlled_run"],
        "train_count": int(len(splits["train"])),
        "validation_count": int(len(splits["val"])),
        "test_count": int(len(splits["test"])),
        "training_safe_counts": {str(k): int(v) for k, v in splits["train"].safe_label.value_counts().sort_index().items()},
        "validation_safe_counts": {str(k): int(v) for k, v in splits["val"].safe_label.value_counts().sort_index().items()},
        "test_safe_counts": {str(k): int(v) for k, v in splits["test"].safe_label.value_counts().sort_index().items()},
        "normalization": bounds,
    }
    conf_dir = Path(base["controlled_root"]) / ("smoke/configs" if smoke else "configs")
    conf_dir.mkdir(parents=True, exist_ok=True)
    conf_path = conf_dir / f"fold_{fold}_seed_{seed}.yaml"
    conf_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return cfg, meta


def validation_predictions(cfg: dict[str, Any], model_name: str, variant: str, fold: int, seed: int) -> pd.DataFrame:
    metadata = load_or_generate_metadata(cfg)
    split = get_fold_indices(metadata, test_fold=fold, n_splits=5)
    scaler = TargetScaler(cfg["preprocessing"]["target_cols"], cfg["preprocessing"]["log1p_targets"]).fit(split["train"])
    embeddings_all, quanv_all = prepare_run_caches(cfg, model_name, variant, fold, seed, metadata)
    tile_to_idx = {str(row.tile_id): i for i, row in metadata.iterrows()}
    indices = [tile_to_idx[str(tid)] for tid in split["val"].tile_id]
    embeddings = embeddings_all[indices] if embeddings_all is not None else None
    quanv = quanv_all[indices] if quanv_all is not None else None
    ds = MarsTerrainDataset(split["val"], scaler=scaler, config=cfg, is_train=False, embeddings=embeddings, quanv_maps=quanv)
    model = build_model(model_name, variant, config=cfg, use_sun=ds.effective_use_sun)
    device = cfg["training"].get("device", "cpu")
    run_dir = Path(cfg["paths"]["runs_dir"]) / f"{model_name}_{variant}" / f"fold_{fold}_seed_{seed}"
    checkpoint = torch.load(run_dir / "best_checkpoint.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    loader = DataLoader(ds, batch_size=int(cfg["training"]["batch_size"]), shuffle=False, num_workers=0)
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            outputs.append(model(batch).detach().cpu().numpy())
    raw = scaler.inverse_transform(np.concatenate(outputs, axis=0))
    safe = derive_safe_decisions(raw, cfg["preprocessing"]["safe_mean_slope_deg"], cfg["preprocessing"]["safe_hazard_fraction"])
    frame = pd.DataFrame({
        "tile_id": split["val"].tile_id.to_numpy(), "image_id": split["val"].image_id.to_numpy(),
        "fold": fold, "seed": seed, "model_name": model_name, "variant": variant,
        "pred_mean_slope_5m": raw[:, 0], "true_mean_slope_5m": split["val"].mean_slope_5m.to_numpy(),
        "pred_roughness_m": raw[:, 1], "true_roughness_m": split["val"].roughness_m.to_numpy(),
        "pred_hazard_fraction": raw[:, 2], "true_hazard_fraction": split["val"].hazard_fraction.to_numpy(),
        "pred_safe": safe, "true_safe": split["val"].safe_label.to_numpy(dtype=int),
    })
    dest = run_dir / "validation_predictions.parquet"
    tmp = dest.with_suffix(".parquet.tmp")
    frame.to_parquet(tmp, index=False)
    os.replace(tmp, dest)
    return frame


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
    yt = np.asarray(y_true, dtype=int)
    yp = np.asarray(y_pred, dtype=int)
    tp = int(np.sum((yt == 1) & (yp == 1)))
    fp = int(np.sum((yt == 0) & (yp == 1)))
    tn = int(np.sum((yt == 0) & (yp == 0)))
    fn = int(np.sum((yt == 1) & (yp == 0)))
    div = lambda a, b: float(a / b) if b else float("nan")
    return {
        "TP": tp, "FP": fp, "TN": tn, "FN": fn,
        "safe_precision": div(tp, tp + fp), "safe_recall": div(tp, tp + fn),
        "specificity": div(tn, tn + fp), "false_positive_rate": div(fp, fp + tn),
        "false_discovery_rate": div(fp, tp + fp), "predicted_safe_fraction": div(tp + fp, len(yt)),
        "n_tiles": int(len(yt)), "n_actually_safe": int(np.sum(yt == 1)),
        "n_actually_unsafe": int(np.sum(yt == 0)),
    }


def metrics_for_frame(frame: pd.DataFrame, *, threshold: float | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {}
    if threshold is None:
        pred_safe = frame.pred_safe.to_numpy(dtype=int)
    else:
        risk = np.maximum(
            frame.pred_mean_slope_5m.to_numpy(dtype=float) / 8.0,
            frame.pred_hazard_fraction.to_numpy(dtype=float) / 0.05,
        )
        pred_safe = (risk < threshold).astype(int)
    result.update(binary_metrics(frame.true_safe.to_numpy(dtype=int), pred_safe))
    for target in TARGETS:
        y = frame[f"true_{target}"].to_numpy(dtype=float)
        p = frame[f"pred_{target}"].to_numpy(dtype=float)
        ok = np.isfinite(y) & np.isfinite(p)
        result[f"n_finite_{target}"] = int(ok.sum())
        result[f"RMSE_{target}"] = float(np.sqrt(np.mean((y[ok] - p[ok]) ** 2))) if ok.any() else float("nan")
        result[f"MAE_{target}"] = float(np.mean(np.abs(y[ok] - p[ok]))) if ok.any() else float("nan")
    return result


def select_validation_threshold(frame: pd.DataFrame, target_fpr: float = 0.05) -> tuple[float, dict[str, Any]]:
    """Choose largest risk threshold meeting FPR target on validation only."""
    risk = np.maximum(
        frame.pred_mean_slope_5m.to_numpy(dtype=float) / 8.0,
        frame.pred_hazard_fraction.to_numpy(dtype=float) / 0.05,
    )
    candidates = np.unique(risk)
    candidates = np.append(candidates, np.nextafter(float(np.max(risk)), np.inf))
    eligible: list[tuple[float, dict[str, Any]]] = []
    for threshold in candidates:
        m = binary_metrics(frame.true_safe.to_numpy(dtype=int), (risk < threshold).astype(int))
        if np.isfinite(m["false_positive_rate"]) and m["false_positive_rate"] <= target_fpr:
            eligible.append((float(threshold), m))
    if not eligible:
        raise ValueError("Validation data has no threshold satisfying target FPR")
    return max(eligible, key=lambda x: x[0])


def aggregate_metrics(run_df: pd.DataFrame, root: Path, stem: str) -> None:
    run_df.to_csv(root / f"{stem}_per_site_seed.csv", index=False)
    metric_cols = [c for c in run_df.select_dtypes(include=[np.number]).columns if c not in {"fold", "seed", "n_tiles", "n_actually_safe", "n_actually_unsafe"} and not c.startswith("n_finite_")]
    site = run_df.groupby(["model", "test_site"], as_index=False)[metric_cols].mean()
    site.to_csv(root / f"{stem}_site_mean.csv", index=False)
    rows = []
    for model, group in site.groupby("model"):
        for metric in metric_cols:
            values = group[metric].dropna().to_numpy(dtype=float)
            if not len(values):
                mean = sd = lo = hi = float("nan")
            else:
                mean = float(values.mean())
                sd = float(values.std(ddof=1)) if len(values) > 1 else float("nan")
                half = float(stats.t.ppf(0.975, len(values)-1) * sd / np.sqrt(len(values))) if len(values) > 1 else float("nan")
                lo, hi = mean-half, mean+half
            rows.append({"model": model, "metric": metric, "n_sites": len(values), "mean_site_metric": mean, "sd_across_sites": sd, "t95_low": lo, "t95_high": hi})
    pd.DataFrame(rows).to_csv(root / f"{stem}_aggregate.csv", index=False)

    # Paired comparison: first average seeds within each site, then use five sites
    # as the units for descriptive paired intervals.
    paired_rows = []
    for metric in metric_cols:
        wide = site.pivot(index="test_site", columns="model", values=metric).dropna()
        if {"CNN", "CNN_QCNN"}.issubset(wide.columns):
            diff = (wide["CNN_QCNN"] - wide["CNN"]).to_numpy(dtype=float)
            n = len(diff)
            if n == 0:
                paired_rows.append({"metric": metric, "difference_QCNN_minus_CNN": float("nan"), "sd_across_sites": float("nan"), "n_sites": 0, "t95_low": float("nan"), "t95_high": float("nan")})
                continue
            mean = float(diff.mean())
            sd = float(diff.std(ddof=1)) if n > 1 else float("nan")
            half = float(stats.t.ppf(0.975, n-1) * sd / np.sqrt(n)) if n > 1 else float("nan")
            paired_rows.append({"metric": metric, "difference_QCNN_minus_CNN": mean, "sd_across_sites": sd, "n_sites": n, "t95_low": mean-half, "t95_high": mean+half})
    pd.DataFrame(paired_rows).to_csv(root / f"{stem}_paired_site_differences.csv", index=False)


def collect_results(root: Path, base: dict[str, Any], target_fpr: float = 0.05) -> None:
    rows: list[dict[str, Any]] = []
    threshold_rows: list[dict[str, Any]] = []
    threshold_test_rows: list[dict[str, Any]] = []
    run_root = root / "runs"
    for model, variant, name in [("model1", "resnet18", "CNN"), ("model2", "qcnn", "CNN_QCNN")]:
        for test_path in run_root.glob(f"{model}_{variant}/fold_*/test_predictions.parquet"):
            test = pd.read_parquet(test_path)
            if not len(test):
                continue
            fold, seed = int(test.fold.iloc[0]), int(test.seed.iloc[0])
            cfg_path = root / "configs" / f"fold_{fold}_seed_{seed}.yaml"
            cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
            test_site = cfg["controlled_run"]["test_site"]
            val_path = test_path.with_name("validation_predictions.parquet")
            if not val_path.exists():
                logging.warning("Missing validation predictions for %s", test_path)
                continue
            val = pd.read_parquet(val_path)
            if set(test.tile_id) & set(val.tile_id):
                raise AssertionError("Validation and test tile IDs overlap")
            current = {"model": name, "fold": fold, "seed": seed, "test_site": test_site}
            rows.append({**current, **metrics_for_frame(test)})
            tau, valm = select_validation_threshold(val, target_fpr)
            testm = metrics_for_frame(test, threshold=tau)
            threshold_rows.append({**current, "risk_threshold": tau, "target_validation_fpr": target_fpr, **{f"validation_{k}": v for k,v in valm.items()}})
            threshold_test_rows.append({**current, "risk_threshold": tau, **testm})
    if not rows:
        raise FileNotFoundError(f"No complete predictions found under {run_root}")
    result_root = root / "metrics"
    result_root.mkdir(parents=True, exist_ok=True)
    fixed = pd.DataFrame(rows)
    aggregate_metrics(fixed, result_root, "fixed_threshold")
    pd.DataFrame(threshold_rows).to_csv(result_root / "thresholds_selected_on_validation.csv", index=False)
    calibrated = pd.DataFrame(threshold_test_rows)
    aggregate_metrics(calibrated, result_root, "validation_calibrated_test")
    pd.DataFrame(threshold_rows).to_csv(result_root / "thresholds_selected_on_validation.csv", index=False)
    fixed.to_parquet(result_root / "fixed_threshold_per_site_seed.parquet", index=False)
    calibrated.to_parquet(result_root / "validation_calibrated_test_per_site_seed.parquet", index=False)
    fig, ax = plt.subplots(figsize=(7, 5))
    for model, grp in fixed.groupby("model"):
        ax.scatter(grp.false_positive_rate, grp.safe_recall, s=70, label=model)
        for _, r in grp.iterrows():
            ax.annotate(str(r.test_site), (r.false_positive_rate, r.safe_recall), xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax.set_xlabel("False-positive rate among unsafe tiles")
    ax.set_ylabel("Safe recall")
    ax.set_title("Fixed 8° / 0.05 safety thresholds (site-level runs)")
    ax.grid(True, alpha=0.3); ax.legend(); fig.tight_layout()
    fig.savefig(result_root / "fpr_vs_safe_recall.png", dpi=160); plt.close(fig)
    cov, ax = plt.subplots(figsize=(7, 5))
    for model, grp in fixed.groupby("model"):
        ax.scatter(grp.false_positive_rate, grp.predicted_safe_fraction, s=70, label=model)
        for _, r in grp.iterrows():
            ax.annotate(str(r.test_site), (r.false_positive_rate, r.predicted_safe_fraction), xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax.set_xlabel("False-positive rate among unsafe tiles")
    ax.set_ylabel("Fraction of test tiles predicted safe")
    ax.set_title("Fixed-threshold unsafe rejection versus accepted-tile coverage")
    ax.grid(True, alpha=0.3); ax.legend(); cov.tight_layout()
    cov.savefig(result_root / "fpr_vs_safe_coverage.png", dpi=160); plt.close(cov)


def run_one(base: dict[str, Any], root: Path, fold: int, seed: int, *, smoke: bool = False) -> dict[str, Any]:
    cfg, meta = make_fold_config(base, fold, seed, smoke=smoke)
    conf_root = Path(cfg["paths"]["runs_dir"])
    (conf_root / "protocol").mkdir(parents=True, exist_ok=True)
    atomic_json(conf_root / "protocol" / f"fold_{fold}_seed_{seed}.json", meta)
    started = time.perf_counter()

    cnn_cfg = copy.deepcopy(cfg)
    cnn_cfg["training"]["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    cnn_started = time.perf_counter()
    m1 = train_single_run(cnn_cfg, "model1", "resnet18", fold, seed)
    cnn_seconds = time.perf_counter() - cnn_started
    gpu_peak_bytes = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0

    # Feature extraction is inference on each partition after the CNN backbone
    # has been fit using training sites and selected with validation only.
    # Test imagery is forwarded through the frozen encoder to obtain predictions;
    # no test labels or test-fitted parameters enter the cache.
    cache_cfg = copy.deepcopy(cfg)
    cache_cfg["training"]["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    metadata = load_or_generate_metadata(cache_cfg)
    cache_started = time.perf_counter()
    prepare_run_caches(cache_cfg, "model2", "qcnn", fold, seed, metadata)
    embedding_seconds = time.perf_counter() - cache_started
    qcnn_cfg = copy.deepcopy(cfg)
    qcnn_cfg["training"]["device"] = "cpu"  # PennyLane default.qubit is CPU-backed here.
    qcnn_started = time.perf_counter()
    m2 = train_single_run(qcnn_cfg, "model2", "qcnn", fold, seed)
    qcnn_seconds = time.perf_counter() - qcnn_started

    v1 = validation_predictions(cnn_cfg, "model1", "resnet18", fold, seed)
    v2 = validation_predictions(qcnn_cfg, "model2", "qcnn", fold, seed)
    test1 = pd.read_parquet(Path(cnn_cfg["paths"]["runs_dir"]) / "model1_resnet18" / f"fold_{fold}_seed_{seed}" / "test_predictions.parquet")
    test2 = pd.read_parquet(Path(qcnn_cfg["paths"]["runs_dir"]) / "model2_qcnn" / f"fold_{fold}_seed_{seed}" / "test_predictions.parquet")
    for a, b, label in [(test1,test2,"test"),(v1,v2,"validation")]:
        if not a.tile_id.equals(b.tile_id) or not a[["true_safe", *[f"true_{t}" for t in TARGETS]]].equals(b[["true_safe", *[f"true_{t}" for t in TARGETS]]]):
            raise AssertionError(f"CNN/QCNN {label} IDs or labels differ in fold {fold}")
    duration = time.perf_counter() - started
    run_info = {
        **meta, "seed": seed, "cnn_status": m1.get("status"), "qcnn_status": m2.get("status"),
        "elapsed_seconds": duration, "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda, "cuda_available": bool(torch.cuda.is_available()),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cnn_wall_seconds": cnn_seconds, "embedding_wall_seconds": embedding_seconds,
        "qcnn_wall_seconds": qcnn_seconds, "cnn_peak_cuda_allocated_bytes": gpu_peak_bytes,
        "gpu_total_memory_bytes": int(torch.cuda.get_device_properties(0).total_memory) if torch.cuda.is_available() else 0,
        "qcnn_simulator": "PennyLane default.qubit on CPU", "smoke": smoke,
        "identical_test_ids": True, "identical_validation_ids": True,
        "finite_cnn_test": bool(np.isfinite(test1.select_dtypes(include=[np.number]).to_numpy()).all()),
        "finite_qcnn_test": bool(np.isfinite(test2.select_dtypes(include=[np.number]).to_numpy()).all()),
    }
    try:
        import psutil
        run_info["system_ram_total_bytes"] = int(psutil.virtual_memory().total)
        run_info["system_ram_available_bytes_at_completion"] = int(psutil.virtual_memory().available)
    except ImportError:
        run_info["system_ram_total_bytes"] = None
        run_info["system_ram_available_bytes_at_completion"] = None
    atomic_json(conf_root / "protocol" / f"fold_{fold}_seed_{seed}_completion.json", run_info)
    return run_info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", choices=["prepare", "smoke", "run", "metrics"], required=True)
    parser.add_argument("--folds", default="all", help="Comma-separated folds or all")
    parser.add_argument("--seeds", default="42", help="Comma-separated seeds")
    args = parser.parse_args()
    base = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    root = Path(base["controlled_root"])
    root.mkdir(parents=True, exist_ok=True)
    if args.mode == "prepare":
        from src.data.preprocess_real import main as preprocess_main
        # The preprocessor's CLI rereads this exact isolated config.
        import sys
        sys.argv = [sys.argv[0], "--config", args.config]
        preprocess_main()
        return
    if args.mode == "metrics":
        collect_results(root, base, target_fpr=float(base.get("safety", {}).get("validation_fpr_target", 0.05)))
        return
    if args.mode == "smoke":
        info = run_one(base, root, 0, int(args.seeds.split(",")[0]), smoke=True)
        print(json.dumps(info, indent=2))
        return
    folds = list(range(5)) if args.folds == "all" else [int(x) for x in args.folds.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]
    if sorted(set(folds)) != folds or any(x not in range(5) for x in folds):
        raise ValueError("folds must be a unique ordered subset of 0..4")
    progress_path = root / "progress.json"
    if progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
    else:
        progress = {"planned_folds": folds, "planned_seeds": seeds, "completed": []}
    progress["planned_folds"] = sorted(set(progress.get("planned_folds", [])) | set(folds))
    progress["planned_seeds"] = sorted(set(progress.get("planned_seeds", [])) | set(seeds))
    progress["status"] = "running"
    atomic_json(progress_path, progress)
    for fold in folds:
        for seed in seeds:
            info = run_one(base, root, fold, seed)
            done = {"fold": fold, "seed": seed, "test_site": info["test_site"]}
            progress["completed"] = [x for x in progress["completed"] if not (x["fold"] == fold and x["seed"] == seed)] + [done]
            progress["last_completed"] = done
            atomic_json(progress_path, progress)
            print(json.dumps(info, sort_keys=True))
    collect_results(root, base, target_fpr=float(base.get("safety", {}).get("validation_fpr_target", 0.05)))
    if len(progress["completed"]) >= len(progress["planned_folds"]) * len(progress["planned_seeds"]):
        progress["status"] = "completed"
        atomic_json(progress_path, progress)


if __name__ == "__main__":
    main()
