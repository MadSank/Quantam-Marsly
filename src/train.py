"""Generic resumable training pipeline for Mars terrain models."""

import argparse
import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml

from src.data.dataset import MarsTerrainDataset, TargetScaler, derive_safe_decisions
from src.data.splits import assign_grouped_folds, get_fold_indices
from src.data.synthetic import generate_smoke_dataset
from src.models import build_model
from src.models.bottleneck_qcnn import precompute_and_cache_backbone_embeddings
from src.models.quanv_cnn import precompute_or_load_quanv_cache
from src.utils import EpochTimerCSV, count_parameters, load_config, save_checkpoint_atomic, set_all_seeds

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def capture_rng_state() -> dict[str, Any]:
    """Serialize Python, NumPy, and Torch RNG state for exact run resumption."""
    np_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": {
            "bit_generator": str(np_state[0]),
            "keys": np_state[1].tolist(),
            "pos": int(np_state[2]),
            "has_gauss": int(np_state[3]),
            "cached_gaussian": float(np_state[4]),
        },
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_state(state: dict[str, Any]) -> None:
    """Restore a state written by :func:`capture_rng_state`, when present."""
    if not state:
        return
    if "python" in state:
        random.setstate(tuple(state["python"]))
    if "numpy" in state:
        ns = state["numpy"]
        np.random.set_state(
            (
                ns["bit_generator"],
                np.asarray(ns["keys"], dtype=np.uint32),
                int(ns["pos"]),
                int(ns["has_gauss"]),
                float(ns["cached_gaussian"]),
            )
        )
    if "torch_cpu" in state:
        torch.set_rng_state(state["torch_cpu"])
    cuda_states = state.get("torch_cuda", [])
    if cuda_states and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_states)


def load_or_generate_metadata(config: dict[str, Any]) -> pd.DataFrame:
    """Loads metadata.parquet or generates synthetic smoke dataset if in smoke mode."""
    processed_dir = Path(config["paths"]["processed_dir"])
    meta_path = processed_dir / "metadata.parquet"

    if meta_path.exists():
        return pd.read_parquet(meta_path)

    if config.get("smoke", {}).get("enabled", False):
        logging.info("Smoke mode enabled: Generating synthetic smoke dataset...")
        return generate_smoke_dataset(config, seed=42)

    raise FileNotFoundError(
        f"Metadata not found at {meta_path}. Run preprocessing first: python -m src.data.preprocess"
    )


def prepare_run_caches(
    config: dict[str, Any],
    model_name: str,
    variant: str,
    fold: int,
    seed: int,
    metadata_df: pd.DataFrame,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Precomputes or loads cached backbone embeddings for Model 2 or quanv maps for Model 3."""
    cache_dir = Path(config["paths"]["cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    device = config["training"].get("device", "cpu")

    embeddings_array: np.ndarray | None = None
    quanv_array: np.ndarray | None = None

    if model_name == "model2":
        emb_cache_path = cache_dir / f"embeddings_fold_{fold}_seed_{seed}.npy"
        if emb_cache_path.exists():
            embeddings_array = np.load(emb_cache_path)
        else:
            runs_dir = Path(config["paths"]["runs_dir"])
            m1_ckpt_path = (
                runs_dir / "model1_resnet18" / f"fold_{fold}_seed_{seed}" / "best_checkpoint.pt"
            )
            if not m1_ckpt_path.exists():
                logging.info(
                    "Model 1 checkpoint not found for fold %d seed %d. Auto-training Model 1 first...",
                    fold,
                    seed,
                )
                train_single_run(config, "model1", "resnet18", fold, seed)

            use_sun = bool(config.get("features", {}).get("use_sun_features", True))
            m1_model = build_model("model1", "resnet18", config=config, use_sun=use_sun)
            ckpt = torch.load(m1_ckpt_path, map_location=device)
            m1_model.load_state_dict(ckpt["model_state_dict"])
            m1_model.to(device)

            full_dataset = MarsTerrainDataset(metadata_df, config=config, is_train=False)
            embeddings_array = precompute_and_cache_backbone_embeddings(
                m1_model, full_dataset, emb_cache_path, batch_size=32, device=device
            )

    elif model_name == "model3" and variant == "quanv":
        quanv_cache_path = cache_dir / "quanv_maps.npy"
        if quanv_cache_path.exists():
            quanv_array = np.load(quanv_cache_path)
        else:
            logging.info("Precomputing 4-qubit quanvolution feature maps...")
            shards_cache: dict[str, np.ndarray] = {}
            img_list: list[np.ndarray] = []
            for _, row in metadata_df.iterrows():
                sp = str(row["shard_path"])
                if sp not in shards_cache:
                    shards_cache[sp] = np.load(sp, mmap_mode="r")
                img_list.append(shards_cache[sp][int(row["shard_idx"])])
            images_np = np.stack(img_list, axis=0)

            n_layers = int(config["models"]["model3"].get("n_random_layers", 2))
            q_seed = int(config["models"]["model3"].get("quanv_seed", 2026))
            quanv_array = precompute_or_load_quanv_cache(
                images_np, quanv_cache_path, n_layers=n_layers, seed=q_seed
            )

    return embeddings_array, quanv_array


def train_single_run(
    config: dict[str, Any], model_name: str, variant: str, fold: int, seed: int
) -> dict[str, Any]:
    """Executes a single training run with crash-safe checkpointing, logging, and evaluation."""
    seed_info = set_all_seeds(seed)
    logging.info(
        "Starting run: model=%s, variant=%s, fold=%d, seed=%d (torch=%s, pennylane=%s)",
        model_name,
        variant,
        fold,
        seed,
        seed_info["torch"],
        seed_info["pennylane"],
    )

    runs_dir = Path(config["paths"]["runs_dir"])
    run_dir = runs_dir / f"{model_name}_{variant}" / f"fold_{fold}_seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    metadata_df = load_or_generate_metadata(config)
    n_splits = int(config.get("splits", {}).get("n_splits", 5))

    if "fold" not in metadata_df.columns:
        metadata_df = assign_grouped_folds(
            metadata_df, n_splits=n_splits, seed=int(config.get("splits", {}).get("split_seed", 42))
        )

    splits = get_fold_indices(metadata_df, test_fold=fold, n_splits=n_splits)

    target_cols = config["preprocessing"]["target_cols"]
    log1p_cols = config["preprocessing"]["log1p_targets"]

    # TargetScaler fit STRICTLY on training fold only
    scaler = TargetScaler(target_cols=target_cols, log1p_cols=log1p_cols).fit(splits["train"])

    # Prepare caches
    embeddings_all, quanv_all = prepare_run_caches(config, model_name, variant, fold, seed, metadata_df)

    # Index alignment for cached arrays
    tile_to_idx = {row["tile_id"]: idx for idx, row in metadata_df.iterrows()}

    def get_slice(arr: np.ndarray | None, split_df: pd.DataFrame) -> np.ndarray | None:
        if arr is None:
            return None
        idxs = [tile_to_idx[tid] for tid in split_df["tile_id"]]
        return arr[idxs]

    train_ds = MarsTerrainDataset(
        splits["train"],
        scaler=scaler,
        config=config,
        is_train=True,
        embeddings=get_slice(embeddings_all, splits["train"]),
        quanv_maps=get_slice(quanv_all, splits["train"]),
    )
    val_ds = MarsTerrainDataset(
        splits["val"],
        scaler=scaler,
        config=config,
        is_train=False,
        embeddings=get_slice(embeddings_all, splits["val"]),
        quanv_maps=get_slice(quanv_all, splits["val"]),
    )
    test_ds = MarsTerrainDataset(
        splits["test"],
        scaler=scaler,
        config=config,
        is_train=False,
        embeddings=get_slice(embeddings_all, splits["test"]),
        quanv_maps=get_slice(quanv_all, splits["test"]),
    )

    effective_use_sun = train_ds.effective_use_sun

    # Build model
    model = build_model(model_name, variant, config=config, use_sun=effective_use_sun)
    param_counts = count_parameters(model)

    with open(run_dir / "param_counts.json", "w", encoding="utf-8") as f:
        json.dump(param_counts, f, indent=2)

    with open(run_dir / "config_snapshot.yaml", "w", encoding="utf-8") as f:
        yaml.dump(config, f)

    train_cfg = config["training"]
    device = train_cfg.get("device", "cpu")
    batch_size = int(train_cfg.get("batch_size", 32))
    num_workers = int(train_cfg.get("num_workers", 0))
    epochs = int(train_cfg.get("epochs", 30))
    lr = float(train_cfg.get("lr", 1e-4))
    weight_decay = float(train_cfg.get("weight_decay", 1e-4))
    patience = int(train_cfg.get("early_stopping_patience", 7))

    model.to(device)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()

    timer_csv = EpochTimerCSV(run_dir / "epoch_log.csv")

    start_epoch = 0
    best_val_loss = float("inf")
    patience_counter = 0
    cumulative_wall_time = 0.0

    last_ckpt_path = run_dir / "last_checkpoint.pt"
    best_ckpt_path = run_dir / "best_checkpoint.pt"
    test_preds_path = run_dir / "test_predictions.parquet"

    # Resumption check
    if last_ckpt_path.exists():
        logging.info("Found existing checkpoint %s. Restoring state...", last_ckpt_path)
        ckpt = torch.load(last_ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = int(ckpt["epoch"]) + 1
        best_val_loss = float(ckpt["best_val_loss"])
        patience_counter = int(ckpt.get("patience_counter", 0))
        cumulative_wall_time = float(ckpt.get("cumulative_wall_time_sec", 0.0))

        restore_rng_state(ckpt.get("rng_state", {}))

        if ckpt.get("training_complete", False) and test_preds_path.exists():
            logging.info("Run already complete for %s/%s fold %d seed %d.", model_name, variant, fold, seed)
            return {"status": "already_complete", "run_dir": str(run_dir)}

    # Training loop
    for epoch in range(start_epoch, epochs):
        t0 = time.perf_counter()

        model.train()
        train_loss_total = 0.0
        train_batches = 0

        for batch in train_loader:
            batch_dev = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()
            }
            optimizer.zero_grad()
            preds = model(batch_dev)
            loss = criterion(preds, batch_dev["target"])
            loss.backward()
            optimizer.step()

            train_loss_total += loss.item()
            train_batches += 1

        train_loss = train_loss_total / max(1, train_batches)

        # Validation
        model.eval()
        val_loss_total = 0.0
        val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                batch_dev = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()
                }
                preds = model(batch_dev)
                v_loss = criterion(preds, batch_dev["target"])
                val_loss_total += v_loss.item()
                val_batches += 1

        val_loss = val_loss_total / max(1, val_batches)
        current_lr = float(optimizer.param_groups[0]["lr"])
        scheduler.step()

        wall_time = time.perf_counter() - t0
        cumulative_wall_time += wall_time

        timer_csv.log_epoch(epoch, train_loss, val_loss, current_lr, wall_time, cumulative_wall_time)
        logging.info(
            "Epoch [%d/%d] Train Loss: %.4f | Val Loss: %.4f | LR: %.2e | Time: %.2fs",
            epoch + 1,
            epochs,
            train_loss,
            val_loss,
            current_lr,
            wall_time,
        )

        # Best checkpoint update
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            save_checkpoint_atomic(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "best_val_loss": best_val_loss,
                    "val_loss": val_loss,
                    "config": config,
                },
                best_ckpt_path,
            )
        else:
            patience_counter += 1

        # Atomic last checkpoint save
        save_checkpoint_atomic(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_val_loss": best_val_loss,
                "patience_counter": patience_counter,
                "cumulative_wall_time_sec": cumulative_wall_time,
                "training_complete": False,
                "rng_state": capture_rng_state(),
            },
            last_ckpt_path,
        )

        if patience_counter >= patience:
            logging.info("Early stopping triggered at epoch %d.", epoch + 1)
            break

    # Evaluation on test fold with best checkpoint
    if best_ckpt_path.exists():
        best_ckpt = torch.load(best_ckpt_path, map_location=device)
        model.load_state_dict(best_ckpt["model_state_dict"])

    model.eval()
    test_preds_list: list[np.ndarray] = []
    with torch.no_grad():
        for batch in test_loader:
            batch_dev = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()
            }
            preds = model(batch_dev)
            test_preds_list.append(preds.cpu().numpy())

    if test_preds_list:
        preds_scaled = np.concatenate(test_preds_list, axis=0)
        preds_raw = scaler.inverse_transform(preds_scaled)
        pred_safe = derive_safe_decisions(
            preds_raw,
            safe_slope_deg=config["preprocessing"]["safe_mean_slope_deg"],
            safe_hazard_frac=config["preprocessing"]["safe_hazard_fraction"],
        )
    else:
        preds_raw = np.zeros((0, len(target_cols)), dtype=np.float32)
        pred_safe = np.zeros(0, dtype=int)

    test_df = splits["test"].copy()
    test_preds_df = pd.DataFrame(
        {
            "tile_id": test_df["tile_id"],
            "image_id": test_df["image_id"],
            "fold": fold,
            "seed": seed,
            "model_name": model_name,
            "variant": variant,
            "pred_mean_slope_5m": preds_raw[:, 0],
            "true_mean_slope_5m": test_df["mean_slope_5m"],
            "pred_roughness_m": preds_raw[:, 1],
            "true_roughness_m": test_df["roughness_m"],
            "pred_hazard_fraction": preds_raw[:, 2],
            "true_hazard_fraction": test_df["hazard_fraction"],
            "pred_safe": pred_safe,
            "true_safe": test_df["safe_label"],
        }
    )
    test_preds_df.to_parquet(test_preds_path, index=False)
    logging.info("Saved test fold predictions to %s", test_preds_path)

    # Mark training complete in last checkpoint
    save_checkpoint_atomic(
        {
            "epoch": epochs - 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_loss": best_val_loss,
            "patience_counter": patience_counter,
            "cumulative_wall_time_sec": cumulative_wall_time,
            "training_complete": True,
            "rng_state": capture_rng_state(),
        },
        last_ckpt_path,
    )

    return {"status": "complete", "run_dir": str(run_dir)}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train classical and quantum Mars terrain models.")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file.")
    parser.add_argument("--model", type=str, default="model1", help="model1, model2, model3, model4, or all.")
    parser.add_argument("--variant", type=str, default=None, help="Specific variant or 'all'.")
    parser.add_argument("--fold", type=str, default="0", help="Fold index (int) or 'all'.")
    parser.add_argument("--seed", type=str, default="0", help="Seed (int) or 'all'.")
    parser.add_argument("--smoke", action="store_true", help="Run in fast smoke mode.")
    parser.add_argument("--all", action="store_true", help="Train all models, variants, folds, and seeds.")
    parser.add_argument("--override", nargs="*", default=None, help="Config overrides key=value.")
    args = parser.parse_args(argv)

    cfg = load_config(args.config, overrides=args.override, smoke=args.smoke)

    if args.all:
        models = ["model1", "model2", "model3", "model4"]
    elif args.model.lower() == "all":
        models = ["model1", "model2", "model3", "model4"]
    else:
        models = [args.model.lower()]

    folds = (
        list(range(int(cfg["splits"]["n_splits"])))
        if args.fold.lower() == "all"
        else [int(args.fold)]
    )
    seeds = (
        list(cfg["training"]["seeds"])
        if args.seed.lower() == "all"
        else [int(args.seed)]
    )

    for m in models:
        m_cfg = cfg["models"].get(m, {})
        configured_variants = m_cfg.get("variants", ["default"])
        if args.variant is None or args.variant.lower() == "all":
            variants = configured_variants
        else:
            variants = [args.variant.lower()]

        for v in variants:
            for f in folds:
                for s in seeds:
                    train_single_run(cfg, model_name=m, variant=v, fold=f, seed=s)


if __name__ == "__main__":
    main()
