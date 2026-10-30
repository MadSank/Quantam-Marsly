"""Live walkthrough of the Mars fold 0 / seed 42 CNN and hybrid CNN-QCNN checkpoints.

Run from the repository root with the pipeline venv:

    .venv\\Scripts\\python.exe demo\\show_model.py

Read-only: CPU only, no training, nothing is written. Checkpoints are loaded with
map_location="cpu", models are put in eval() mode and every forward pass runs under no_grad().
The models are rebuilt with the repo's own classes (src.models.build_model) from each run's
config_snapshot.yaml. Press Enter to move between sections (--no-pause runs straight through).
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pennylane as qml  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

from src.data.dataset import MarsTerrainDataset, TargetScaler, derive_safe_decisions  # noqa: E402
from src.data.splits import get_fold_indices  # noqa: E402
from src.models import build_model  # noqa: E402

RUNS = REPO_ROOT / "runs" / "pipeline_repair_audit" / "retraining_seedset" / "runs"
FOLD, SEED = 0, 42
CNN_DIR = RUNS / "model1_resnet18" / f"fold_{FOLD}_seed_{SEED}"
HYB_DIR = RUNS / "model2_qcnn" / f"fold_{FOLD}_seed_{SEED}"
TILE_ID = "site_01_r128_c4096"  # the /lab page's example tile, the first fold 0 test tile
PAUSE = "--no-pause" not in sys.argv


def section(title: str) -> None:
    if PAUSE:
        input("\nPress Enter...")
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def count(module: torch.nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable


def load(run_dir: Path, name: str, variant: str) -> tuple[torch.nn.Module, dict]:
    cfg = yaml.safe_load((run_dir / "config_snapshot.yaml").read_text(encoding="utf-8"))
    use_sun = bool(cfg.get("features", {}).get("use_sun_features", True))
    model = build_model(name, variant, cfg, use_sun=use_sun)
    ckpt = torch.load(run_dir / "best_checkpoint.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded {run_dir.relative_to(REPO_ROOT)}\\best_checkpoint.pt (best epoch {ckpt['epoch'] + 1}, "
          f"validation loss {ckpt['best_val_loss']:.4f})")
    return model, cfg


def main() -> None:
    # qml.draw uses box-drawing characters, which the Windows console code page cannot encode.
    sys.stdout.reconfigure(encoding="utf-8")
    torch.set_grad_enabled(False)
    np.set_printoptions(precision=4, suppress=True, linewidth=110)

    print("Mars terrain: CNN (ResNet-18) vs hybrid CNN-QCNN, fold 0, seed 42 (CPU, read-only)")
    cnn, cfg = load(CNN_DIR, "model1", "resnet18")
    hyb, hcfg = load(HYB_DIR, "model2", "qcnn")

    section("1. Architecture and parameter counts")
    print(cnn)
    total, trainable = count(cnn)
    print(f"\nCNN: {total:,} parameters, {trainable:,} trainable (trained end to end)")
    print(hyb)
    total, trainable = count(hyb)
    print(f"\nHybrid head: {total:,} parameters, {trainable:,} trainable")
    print(f"  bottleneck {count(hyb.bottleneck)[0]:,} + QCNN head {count(hyb.head)[0]:,}")
    print(f"  It reads the frozen CNN's 512-number embedding ({count(cnn.backbone)[0]:,} frozen parameters).")

    section("2. First convolution layer of the CNN")
    w = cnn.backbone.conv1.weight
    print(f"conv1.weight shape: {tuple(w.shape)}  (64 filters, 3 input channels, 7 x 7)")
    print("Filter 0, image channel, top-left 3 x 3:")
    print(w[0, 0, :3, :3].numpy())
    print(f"Range over all filters: {w.min():.4f} to {w.max():.4f}")

    section("3. The hybrid's 48 trained circuit parameters")
    head = hyb.head
    print("Convolution angles (3 stages x 12), radians:")
    print(head.conv_weights.numpy())
    print("Pooling angles (2 stages x 3), radians:")
    print(head.pool_weights.numpy())
    print("Readout scale (3):", head.out_scale.numpy())
    print("Readout bias  (3):", head.out_bias.numpy())
    print(f"Total: {head.conv_weights.numel()} + {head.pool_weights.numel()} + "
          f"{head.out_scale.numel()} + {head.out_bias.numel()} = {count(head)[0]}")

    section("4. The circuit (qml.draw), simulated with PennyLane default.qubit")
    example = torch.full((8,), float(np.pi / 2))
    print(qml.draw(head.circuit, decimals=None)(example, head.conv_weights, head.pool_weights))

    section(f"5. One real test tile through both models: {TILE_ID}")
    meta = pd.read_parquet(Path(cfg["paths"]["processed_dir"]) / "metadata.parquet")
    splits = get_fold_indices(meta, test_fold=FOLD, n_splits=int(cfg.get("splits", {}).get("n_splits", 5)))
    pre = cfg["preprocessing"]
    # As in src/train.py: the target scaler is fitted on the training fold only.
    scaler = TargetScaler(target_cols=pre["target_cols"], log1p_cols=pre["log1p_targets"]).fit(splits["train"])
    row = splits["test"][splits["test"]["tile_id"] == TILE_ID]
    item = MarsTerrainDataset(row, scaler=None, config=cfg, is_train=False)[0]
    image = item["image"].unsqueeze(0)
    print(f"Site {row.iloc[0]['site_id']} (the fold 0 test site), input tensor {tuple(image.shape)}")

    emb = cnn.extract_embeddings(image)
    preds = {
        "CNN": scaler.inverse_transform(cnn({"image": image}).numpy()),
        "Hybrid": scaler.inverse_transform(hyb({"embedding": emb}).numpy()),
    }
    angles = hyb.encode_bottleneck(emb)[0].numpy()
    print(f"Hybrid encoding angles for this tile (radians): {angles}")

    t = row.iloc[0]
    true = [t["mean_slope_5m"], t["roughness_m"], t["hazard_fraction"], int(t["safe_label"])]
    print(f"\n{'':<10}{'slope (deg)':>13}{'roughness (m)':>15}{'hazard frac':>13}{'safe (proxy)':>14}")
    print(f"{'True':<10}{true[0]:>13.3f}{true[1]:>15.3f}{true[2]:>13.3f}{true[3]:>14d}")
    for name, p in preds.items():
        safe = int(derive_safe_decisions(p, pre["safe_mean_slope_deg"], pre["safe_hazard_fraction"])[0])
        print(f"{name:<10}{p[0, 0]:>13.3f}{p[0, 1]:>15.3f}{p[0, 2]:>13.3f}{safe:>14d}")
    print(f"\n'Safe' is a proxy: mean slope < {pre['safe_mean_slope_deg']} deg and hazard fraction "
          f"< {pre['safe_hazard_fraction']}; not an operational landing-safety system.")

    # The same tile in each run's saved test predictions, as a check on this live run.
    for name, run_dir in (("CNN", CNN_DIR), ("Hybrid", HYB_DIR)):
        saved = pd.read_parquet(run_dir / "test_predictions.parquet").set_index("tile_id").loc[TILE_ID]
        live = preds[name][0]
        diff = max(abs(live[0] - saved["pred_mean_slope_5m"]), abs(live[1] - saved["pred_roughness_m"]),
                   abs(live[2] - saved["pred_hazard_fraction"]))
        print(f"{name}: saved test prediction matches this run to within {diff:.1e}")
    print("\nOne tile is an illustration, not a result; see the site's results for all five folds.")


if __name__ == "__main__":
    main()
