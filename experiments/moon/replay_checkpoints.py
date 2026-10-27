"""Read-only replay of the reported Moon run's saved checkpoints (no training).

For each seed: load the saved CNN, hybrid and frozen-MLP weights on CPU, then
  1. recompute test logits and compare them with the logits saved in results.json;
  2. recompute validation macro-F1 and compare it with the best val_macro_f1 in history.csv.

Run from the repository root:

    .venv/Scripts/python.exe experiments/moon/replay_checkpoints.py

Writes nothing; prints a JSON report. See README.md "Code history".
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))

from train import CNN, ClassicalHead, HybridHead, LunarImages  # noqa: E402

DATASET = REPO / "data" / "raw" / "moon" / "lrocnet"
RUNS = DATASET / "experiment_lunar_cnn_qcnn" / "runs"
SEEDS = (42, 43, 44)
HISTORY_KEY = {"cnn": "cnn", "hybrid_qcnn": "qcnn", "frozen_embedding_mlp": "frozen_mlp"}


def load(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)["state_dict"]


@torch.no_grad()
def images_through(model, ds, batch_size=64):
    model.eval()
    out = [model(x) for x, _ in DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)]
    return torch.cat(out).numpy()


def labels(ds) -> np.ndarray:
    return np.array([y for _, y in ds.items], dtype=np.int64)


def main() -> None:
    torch.set_num_threads(max(1, torch.get_num_threads()))
    test_ds, val_ds = LunarImages(DATASET, "test"), LunarImages(DATASET, "val")
    test_ids = [p.relative_to(DATASET).as_posix() for p, _ in test_ds.items]
    y_val = labels(val_ds)
    report = {"seeds": {}}
    for seed in SEEDS:
        saved = json.loads((RUNS / "logs" / f"seed_{seed}" / "results.json").read_text(encoding="utf-8"))
        assert saved["test"]["sample_ids"] == test_ids, "test order differs from the saved run"
        history: dict[str, list[float]] = {}
        with (RUNS / "logs" / f"seed_{seed}" / "history.csv").open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                history.setdefault(row["model"], []).append(float(row["val_macro_f1"]))

        cnn = CNN()
        cnn.load_state_dict(load(RUNS / "checkpoints" / f"cnn_seed_{seed}.pt"))
        hybrid, mlp = HybridHead(), ClassicalHead()
        hybrid.load_state_dict(load(RUNS / "checkpoints" / f"hybrid_seed_{seed}.pt"))
        mlp.load_state_dict(load(RUNS / "checkpoints" / f"frozen_mlp_seed_{seed}.pt"))

        test_logits = {"cnn": images_through(cnn, test_ds)}
        val_logits = {"cnn": images_through(cnn, val_ds)}
        test_emb = torch.from_numpy(images_through(cnn.encoder, test_ds))
        val_emb = torch.from_numpy(images_through(cnn.encoder, val_ds))
        with torch.no_grad():
            for name, head in (("hybrid_qcnn", hybrid), ("frozen_embedding_mlp", mlp)):
                head.eval()
                test_logits[name] = head(test_emb).numpy()
                val_logits[name] = head(val_emb).numpy()

        seed_report = {}
        for name in ("cnn", "hybrid_qcnn", "frozen_embedding_mlp"):
            ref = np.asarray(saved["test"]["logits"][name], dtype=np.float64)
            got = test_logits[name].astype(np.float64)
            val_f1 = float(f1_score(y_val, val_logits[name].argmax(1), average="macro", zero_division=0))
            best_logged = max(history[HISTORY_KEY[name]])
            seed_report[name] = {
                "test_logits_max_abs_diff": float(np.abs(got - ref).max()),
                "test_predictions_identical": bool((got.argmax(1) == ref.argmax(1)).all()),
                "test_prediction_mismatches": int((got.argmax(1) != ref.argmax(1)).sum()),
                "val_macro_f1_recomputed": val_f1,
                "val_macro_f1_best_logged": best_logged,
                "val_macro_f1_abs_diff": abs(val_f1 - best_logged),
            }
        report["seeds"][str(seed)] = seed_report
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
