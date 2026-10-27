"""Compare two Moon run folders bit for bit (read-only).

Everything a run produces is compared except wall-clock timings (`seconds`, `inference`,
per-epoch `seconds`), which legitimately differ: logits, predictions, metrics, histories
(losses and validation macro-F1) and every tensor in every final checkpoint.

    .venv/Scripts/python.exe experiments/moon/compare_runs.py <run_a> <run_b> [--seeds 42 43]

Exit status 0 when identical, 1 otherwise.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1].parent
TIMING_KEYS = {"seconds", "inference"}
CHECKPOINTS = ("cnn", "hybrid", "frozen_mlp")


def strip_timings(obj):
    if isinstance(obj, dict):
        return {k: strip_timings(v) for k, v in obj.items() if k not in TIMING_KEYS}
    if isinstance(obj, list):
        return [strip_timings(v) for v in obj]
    return obj


def diff(a, b, path="") -> list[str]:
    if type(a) is not type(b):
        return [f"{path}: type {type(a).__name__} vs {type(b).__name__}"]
    if isinstance(a, dict):
        out = [f"{path}.{k}: missing on one side" for k in sorted(set(a) ^ set(b))]
        for k in sorted(set(a) & set(b)):
            out += diff(a[k], b[k], f"{path}.{k}")
        return out
    if isinstance(a, list):
        if len(a) != len(b):
            return [f"{path}: length {len(a)} vs {len(b)}"]
        out = []
        for i, (x, y) in enumerate(zip(a, b)):
            out += diff(x, y, f"{path}[{i}]")
            if len(out) > 5:
                break
        return out
    return [] if a == b else [f"{path}: {a!r} vs {b!r}"]


def resolve(p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else REPO / path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_a")
    parser.add_argument("run_b")
    parser.add_argument("--seeds", type=int, nargs="*", help="Seeds to compare (default: those in run_b)")
    args = parser.parse_args()
    a, b = resolve(args.run_a), resolve(args.run_b)
    seeds = args.seeds or json.loads((b / "summary.json").read_text(encoding="utf-8"))["config"]["seeds"]

    problems: list[str] = []
    for seed in seeds:
        ra = strip_timings(json.loads((a / "logs" / f"seed_{seed}" / "results.json").read_text(encoding="utf-8")))
        rb = strip_timings(json.loads((b / "logs" / f"seed_{seed}" / "results.json").read_text(encoding="utf-8")))
        problems += [f"seed {seed} results{d}" for d in diff(ra, rb)]
        for name in CHECKPOINTS:
            sa = torch.load(a / "checkpoints" / f"{name}_seed_{seed}.pt", map_location="cpu", weights_only=False)["state_dict"]
            sb = torch.load(b / "checkpoints" / f"{name}_seed_{seed}.pt", map_location="cpu", weights_only=False)["state_dict"]
            if sa.keys() != sb.keys():
                problems.append(f"seed {seed} {name}: checkpoint keys differ")
                continue
            for key in sa:
                if not torch.equal(sa[key], sb[key]):
                    problems.append(f"seed {seed} {name}.{key}: max |diff| {(sa[key].double() - sb[key].double()).abs().max().item():.3g}")
    if problems:
        print(f"DIFFERENT ({len(problems)} differences; first 20):")
        print("\n".join(problems[:20]))
        sys.exit(1)
    print(f"IDENTICAL: seeds {seeds}; results (timings excluded) and all checkpoint tensors match bit for bit")


if __name__ == "__main__":
    main()
