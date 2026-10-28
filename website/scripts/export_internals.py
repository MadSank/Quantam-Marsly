"""Export the /lab page's internals: figures, parameters, curves and code excerpts.

Run from the repository root with the root venv (torch + pennylane + matplotlib):

    .venv/Scripts/python.exe website/scripts/export_internals.py

Read-only by construction (PLAN.md, "Export scripts"): every checkpoint is loaded on the CPU
with map_location="cpu", models are put in eval() mode and run under torch.no_grad(), and each
checkpoint's SHA-256 is recorded before and after loading (the export fails if it changed).

Inputs
  Mars: runs/pipeline_repair_audit/retraining_seedset/runs/{model1_resnet18,model2_qcnn}/fold_0_seed_42
        (best_checkpoint.pt, config_snapshot.yaml, epoch_log.csv for all folds and seeds).
        The feature-map tile is the first tile of fold 0's test predictions, loaded through
        MarsTerrainDataset with the fold's own config, so the training-fold image normalisation
        is applied exactly as in evaluation.
  Moon: the reported deterministic 10-seed run (FINAL_READINESS_AUDIT.md, Addendum):
        data/raw/moon/lrocnet/experiment_lunar_cnn_qcnn/runs_deterministic_10seed_20261028
        (checkpoints/*_seed_42.pt, logs/seed_*/history.csv). Its summary.json records the SHA-256
        of experiments/moon/train.py; the export checks that it matches the committed file.
  Code: excerpts are sliced from `git show <HEAD>:<path>` and located by function or class name
        (plus optional anchor lines inside it), never by fixed line numbers. The export refuses to
        run if an excerpted file has uncommitted changes or HEAD is not on the remote.

Writes website/public/data/lab/: checkpoints.json, params.json, mars_cnn_conv1.png,
mars_cnn_features.json + mars_cnn_stage_*.png, qcnn_circuit.svg,
{mars,moon}_rotations.json, {mars,moon}_projection.json, curves.json, excerpts.json.
"""

from __future__ import annotations

import ast
import hashlib
import io
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from PIL import Image  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "experiments" / "moon"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pennylane as qml  # noqa: E402

from src.data.dataset import MarsTerrainDataset  # noqa: E402
from src.models import build_model  # noqa: E402
from stable_io import write_bytes, write_json  # noqa: E402

OUT = REPO_ROOT / "website" / "public" / "data" / "lab"
REPO = "MadSank/Quantam-Marsly"
REMOTE_BRANCH = "origin/website-redesign"

MARS_RUNS = REPO_ROOT / "runs" / "pipeline_repair_audit" / "retraining_seedset" / "runs"
MARS_FOLD, MARS_SEED = 0, 42
MARS_CNN_DIR = MARS_RUNS / "model1_resnet18" / f"fold_{MARS_FOLD}_seed_{MARS_SEED}"
MARS_HYB_DIR = MARS_RUNS / "model2_qcnn" / f"fold_{MARS_FOLD}_seed_{MARS_SEED}"
MARS_METADATA = REPO_ROOT / "runs" / "controlled_evaluation" / "processed" / "metadata.parquet"

MOON_RUN = REPO_ROOT / "data" / "raw" / "moon" / "lrocnet" / "experiment_lunar_cnn_qcnn" / "runs_deterministic_10seed_20261028"
MOON_SEED = 42
MOON_TRAIN_PY = REPO_ROOT / "experiments" / "moon" / "train.py"
MOON_README = REPO_ROOT / "experiments" / "moon" / "README.md"

MOON_CURVE_KEYS = {"cnn": "cnn", "qcnn": "hybrid_qcnn", "frozen_mlp": "frozen_embedding_mlp"}  # history.csv -> site keys
GRID_CHANNELS = 16  # feature-map channels shown per stage (4 x 4)
CELL = 64  # px per feature-map cell in the PNG grids


def rel(p: Path) -> str:
    return p.resolve().relative_to(REPO_ROOT).as_posix()


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_checkpoint(p: Path, record: list[dict[str, Any]], **meta: Any) -> dict[str, Any]:
    before = sha256(p)
    mtime = p.stat().st_mtime_ns
    ck = torch.load(p, map_location="cpu", weights_only=False)
    assert sha256(p) == before and p.stat().st_mtime_ns == mtime, f"{p} changed while loading"
    record.append({**meta, "path": rel(p), "sha256": before})
    return ck


def png_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, "PNG", optimize=True)
    return buf.getvalue()


def to_u8(a: np.ndarray) -> np.ndarray:
    lo, hi = float(np.min(a)), float(np.max(a))
    return np.zeros(a.shape, np.uint8) if hi <= lo else ((a - lo) / (hi - lo) * 255).round().astype(np.uint8)


def grid(tiles: list[np.ndarray], cols: int, cell: int, gap: int = 2) -> np.ndarray:
    """Tiles (each normalised on its own) as one grayscale grid with dark gaps."""
    rows = (len(tiles) + cols - 1) // cols
    out = np.full((rows * cell + (rows - 1) * gap, cols * cell + (cols - 1) * gap), 18, np.uint8)
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        img = Image.fromarray(to_u8(t)).resize((cell, cell), Image.Resampling.NEAREST)
        out[r * (cell + gap) : r * (cell + gap) + cell, c * (cell + gap) : c * (cell + gap) + cell] = np.asarray(img)
    return out


def count(module: torch.nn.Module) -> int:
    return int(sum(p.numel() for p in module.parameters()))


# ---------------------------------------------------------------- Mars

def mars(written: list[Path], checkpoints: list[dict[str, Any]]) -> dict[str, Any]:
    cfg = yaml.safe_load((MARS_CNN_DIR / "config_snapshot.yaml").read_text(encoding="utf-8"))
    hcfg = yaml.safe_load((MARS_HYB_DIR / "config_snapshot.yaml").read_text(encoding="utf-8"))
    use_sun = bool(cfg.get("features", {}).get("use_sun_features", True))

    cnn = build_model("model1", "resnet18", cfg, use_sun=use_sun)
    ck = load_checkpoint(MARS_CNN_DIR / "best_checkpoint.pt", checkpoints, model="CNN", world="mars", fold=MARS_FOLD, seed=MARS_SEED)
    cnn.load_state_dict(ck["model_state_dict"])
    cnn.eval()

    hyb = build_model("model2", "qcnn", hcfg, use_sun=bool(hcfg.get("features", {}).get("use_sun_features", True)))
    hk = load_checkpoint(MARS_HYB_DIR / "best_checkpoint.pt", checkpoints, model="CNN_QCNN", world="mars", fold=MARS_FOLD, seed=MARS_SEED)
    hyb.load_state_dict(hk["model_state_dict"])
    hyb.eval()

    # Parameter counts, cross-checked against the run's own param_counts.json.
    pc_cnn = json.loads((MARS_CNN_DIR / "param_counts.json").read_text(encoding="utf-8"))
    pc_hyb = json.loads((MARS_HYB_DIR / "param_counts.json").read_text(encoding="utf-8"))
    cnn_total = count(cnn)
    hyb_total = count(hyb)
    assert cnn_total == pc_cnn["total_trainable"], (cnn_total, pc_cnn)
    assert hyb_total == pc_hyb["total_trainable"], (hyb_total, pc_hyb)
    params = {
        "cnn": {"total": cnn_total, "trainable": cnn_total, "head": count(cnn.head)},
        "hybrid": {
            "total": hyb_total, "trainable": hyb_total,
            "frozen_backbone": count(cnn.backbone),
            "bottleneck": count(hyb.bottleneck),
            "qcnn_head": count(hyb.head),
        },
    }

    # conv1 filters: the image channel of each of the 64 filters, each normalised on its own.
    w = cnn.backbone.conv1.weight.detach().numpy()  # (64, 3, 7, 7)
    path = OUT / "mars_cnn_conv1.png"
    write_bytes(path, png_bytes(grid([w[i, 0] for i in range(w.shape[0])], cols=8, cell=28)))
    written.append(path)

    # Feature maps for one real fold-0 test tile.
    preds = pd.read_parquet(MARS_CNN_DIR / "test_predictions.parquet")
    tile_id = str(preds.iloc[0]["tile_id"])
    meta = pd.read_parquet(MARS_METADATA)
    row = meta[meta["tile_id"] == tile_id]
    assert len(row) == 1, tile_id
    ds = MarsTerrainDataset(row, scaler=None, config=cfg, is_train=False)
    item = ds[0]
    image = item["image"].unsqueeze(0)
    stages = [("stem", cnn.backbone.relu), ("layer1", cnn.backbone.layer1), ("layer2", cnn.backbone.layer2),
              ("layer3", cnn.backbone.layer3), ("layer4", cnn.backbone.layer4)]
    acts: dict[str, np.ndarray] = {}
    hooks = [m.register_forward_hook(lambda _m, _i, o, n=n: acts.__setitem__(n, o.detach().numpy()[0])) for n, m in stages]
    with torch.no_grad():
        emb = cnn.extract_embeddings(image)
        angles = hyb.encode_bottleneck(emb)
    for h in hooks:
        h.remove()
    input_png = OUT / "mars_cnn_input.png"
    write_bytes(input_png, png_bytes(np.asarray(Image.fromarray(to_u8(image[0, 0].numpy())).resize((256, 256), Image.Resampling.NEAREST))))
    written.append(input_png)
    stage_out = []
    for name, _ in stages:
        a = acts[name]
        p = OUT / f"mars_cnn_stage_{name}.png"
        write_bytes(p, png_bytes(grid([a[i] for i in range(min(GRID_CHANNELS, a.shape[0]))], cols=4, cell=CELL)))
        written.append(p)
        stage_out.append({"name": name, "shape": list(a.shape), "channels_shown": min(GRID_CHANNELS, a.shape[0]), "png": f"lab/{p.name}"})
    features = {
        "tile_id": tile_id, "site_id": str(row.iloc[0]["site_id"]), "fold": MARS_FOLD, "seed": MARS_SEED,
        "input": {"png": f"lab/{input_png.name}", "channel": "calibrated image (channel 0 of 3), training-fold normalisation"},
        "input_channels": ["calibrated image", "DTM-valid coverage", "image-valid coverage"],
        "stages": stage_out,
        "bottleneck_angles": [round(float(v), 4) for v in angles[0].tolist()],
    }

    # Mars and Moon share one circuit (BatchedQCNNHead, 8 qubits, angle encoding); only the trained
    # angles differ, so one drawing serves both. Angles are drawn without values.
    circuit(hyb.head, angles[0], OUT / "qcnn_circuit.svg", written)
    rotations(hyb.head, OUT / "mars_rotations.json", written, "mars")
    projection(OUT / "mars_projection.json", written, in_dim=512, world="mars",
               notes=["The trained ResNet-18 is frozen; its 512-number embedding of the tile is the input.",
                      "A linear layer maps those 512 numbers to 8 (the bottleneck, 4,104 parameters).",
                      "Each of the 8 values goes through tanh and is rescaled to an angle between 0 and pi.",
                      "Each angle sets one RY rotation on one of the 8 qubits (angle encoding)."])
    return {"params": params, "features": features}


# ---------------------------------------------------------------- Moon

def moon(written: list[Path], checkpoints: list[dict[str, Any]]) -> dict[str, Any]:
    recorded = json.loads((MOON_RUN / "summary.json").read_text(encoding="utf-8"))["determinism"]["train_py_sha256"]
    assert sha256(MOON_TRAIN_PY) == recorded, "experiments/moon/train.py differs from the source that produced the run"
    import train as moon_train  # experiments/moon/train.py

    cnn = moon_train.CNN()
    cnn.load_state_dict(load_checkpoint(MOON_RUN / "checkpoints" / f"cnn_seed_{MOON_SEED}.pt", checkpoints, model="cnn", world="moon", seed=MOON_SEED)["state_dict"])
    hyb = moon_train.HybridHead()
    hyb.load_state_dict(load_checkpoint(MOON_RUN / "checkpoints" / f"hybrid_seed_{MOON_SEED}.pt", checkpoints, model="hybrid_qcnn", world="moon", seed=MOON_SEED)["state_dict"])
    mlp = moon_train.ClassicalHead()
    mlp.load_state_dict(load_checkpoint(MOON_RUN / "checkpoints" / f"frozen_mlp_seed_{MOON_SEED}.pt", checkpoints, model="frozen_embedding_mlp", world="moon", seed=MOON_SEED)["state_dict"])
    for m in (cnn, hyb, mlp):
        m.eval()
    params = {
        "cnn": {"trainable_during_training": count(cnn), "trainable_post_freeze": count(cnn.classifier)},
        "hybrid_head": count(hyb), "mlp_head": count(mlp),
        "hybrid_bottleneck": count(hyb.bottleneck), "hybrid_qcnn": count(hyb.qcnn),
    }
    rotations(hyb.qcnn, OUT / "moon_rotations.json", written, "moon")
    projection(OUT / "moon_projection.json", written, in_dim=64, world="moon",
               notes=["The trained CNN encoder is frozen; its 64-number embedding of the crop is the input.",
                      "A linear layer maps those 64 numbers to 8 (the bottleneck, 520 parameters).",
                      "Each of the 8 values goes through tanh and is rescaled to an angle between 0 and pi.",
                      "Each angle sets one RY rotation on one of the 8 qubits (angle encoding)."])
    return {"params": params}


# ---------------------------------------------------------------- circuit, rotations, projection

def circuit(head: torch.nn.Module, angles: torch.Tensor, path: Path, written: list[Path]) -> None:
    """The head's own QNode drawn by qml.draw_mpl, saved as SVG with text kept as text."""
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["svg.hashsalt"] = "lab"  # stable ids, so unchanged circuits are not rewritten
    with torch.no_grad():
        fig, _ = qml.draw_mpl(head.circuit, style="black_white_dark", decimals=None)(angles, head.conv_weights, head.pool_weights)
    fig.patch.set_alpha(0)
    for ax in fig.axes:
        ax.patch.set_alpha(0)
    buf = io.StringIO()
    fig.savefig(buf, format="svg", transparent=True, bbox_inches="tight", metadata={"Date": None})
    plt.close(fig)
    svg = re.sub(r"<!--.*?-->|<metadata>.*?</metadata>|<\?xml[^>]*\?>|<!DOCTYPE[^>]*>", "", buf.getvalue(), flags=re.S)
    # Theme hooks: the page inlines the SVG, so lines and labels follow currentColor and gate boxes
    # take the page surface. The drawing style uses only these two colours.
    assert set(re.findall(r"#[0-9a-f]{6}", svg)) <= {"#ffffff", "#151515"}, "unexpected colour in circuit drawing"
    svg = svg.replace("#ffffff", "currentColor").replace("#151515", "var(--circuit-fill)")
    svg = re.sub(r">\s+<", "><", svg).strip() + "\n"
    write_bytes(path, svg.encode("utf-8"))
    written.append(path)


def rotations(head: torch.nn.Module, path: Path, written: list[Path], world: str) -> None:
    conv = head.conv_weights.detach().numpy()
    pool = head.pool_weights.detach().numpy()
    layers = [
        {"name": f"Convolution stage {i + 1}", "kind": "angles", "shape": [12], "values": [[round(float(v), 5) for v in conv[i]]],
         "labels": ["RY", "RZ", "RY", "RZ", "RY", "RZ"] * 2}
        for i in range(conv.shape[0])
    ] + [
        {"name": f"Pooling stage {i + 1}", "kind": "angles", "shape": [3], "values": [[round(float(v), 5) for v in pool[i]]],
         "labels": ["RZ", "RY", "RY"]}
        for i in range(pool.shape[0])
    ]
    angles = np.concatenate([conv.ravel(), pool.ravel()])
    write_json(path, {
        "world": world,
        "source": "the head's conv_weights (3 x 12) and pool_weights (2 x 3) from the seed-42 checkpoint",
        "layers": layers,
        "readout": {"scale": [round(float(v), 5) for v in head.out_scale.detach().numpy()],
                    "bias": [round(float(v), 5) for v in head.out_bias.detach().numpy()]},
        "value_range": [round(float(angles.min()), 5), round(float(angles.max()), 5)],
        "unit": "radians",
    })
    written.append(path)


def projection(path: Path, written: list[Path], *, in_dim: int, world: str, notes: list[str]) -> None:
    write_json(path, {"world": world, "in_dim": in_dim, "out_dim": 8,
                      "encoding": "angle: theta = (tanh(z) + 1) * pi / 2, one RY per qubit", "notes": notes})
    written.append(path)


# ---------------------------------------------------------------- curves

def curves() -> dict[str, Any]:
    out: dict[str, Any] = {"epoch_base": 1, "mars": {}, "moon": {}}
    for fold in range(5):
        for seed in (42, 43, 44):
            key = f"{fold}_{seed}"
            out["mars"][key] = {}
            for model, d in (("cnn", "model1_resnet18"), ("qcnn", "model2_qcnn")):
                log = pd.read_csv(MARS_RUNS / d / f"fold_{fold}_seed_{seed}" / "epoch_log.csv")
                out["mars"][key][model] = {
                    "epoch": [int(e) + 1 for e in log["epoch"]],  # logged 0-based
                    "train_loss": [round(float(v), 5) for v in log["train_loss"]],
                    "val_loss": [round(float(v), 5) for v in log["val_loss"]],
                }
    for seed_dir in sorted((MOON_RUN / "logs").glob("seed_*"), key=lambda p: int(p.name.split("_")[1])):
        h = pd.read_csv(seed_dir / "history.csv")
        out["moon"][seed_dir.name.split("_")[1]] = {
            MOON_CURVE_KEYS[m]: {"epoch": [int(e) for e in g["epoch"]],
                "train_loss": [round(float(v), 5) for v in g["train_loss"]],
                "val_macro_f1": [round(float(v), 5) for v in g["val_macro_f1"]]}
            for m, g in h.groupby("model", sort=False)
        }
    return out


# ---------------------------------------------------------------- code excerpts

# (id, title, path, symbol, start anchor, end anchor(s), world, stage, note). The symbol is a top-level
# function, class or assignment, or "First..Last" for a run of them; a list of end anchors is matched in turn.
EXCERPTS: list[tuple[str, str, str, str, str | None, str | list[str] | None, str, str | None, str]] = [
    ("dtm_grid", "Reading the DTM grid", "src/data/preprocess_real.py", "pds_label_grid", None, None, "mars", "strip",
     "Each HiRISE strip comes with a terrain model. Its map grid (projection and pixel transform) is read from the PDS label, so ortho and DTM pixels share one coordinate frame."),
    ("tiling", "Cutting the strip into tiles", "src/data/preprocess_real.py", "preprocess_site",
     r"for r in range\(0, d\.height", [r'"excluded_joint_coverage"', r"continue"], "mars", "tiling",
     "The strip is walked on a regular grid. A window is kept only if the terrain model and the image both cover at least 95% of it."),
    ("targets", "Terrain targets and the safe proxy", "src/data/preprocess_real.py", "preprocess_site",
     r"rr0, cc0 = r-max_halo", r"safe = int\(\(mean5 < 8\.0\)", "mars", "targets",
     "Slope is computed on the native terrain model with a halo, so no artificial edge enters the target. Mean slope, roughness and the share of steep cells become the three targets; the safe label is derived from two of them."),
    ("safe_decisions", "Safe decisions from predictions", "src/data/dataset.py", "derive_safe_decisions", None, None, "mars", "safe",
     "A model's predicted slope and hazard fraction are turned into a safe or unsafe decision with the same thresholds as the proxy label."),
    ("cnn_model", "The CNN (ResNet-18)", "src/models/cnn.py", "Model1ResNet18", None, None, "mars", "models",
     "A ResNet-18 trained end to end. Its 512-number embedding feeds a small classifier that predicts the three targets."),
    ("hybrid_model", "The hybrid: frozen embedding, bottleneck, QCNN head", "src/models/bottleneck_qcnn.py", "BottleneckQCNNModel", None, None, "mars", "models",
     "The hybrid reuses the trained CNN's embedding, frozen. A linear bottleneck reduces it to 8 numbers, which become rotation angles for the 8-qubit circuit."),
    ("qcnn_circuit", "The QCNN circuit and batched evaluation", "src/quantum/circuits.py", "BatchedQCNNHead", None, None, "both", None,
     "The circuit used on Mars and on the Moon. It runs on PennyLane's default.qubit simulator with the whole batch at once (parameter broadcasting); a test checks it against a per-sample loop."),
    ("training_loop", "Training loop with early stopping", "src/train.py", "train_single_run",
     r"for epoch in range\(start_epoch, epochs\):", r"if patience_counter >= patience:", "mars", None,
     "Each epoch trains on the training sites, then scores the validation site. The best validation loss is kept, and training stops after a run of epochs without improvement."),
    ("target_scaler", "Target scaling fitted on the training fold only", "src/data/dataset.py", "TargetScaler", None, None, "mars", None,
     "Targets are log-scaled and standardised with statistics from the training fold only, so validation and test sites never inform the scaling."),
    ("image_bounds", "Image normalisation fitted on the training fold only", "src/controlled_evaluation.py", "fit_training_image_bounds", None, None, "mars", None,
     "The image brightness range is fitted on valid pixels of the training fold's tiles only, then applied unchanged to validation and test."),
    ("loso_split", "Leave-one-site-out split", "src/data/splits.py", "get_fold_indices", None, None, "mars", "evaluation",
     "Each fold holds out one site as the test site and one as the validation site; tiles from one site never appear in two roles."),
    ("validation_threshold", "Validation-calibrated threshold", "src/controlled_evaluation.py", "select_validation_threshold", None, None, "mars", "evaluation",
     "The calibrated regime picks the largest risk threshold that keeps the validation site's false-positive rate at or below the target, then applies it to the test site unchanged."),
    ("moon_models", "Moon: the CNN encoder and classifier", "experiments/moon/train.py", "Encoder..CNN", None, None, "moon", None,
     "A small CNN with a 64-number pooled embedding. Trained end to end, the encoder and classifier have 59,347 trainable parameters."),
    ("moon_heads", "Moon: the hybrid head", "experiments/moon/train.py", "HybridHead", None, None, "moon", None,
     "The hybrid head maps the frozen 64-number embedding to 8 angles and evaluates the same QCNN circuit as on Mars."),
    ("moon_control", "Moon: the matched classical control", "experiments/moon/train.py", "ClassicalHead", None, None, "moon", None,
     "The MLP control sees the same frozen embedding with a similar parameter budget (547 against the hybrid's 568)."),
    ("moon_fit", "Moon: training loop with early stopping", "experiments/moon/train.py", "fit", None, None, "moon", None,
     "Each model is trained on the training crops and selected on validation macro-F1, with early stopping; test data is not touched here."),
    ("moon_streams", "Moon: named random streams", "experiments/moon/train.py", "STREAMS..stream_generator", None, None, "moon", None,
     "Each initialisation and data loader draws from its own random stream, derived from the seed and a stream name, which makes a run repeatable bit for bit."),
]


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO_ROOT, check=True, capture_output=True, text=True, encoding="utf-8").stdout


def top_level(tree: ast.Module, name: str) -> ast.stmt:
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name:
            return n
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in n.targets):
            return n
    raise AssertionError(f"{name} not found")


def span(source: str, symbol: str, start: str | None, end: str | list[str] | None) -> tuple[int, int]:
    """1-based line span of a top-level symbol (with the comment block right above it), or of the
    anchored block inside it."""
    tree = ast.parse(source)
    lines = source.splitlines()
    head, _, tail = symbol.partition("..")
    a, b = top_level(tree, head), top_level(tree, tail or head)
    first = min([a.lineno] + [d.lineno for d in getattr(a, "decorator_list", [])])
    last = b.end_lineno
    if start is None:
        while first > 1 and lines[first - 2].lstrip().startswith("#"):
            first -= 1
        return first, last
    s = next(i for i in range(first - 1, last) if re.search(start, lines[i])) + 1
    e = s
    for pattern in [end] if isinstance(end, str) else end or []:
        e = next(i for i in range(e - 1, last) if re.search(pattern, lines[i])) + 1
    return s, (e if end else last)


def history_note(commit: str) -> dict[str, Any]:
    text = git("show", f"{commit}:experiments/moon/README.md")
    lines = text.splitlines()
    s = next(i for i, l in enumerate(lines) if l.strip() == "## Code history")
    e = next((i for i in range(s + 1, len(lines)) if lines[i].startswith("## ")), len(lines))
    while e > s and not lines[e - 1].strip():
        e -= 1
    return {"text": "\n".join(lines[s:e]), "start": s + 1, "end": e,
            "permalink": f"https://github.com/{REPO}/blob/{commit}/experiments/moon/README.md#L{s + 1}-L{e}"}


def excerpts() -> dict[str, Any]:
    paths = sorted({e[2] for e in EXCERPTS} | {"experiments/moon/README.md"})
    dirty = git("status", "--porcelain", "--", *paths).strip()
    if dirty:
        raise SystemExit(f"Refusing to export: excerpted files have uncommitted changes:\n{dirty}")
    commit = git("rev-parse", "HEAD").strip()
    if REMOTE_BRANCH not in git("branch", "-r", "--contains", commit):
        raise SystemExit(f"Refusing to export: {commit} is not on {REMOTE_BRANCH}, so permalinks would not resolve")
    note = history_note(commit)
    out = []
    for ex_id, title, path, symbol, start, end, world, stage, text in EXCERPTS:
        source = git("show", f"{commit}:{path}")
        s, e = span(source, symbol, start, end)
        code = "\n".join(source.splitlines()[s - 1 : e])
        entry = {"id": ex_id, "title": title, "world": world, "stage": stage, "path": path, "symbol": symbol,
                 "start": s, "end": e, "note": text,
                 "permalink": f"https://github.com/{REPO}/blob/{commit}/{path}#L{s}-L{e}", "code": code, "html": None}
        if path.startswith("experiments/moon/"):
            entry["history_note"] = {"text": note["text"], "permalink": note["permalink"]}
        out.append(entry)
    return {"commit": commit, "repo": REPO, "excerpts": out}


# ---------------------------------------------------------------- main

def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    checkpoints: list[dict[str, Any]] = []
    with torch.no_grad():
        m = mars(written, checkpoints)
        mo = moon(written, checkpoints)
    stamp = {"generated_at": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds")}

    def put(name: str, payload: dict[str, Any], sources: list[Path]) -> None:
        p = OUT / name
        write_json(p, {"source": [rel(s) for s in sources], **stamp, **payload})
        written.append(p)

    put("checkpoints.json", {"checkpoints": checkpoints}, [])
    put("params.json", {"mars": m["params"], "moon": mo["params"]},
        [MARS_CNN_DIR / "param_counts.json", MARS_HYB_DIR / "param_counts.json", MOON_TRAIN_PY])
    put("mars_cnn_features.json", m["features"], [MARS_CNN_DIR / "best_checkpoint.pt", MARS_METADATA])
    put("curves.json", curves(), [MARS_RUNS, MOON_RUN / "logs"])
    ex = excerpts()
    old = OUT / "excerpts.json"
    if old.exists():  # keep highlight.mjs's html when the code or note is unchanged
        prev = {e["id"]: e for e in json.loads(old.read_text(encoding="utf-8")).get("excerpts", [])}
        for e in ex["excerpts"]:
            p = prev.get(e["id"])
            if p and p.get("code") == e["code"] and p.get("permalink") == e["permalink"]:
                e["html"] = p.get("html")
            if p and "history_note" in e and p.get("history_note", {}).get("text") == e["history_note"]["text"]:
                e["history_note"]["html"] = p["history_note"].get("html")
    put("excerpts.json", ex, [])

    total = 0
    for p in sorted(set(written)):
        n = p.stat().st_size
        total += n
        print(f"{p.relative_to(OUT.parent).as_posix():<44}{n:>10,}")
    lab_total = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file())
    print(f"{'lab/ total':<44}{lab_total:>10,}  (budget about 3 MB)")


if __name__ == "__main__":
    main()
