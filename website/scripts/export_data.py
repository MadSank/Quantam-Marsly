"""Export the website's JSON data from the saved experiment artifacts.

Run from the repository root with the Streamlit app's venv:

    visualization_app/.venv/Scripts/python.exe website/scripts/export_data.py

Reads results through the app's existing ``data_loading`` code (read-only) and
writes small JSON files to ``website/public/data``. See website/PLAN.md for the
schema. Paired differences and interval columns are deliberately not exported.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
APP_DIR = REPO_ROOT / "visualization_app"
OUT_DIR = REPO_ROOT / "website" / "public" / "data"
SCHEMA_VERSION = 1

sys.path.insert(0, str(APP_DIR))
# Silence Streamlit's bare-mode warnings (cache decorators used outside a running app).
os.environ.setdefault("STREAMLIT_LOGGER_LEVEL", "error")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from data_loading import mars_data, moon_data  # noqa: E402
from stable_io import write_json  # noqa: E402

for _name in list(logging.root.manager.loggerDict):
    if _name.startswith("streamlit"):
        logging.getLogger(_name).setLevel(logging.ERROR)
from data_loading.paths import (  # noqa: E402
    MARS_METRICS_DIR,
    MARS_RUNS_DIR,
    MARS_METADATA_PARQUET,
    MOON_EXPERIMENT_ROOT,
    MOON_PROVENANCE_MANIFEST,
)

SEEDS = [42, 43, 44]  # Mars seeds; the Moon seed list comes from its run (moon_seeds)
SITE_NAMES = {
    "site_01": "Northern Mid-Latitude Crater",
    "site_02": "Kaiser Crater",
    "site_03": "Layered Yardangs",
    "site_04": "Coprates Chasma",
    "site_05": "Huo Hsing Vallis",
}
MARS_MODELS = ["CNN", "CNN_QCNN"]  # fixed display order: CNN first
REGIMES = {"fixed": "fixed_threshold", "validation_calibrated": "validation_calibrated_test"}
# Interval columns from the aggregate CSVs are never exported (removed statistical section).
DROPPED_AGGREGATE_COLUMNS = {"t95_low", "t95_high"}
TILE_DECIMALS = 3
MOON_MODELS = ["cnn", "hybrid_qcnn", "frozen_embedding_mlp"]
# Reported Moon results: the deterministic 10-seed run (FINAL_READINESS_AUDIT.md, Addendum 2026-10-28).
# The Streamlit app still reads the superseded three-seed run; this export reads the new folder directly.
MOON_RUN = MOON_EXPERIMENT_ROOT / "runs_deterministic_10seed_20261028"
MOON_LOGS_DIR = MOON_RUN / "logs"
MOON_METRICS_CSV = MOON_RUN / "metrics.csv"
MOON_RECOMPUTED_METRICS_CSV = MOON_RUN / "results_verification" / "recomputed_metrics.csv"
MOON_SUMMARY_JSON = MOON_RUN / "summary.json"

AUDIT = REPO_ROOT / "FINAL_READINESS_AUDIT.md"
APP_README = APP_DIR / "README.md"

# Every caveat is a verbatim quote (or verbatim "parts" joined with a space); export fails
# if any quoted text is not found in its source.
CAVEATS: list[dict[str, Any]] = [
    {"id": "safe_proxy", "world": "mars", "title": "“Safe” is a proxy label", "source": "audit",
     "text": "“Safe” is a derived proxy label: mean slope <8° and hazard fraction <0.05. "
             "It is not an observed landing outcome or validated safety assessment."},
    # Two separate audit sentences (the interval sentences between them are deliberately omitted).
    {"id": "mars_mixed_results", "world": "mars", "title": "Mixed results", "source": "audit",
     "parts": ["Numerically, the QCNN reduces the reported regression errors and false-positive rate, "
               "but the CNN has higher safe recall and coverage.",
               "These results therefore do not identify a clear overall winner."]},
    {"id": "mars_pipeline_comparison", "world": "mars", "title": "Pipeline comparison, not equal capacity",
     "source": "audit",
     "text": "Thus the real-data experiment compares a full end-to-end CNN with a small head on a supervised "
             "frozen representation; it is a pipeline comparison, not an architecture-only or "
             "equal-capacity comparison."},
    {"id": "mars_no_matched_control", "world": "mars", "title": "No matched classical head", "source": "audit",
     "text": "QCNN consumes a frozen supervised CNN representation and has far fewer trainable parameters. "
             "No real-data matched classical-head control is present in this primary Mars comparison, "
             "so numerical differences cannot be attributed specifically to quantum computation."},
    {"id": "mars_sites", "world": "mars", "title": "Five sites are the replication units", "source": "audit",
     "text": "Five sites are the geographic replication units. Tiles within a site are not independent "
             "site-level observations. Three random seeds do not create more independent sites."},
    {"id": "mars_terrain_proxy", "world": "mars", "title": "Terrain-proxy results", "source": "audit",
     "text": "These are terrain-proxy results, not evidence for operational landing safety."},
    {"id": "no_quantum_advantage", "world": "both", "title": "No quantum advantage claimed", "source": "app_readme",
     "text": "All quantum circuits were simulated numerically on classical hardware. "
             "No quantum advantage or superiority is claimed."},
    {"id": "moon_param_count", "world": "moon", "title": "Corrected Moon CNN parameter count", "source": "audit",
     "text": "The independent audit correctly establishes 59,347 trainable parameters during CNN fitting, "
             "with the 195-parameter classifier remaining trainable after encoder freeze."},
    {"id": "moon_timing", "world": "moon", "title": "Timings are not comparable", "source": "audit",
     "text": "CNN timing includes image-encoder inference, while recorded QCNN timing is for the head on "
             "already-computed embeddings. It excludes the encoder and is not an apples-to-apples "
             "end-to-end latency comparison."},
    {"id": "moon_imbalance", "world": "moon", "title": "Class imbalance", "source": "audit",
     "text": "The majority baseline’s 68.42% test accuracy is correct but predicts no minority classes "
             "(balanced accuracy 0.333, macro-F1 0.271). Accuracy alone is therefore misleading."},
    {"id": "moon_training_regime", "world": "moon", "title": "Different training regimes", "source": "audit",
     "text": "This can support a frozen-embedding head comparison, not a clean end-to-end "
             "CNN-versus-QCNN attribution."},
    {"id": "moon_mixed_results", "world": "moon", "title": "Descriptive evidence only", "source": "audit",
     "text": "Ten seeds on one supplied split are descriptive evidence, not proof of a general advantage."},
    {"id": "moon_seeds", "world": "moon", "title": "Seeds are not independent test samples", "source": "audit",
     "text": "The 10-seed mean ± SD is not a confidence interval. The same fixed test partition is reused "
             "across seeds, so seeds reflect initialization/training variation, not independent test samples."},
    {"id": "moon_provenance", "world": "moon", "title": "Unverified provenance", "source": "audit",
     "text": "The provenance audit did not verify class annotation criteria, original split-generation "
             "methodology, source-image identity, or geographic/site identity."},
    {"id": "moon_reproducibility", "world": "moon", "title": "Ten deterministic seeds; no model clearly ahead",
     "source": "audit",
     "parts": ["The Moon results come from 10 deterministic seeds on one supplied split.",
               "The spread across seeds is large, and the ranges of the three models overlap almost entirely, "
               "so no model is clearly ahead.",
               "The frozen-embedding MLP control matches or exceeds the hybrid QCNN head on all three mean metrics, "
               "so the hybrid's results do not indicate any quantum contribution."]},
    {"id": "moon_test_embeddings", "world": "moon", "title": "Test data stays locked until evaluation",
     "source": "audit",
     "text": "In this run, test data is loaded and embedded only after every model for a seed has been fitted "
             "and selected."},
]
CAVEAT_SOURCES = {"audit": AUDIT, "app_readme": APP_README}


# ---------------------------------------------------------------- helpers

def rel(path: Path) -> str:
    return Path(path).resolve().relative_to(REPO_ROOT).as_posix()


def clean(value: Any) -> Any:
    """JSON-safe scalar: NaN/inf -> None, numpy -> Python."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        f = float(value)
        return None if not math.isfinite(f) else f
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def rounded(values: Any, decimals: int = TILE_DECIMALS) -> list[float | None]:
    return [None if not math.isfinite(float(v)) else round(float(v), decimals) for v in values]


def stamp(sources: list[Path]) -> dict[str, Any]:
    return {"source": [rel(p) for p in sources], "generated_at": GENERATED_AT}


GENERATED_AT = datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def image_fields() -> dict[str, Any]:
    """Image fields written by export_images.py, kept when this script re-runs."""
    path = OUT_DIR / "img" / "manifest.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"mars": {}, "moon": {}}


# ---------------------------------------------------------------- Mars

def site_center(group: pd.DataFrame) -> dict[str, float]:
    """Centre of a site's tile footprint, from its equirectangular CRS and affine transform."""
    first = group.iloc[0]
    wkt = first["crs_wkt"]
    lat_ts = float(re.search(r'standard_parallel_1",([-\d.]+)', wkt).group(1))
    lon0 = float(re.search(r'central_meridian",([-\d.]+)', wkt).group(1))
    radius = float(re.search(r'SPHEROID\["[^"]*",([\d.]+)', wkt).group(1))
    a, _, c, _, e, f = list(first["transform"])  # row["transform"]: attribute access hits DataFrame.transform
    half = float(first["native_window_px"]) / 2.0
    row = (group["row"].min() + group["row"].max()) / 2.0 + half
    col = (group["col"].min() + group["col"].max()) / 2.0 + half
    x, y = c + a * col, f + e * row
    lat = math.degrees(y / radius)
    lon = lon0 + math.degrees(x / (radius * math.cos(math.radians(lat_ts))))
    return {"lat": round(lat, 4), "lon_e": round(lon % 360.0, 4)}


def _projection(group: pd.DataFrame):
    """(row, col) in native pixels -> (lat, lon_e) degrees, from the site's equirectangular CRS."""
    first = group.iloc[0]
    wkt = first["crs_wkt"]
    lat_ts = float(re.search(r'standard_parallel_1",([-\d.]+)', wkt).group(1))
    lon0 = float(re.search(r'central_meridian",([-\d.]+)', wkt).group(1))
    radius = float(re.search(r'SPHEROID\["[^"]*",([\d.]+)', wkt).group(1))
    a, _, c, _, e, f = list(first["transform"])

    def to_lat_lon(row: np.ndarray, col: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x, y = c + a * col, f + e * row
        lat = np.degrees(y / radius)
        lon = lon0 + np.degrees(x / (radius * math.cos(math.radians(lat_ts))))
        return lat, np.mod(lon, 360.0)

    return to_lat_lon


def _convex_hull(points: np.ndarray) -> np.ndarray:
    """Andrew's monotone chain; points (n, 2) -> hull vertices, counter-clockwise, no repeat."""
    pts = np.unique(points, axis=0)
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list = []
    for q in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], q) <= 0:
            lower.pop()
        lower.append(q)
    upper: list = []
    for q in pts[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], q) <= 0:
            upper.pop()
        upper.append(q)
    return np.asarray(lower[:-1] + upper[:-1])


def site_footprint(group: pd.DataFrame) -> list[list[float]]:
    """Convex hull of every tile's four corners, as [lat, lon_e] pairs (globe strip outline)."""
    w = float(group.iloc[0]["native_window_px"])
    rows, cols = group["row"].to_numpy(float), group["col"].to_numpy(float)
    corners = np.concatenate([np.column_stack([rows + dr, cols + dc]) for dr in (0, w) for dc in (0, w)])
    hull = _convex_hull(corners)
    lat, lon = _projection(group)(hull[:, 0], hull[:, 1])
    return [[round(float(la), 5), round(float(lo), 5)] for la, lo in zip(lat, lon)]


def image_corners(group: pd.DataFrame, height: int, width: int) -> dict[str, list[float]]:
    """[lat, lon_e] of the native ortho grid's top-left, top-right and bottom-left pixel corners,
    so the preview (stretched over the native grid) can be placed on the globe."""
    to_lat_lon = _projection(group)
    rows = np.array([0.0, 0.0, float(height)])
    cols = np.array([0.0, float(width), 0.0])
    lat, lon = to_lat_lon(rows, cols)
    return {k: [round(float(lat[i]), 6), round(float(lon[i]), 6)] for i, k in enumerate(("top_left", "top_right", "bottom_left"))}


def export_mars(written: list[Path]) -> None:
    meta = mars_data.load_mars_metadata()
    dataset_meta = {s["site_id"]: s for s in mars_data.load_mars_dataset_metadata()["sites"]}
    fold_of = {v["test"]: k for k, v in mars_data.FOLD_SITE_MAPPING.items()}
    images = image_fields()["mars"]

    # sites.json (preview/thumb come from export_images.py's img/manifest.json)
    sites = []
    for site_id, group in meta.groupby("site_id", sort=True):
        windows = group["native_window_px"].unique()
        assert len(windows) == 1, f"{site_id}: mixed tile sizes {windows}"
        first = group.iloc[0]
        sites.append({
            "site_id": site_id,
            "name": SITE_NAMES[site_id],
            "fold": int(fold_of[site_id]),
            "ortho_product_id": first["ortho_product_id"],
            "dtm_product_id": first["dtm_product_id"],
            "center": site_center(group),
            "footprint": site_footprint(group),
            "n_tiles": int(len(group)),
            "n_true_safe": int(group["safe_label"].sum()),
            "native_resolution_m": clean(dataset_meta[site_id]["native_resolution_m"]),
            # dataset_metadata "dimensions" is [rows, cols] of the native ortho/DTM grid.
            "native_height_px": int(dataset_meta[site_id]["dimensions"][0]),
            "native_width_px": int(dataset_meta[site_id]["dimensions"][1]),
            "image_corners": image_corners(group, *map(int, dataset_meta[site_id]["dimensions"][:2])),
            "tile_size_native_px": int(windows[0]),
            "preview": images.get(site_id, {}).get("preview"),
            "thumb": images.get(site_id, {}).get("thumb"),
        })
    path = OUT_DIR / "mars" / "sites.json"
    write_json(path, {**stamp([MARS_METADATA_PARQUET]), "sites": sites})
    written.append(path)

    # headline.json: five-site aggregates, both decision regimes
    headline: dict[str, Any] = {}
    sources = []
    for key, stem in REGIMES.items():
        src = MARS_METRICS_DIR / f"{stem}_aggregate.csv"
        sources.append(src)
        df = pd.read_csv(src)
        assert not (DROPPED_AGGREGATE_COLUMNS - set(df.columns)), "aggregate schema changed"
        headline[key] = {
            model: {
                row["metric"]: {"mean": clean(row["mean_site_metric"]), "sd": clean(row["sd_across_sites"]),
                                "n_sites": int(row["n_sites"])}
                for _, row in df[df["model"] == model].iterrows()
            }
            for model in MARS_MODELS
        }
    # Parameter counts are identical for every fold/seed run; verify, then export one copy.
    param_dirs = {"CNN": "model1_resnet18", "CNN_QCNN": "model2_qcnn"}
    param_counts: dict[str, Any] = {}
    for model, run_dir in param_dirs.items():
        files = sorted((MARS_RUNS_DIR / run_dir).glob("fold_*_seed_*/param_counts.json"))
        assert len(files) == 5 * len(SEEDS), f"{model}: expected 15 param_counts.json, found {len(files)}"
        counts = [json.loads(f.read_text(encoding="utf-8")) for f in files]
        assert all(c == counts[0] for c in counts), f"{model}: parameter counts differ between runs"
        param_counts[model] = counts[0]
        sources.append(files[0])
    path = OUT_DIR / "mars" / "headline.json"
    write_json(path, {**stamp(sources), "models": MARS_MODELS, "parameter_counts": param_counts, **headline})
    written.append(path)

    # site_<id>.json: per-site means and per-seed rows, both regimes
    site_mean = {k: pd.read_csv(MARS_METRICS_DIR / f"{s}_site_mean.csv") for k, s in REGIMES.items()}
    per_seed = {k: pd.read_csv(MARS_METRICS_DIR / f"{s}_per_site_seed.csv") for k, s in REGIMES.items()}
    for site in sites:
        sid = site["site_id"]
        mean = {
            k: {m: {c: clean(v) for c, v in df[(df["model"] == m) & (df["test_site"] == sid)].iloc[0].items()
                    if c not in ("model", "test_site")}
                for m in MARS_MODELS}
            for k, df in site_mean.items()
        }
        rows = []
        for k, df in per_seed.items():
            sub = df[df["test_site"] == sid].sort_values(["model", "seed"], key=lambda s: s.map(
                {m: i for i, m in enumerate(MARS_MODELS)}) if s.name == "model" else s)
            for _, r in sub.iterrows():
                rows.append({"regime": k, **{c: clean(v) for c, v in r.items() if c != "test_site"}})
        srcs = [MARS_METRICS_DIR / f"{s}_{kind}.csv" for s in REGIMES.values() for kind in ("site_mean", "per_site_seed")]
        path = OUT_DIR / "mars" / "sites" / f"{sid}.json"
        write_json(path, {**stamp(srcs), "site_id": sid, "models": MARS_MODELS, "mean": mean, "per_seed": rows})
        written.append(path)

    # tiles_<id>.json: columnar per-tile truth and predictions for every seed
    for site in sites:
        sid, fold = site["site_id"], site["fold"]
        frames = {seed: mars_data.load_mars_predictions(fold, seed) for seed in SEEDS}
        base = frames[SEEDS[0]]
        for seed, df in frames.items():
            assert (df["tile_id"].values == base["tile_id"].values).all(), f"{sid} seed {seed}: tile order differs"
            for col in ("true_mean_slope_5m", "true_roughness_m", "true_hazard_fraction", "true_safe"):
                assert np.array_equal(df[col].values, base[col].values), f"{sid} seed {seed}: {col} differs"
        assert len(base) == site["n_tiles"], f"{sid}: {len(base)} predictions vs {site['n_tiles']} tiles"

        def pred(df: pd.DataFrame, prefix: str) -> dict[str, list]:
            return {
                "slope": rounded(df[f"{prefix}_pred_mean_slope_5m"]),
                "rough": rounded(df[f"{prefix}_pred_roughness_m"]),
                "hazard": rounded(df[f"{prefix}_pred_hazard_fraction"]),
                "safe": [int(v) for v in df[f"{prefix}_pred_safe"]],
            }

        payload = {
            **stamp([MARS_RUNS_DIR / f"{d}/fold_{fold}_seed_{s}/test_predictions.parquet"
                     for d in ("model1_resnet18", "model2_qcnn") for s in SEEDS] + [MARS_METADATA_PARQUET]),
            "site_id": sid,
            "fold": fold,
            "seeds": SEEDS,
            "size_native": site["tile_size_native_px"],
            "tile_id": base["tile_id"].tolist(),
            "row": [int(v) for v in base["row"]],
            "col": [int(v) for v in base["col"]],
            "true": {
                "slope": rounded(base["true_mean_slope_5m"]),
                "rough": rounded(base["true_roughness_m"]),
                "hazard": rounded(base["true_hazard_fraction"]),
                "safe": [int(v) for v in base["true_safe"]],
            },
            "pred": {
                "cnn": {str(s): pred(frames[s], "cnn") for s in SEEDS},
                "qcnn": {str(s): pred(frames[s], "qcnn") for s in SEEDS},
            },
        }
        path = OUT_DIR / "mars" / "tiles" / f"{sid}.json"
        write_json(path, payload, compact=True)
        written.append(path)


# ---------------------------------------------------------------- Moon

def moon_seeds() -> list[int]:
    """The seed list from the run's own summary, in run order."""
    return [int(s) for s in json.loads(MOON_SUMMARY_JSON.read_text(encoding="utf-8"))["config"]["seeds"]]


def load_moon_seed(seed: int) -> dict[str, Any]:
    return json.loads((MOON_LOGS_DIR / f"seed_{seed}" / "results.json").read_text(encoding="utf-8"))


def softmax_rows(logits: np.ndarray) -> np.ndarray:
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def export_moon(written: list[Path]) -> None:
    seeds = moon_seeds()
    results = {seed: load_moon_seed(seed) for seed in seeds}
    base = results[seeds[0]]
    for seed, r in results.items():
        assert r["test"]["sample_ids"] == base["test"]["sample_ids"], f"seed {seed}: test order differs"
        assert r["test"]["ground_truth"] == base["test"]["ground_truth"], f"seed {seed}: labels differ"
        assert r["parameter_counts"] == base["parameter_counts"], f"seed {seed}: parameter counts differ"

    classes = list(moon_data.CLASS_NAMES)
    provenance = moon_data.load_moon_provenance()
    split_counts = {
        split: {c: int(((provenance["existing_split"] == split) & (provenance["class_label"] == c)).sum())
                for c in classes}
        for split in ("train", "val", "test")
    }
    assert [split_counts["test"][c] for c in classes] == base["test"]["class_distribution"]

    recomputed = pd.read_csv(MOON_RECOMPUTED_METRICS_CSV)
    models = {
        m: {row["metric"]: {"mean": clean(row["mean"]), "sd": clean(row["sample_sd"])}
            for _, row in recomputed[recomputed["model"] == m].iterrows()}
        for m in MOON_MODELS + ["majority_train_baseline"]
    }
    # Per-seed values for the results panel. Only per-model values: no paired differences are exported.
    per_seed_df = pd.read_csv(MOON_METRICS_CSV)
    per_seed = [{c: clean(v) for c, v in r.items() if c not in ("test_class_counts",)}
                for _, r in per_seed_df.iterrows()]
    ranges = {
        m: {k: {"min": clean(g[k].min()), "max": clean(g[k].max())} for k in ("accuracy", "balanced_accuracy", "macro_f1")}
        for m, g in per_seed_df.groupby("model", sort=False)
    }

    pc = base["parameter_counts"]
    summary = {
        **stamp([MOON_RECOMPUTED_METRICS_CSV, MOON_METRICS_CSV, MOON_SUMMARY_JSON, MOON_PROVENANCE_MANIFEST]
                + [MOON_LOGS_DIR / f"seed_{s}" / "results.json" for s in seeds]),
        "run": rel(MOON_RUN),
        "seeds": seeds,
        "classes": classes,
        "models": MOON_MODELS,
        "split_counts": split_counts,
        "split_sizes": {k: sum(v.values()) for k, v in split_counts.items()},
        "metrics": models,
        "ranges": ranges,
        "per_seed": per_seed,
        "parameter_counts": {
            "cnn": {
                "trainable_during_training": int(pc["cnn_during_fit"]["trainable"]),
                "trainable_recorded_post_freeze": int(pc["cnn_classifier_after_encoder_freeze"]["trainable"]),
                "note": "59,347 parameters are trainable while the CNN is fitted. After the encoder is frozen "
                        "for the head stage, only its 195-parameter classifier remains trainable. "
                        "The run records both counts.",
            },
            "hybrid_head": int(pc["hybrid_head"]["trainable"]),
            "mlp_head": int(pc["frozen_mlp_head"]["trainable"]),
        },
        "timing": {
            "comparable": False,
            "caveat_id": "moon_timing",
            "milliseconds_per_image": {
                m: {str(s): clean(results[s]["test"]["inference"][m]["milliseconds_per_image"]) for s in seeds}
                for m in MOON_MODELS
            },
        },
    }
    path = OUT_DIR / "moon" / "summary.json"
    write_json(path, summary)
    written.append(path)

    index = {sid: i for i, sid in enumerate(base["test"]["sample_ids"])}
    probs = {seed: {m: softmax_rows(np.asarray(results[seed]["test"]["logits"][m], dtype=np.float64)) for m in MOON_MODELS}
             for seed in seeds}
    images = image_fields()["moon"]
    for cls in classes:
        sub_ids = sorted(s for s, t in zip(base["test"]["sample_ids"], base["test"]["ground_truth"])
                         if moon_data.CLASS_IDX_TO_NAME[t] == cls)
        crops = []
        for i, sample_id in enumerate(sub_ids):
            j = index[sample_id]
            entry: dict[str, Any] = {"sample_id": sample_id, "img": i, "truth": cls, "pred": {}}
            for model in MOON_MODELS:
                entry["pred"][model] = {}
                for seed in seeds:
                    pr = probs[seed][model][j]
                    label = int(results[seed]["test"]["predictions"][model][j])
                    assert label == int(np.argmax(pr)), f"{sample_id} {model} {seed}: prediction is not the argmax"
                    entry["pred"][model][str(seed)] = {"label": moon_data.CLASS_IDX_TO_NAME[label],
                                                      "conf": round(float(pr.max()), 4),
                                                      "probs": [round(float(v), 4) for v in pr]}
            crops.append(entry)
        path = OUT_DIR / "moon" / "classes" / f"{cls}.json"
        write_json(path, {**stamp([MOON_LOGS_DIR / f"seed_{s}" / "results.json" for s in seeds]),
                          "class": cls, "classes": classes, "seeds": seeds,
                          "sprite_order": "crops[i].img is the cell index in this class's sprite sheet",
                          "sprite": images.get(cls, {}).get("sprite"), "thumb": images.get(cls, {}).get("thumb"),
                          "crops": crops},
                   compact=True)
        written.append(path)


# ---------------------------------------------------------------- shared

def normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("**", "")).strip()


def export_caveats(written: list[Path]) -> None:
    texts = {k: normalise(p.read_text(encoding="utf-8")) for k, p in CAVEAT_SOURCES.items()}
    out = []
    for c in CAVEATS:
        parts = c.get("parts", [c.get("text")])
        for part in parts:
            if normalise(part) not in texts[c["source"]]:
                raise ValueError(f"Caveat {c['id']!r} is not a verbatim quote of {CAVEAT_SOURCES[c['source']].name}")
        entry = {k: v for k, v in c.items() if k != "parts"}
        entry.update(text=" ".join(parts), source=rel(CAVEAT_SOURCES[c["source"]]), quote_parts=len(parts))
        out.append(entry)
    path = OUT_DIR / "caveats.json"
    write_json(path, {**stamp(list(CAVEAT_SOURCES.values())), "caveats": out})
    written.append(path)


# The About page's method text: facts restated from FINAL_READINESS_AUDIT.md (section 1 and the
# Addendum) and the run configurations. No result figures: those live in the explorers.
ABOUT_METHOD = [
    {"id": "mars_task", "world": "mars", "title": "Mars: terrain proxies from HiRISE",
     "paragraphs": [
         "Five HiRISE sites, each an orthoimage strip with its digital terrain model, are cut into 15,233 "
         "coverage-eligible patches at 2 m per pixel.",
         "Each patch has three terrain proxy targets taken from the terrain model: 5 m mean slope in degrees, "
         "roughness in metres and hazard fraction. A patch counts as safe, a derived proxy label, when its mean "
         "slope is under 8 degrees and its hazard fraction is under 0.05. It is not an observed landing outcome.",
     ]},
    {"id": "mars_models", "world": "mars", "title": "Mars: the two models",
     "paragraphs": [
         "The CNN is a ResNet-18 trained end to end from random initialisation, with 11,242,563 trainable parameters.",
         "The hybrid keeps that trained CNN representation frozen and fits an 8-dimensional bottleneck and an "
         "8-qubit QCNN head on it, 4,152 trainable parameters in all. The comparison is between two pipelines, "
         "not two architectures of equal capacity.",
     ]},
    {"id": "mars_evaluation", "world": "mars", "title": "Mars: how it was tested",
     "paragraphs": [
         "Leave one site out: in each of five folds, three sites train, one validates and one is the test site. "
         "Every fold was run with seeds 42, 43 and 44.",
         "Safe decisions are scored two ways: at the fixed proxy thresholds, and at a risk threshold chosen on the "
         "validation site so that its false-positive rate is at most 5%, then applied to the test site.",
     ]},
    {"id": "moon_task", "world": "moon", "title": "Moon: crater classes from LROCNet",
     "paragraphs": [
         "LROCNet crops from the LRO Narrow Angle Camera are labelled ejecta, old crater or none. The supplied split "
         "has 3,608 training, 613 validation and 779 test crops. The test set is 68.4% none, so balanced accuracy "
         "and macro-F1 are reported alongside accuracy.",
     ]},
    {"id": "moon_models", "world": "moon", "title": "Moon: three models",
     "paragraphs": [
         "The CNN is trained end to end, with 59,347 trainable parameters. The hybrid QCNN head (568 trainable "
         "parameters) and a frozen-embedding MLP control (547) are both trained on the CNN's frozen embeddings, "
         "so the MLP is the matched classical comparison for the quantum head.",
         "The reported results come from 10 deterministic seeds, 42 to 51, which supersede an earlier three-seed run "
         "whose head initialisation could not be reproduced.",
     ]},
    {"id": "simulation", "world": "both", "title": "Quantum circuits were simulated",
     "paragraphs": [
         "Every quantum circuit ran on a classical simulator (PennyLane default.qubit). Nothing here ran on quantum "
         "hardware, and no quantum advantage is claimed.",
     ]},
    {"id": "site", "world": "both", "title": "What this site shows",
     "paragraphs": [
         "Every figure, prediction and label on this site comes from saved experiment outputs. Nothing is retrained "
         "or recomputed in the browser, and the numbers match the project's readiness audit.",
     ]},
]

ABOUT_SOURCES = [
    {"name": "HiRISE orthoimages and digital terrain models",
     "provider": "NASA/JPL/University of Arizona",
     "licence": "Public domain under the HiRISE image usage policy: no restrictions on use; credit requested.",
     "credit": "NASA/JPL/University of Arizona",
     "citation": "McEwen, A. S., et al. (2007). Mars Reconnaissance Orbiter's High Resolution Imaging Science "
                 "Experiment (HiRISE). Journal of Geophysical Research: Planets, 112, E05S02. doi:10.1029/2005JE002605",
     "url": "https://www.uahirise.org/",
     "licence_url": "https://uahirise.org/media/usage.php",
     "used_for": "Mars strips, terrain targets and tiles; the strip previews and contour lines"},
    {"name": "LROCNet training imagery",
     "provider": "NASA Jet Propulsion Laboratory, California Institute of Technology (Emily Dunkel)",
     "licence": "Creative Commons Attribution 4.0 International (CC BY 4.0).",
     "credit": "Dunkel, NASA JPL, California Institute of Technology; imagery from the LRO Narrow Angle Camera",
     "citation": "Dunkel, E. (2022). Lunar Reconnaissance Orbiter Imagery for LROCNet Moon Classifier (v1.0.0) "
                 "[Data set]. Zenodo. doi:10.5281/zenodo.7041842",
     "url": "https://zenodo.org/records/7041842",
     "licence_url": "https://creativecommons.org/licenses/by/4.0/",
     "used_for": "Moon crater classification crops. Shown resized to 128 px sprite sheets; no other changes."},
    {"name": "Mars Viking MDIM 2.1 colorized global mosaic",
     "provider": "NASA Ames / USGS Astrogeology Science Center",
     "licence": "Public domain (USGS catalog: access constraints public domain, use constraints none).",
     "credit": "NASA Ames / USGS Astrogeology Science Center",
     "citation": "USGS Astrogeology Science Center (2009). Mars Viking Colorized Global Mosaic 232m. Originator: NASA Ames.",
     "url": "https://astrogeology.usgs.gov/search/map/mars_viking_colorized_global_mosaic_232m",
     "licence_url": "https://astrogeology.usgs.gov/search/map/mars_viking_colorized_global_mosaic_232m",
     "used_for": "Mars globe and flat-map colour texture (1 km/px JPG, downscaled)"},
    {"name": "MGS MOLA Mission Experiment Gridded Data Record (topography, 16 px/deg)",
     "provider": "NASA Goddard Space Flight Center, MGS MOLA team, via the PDS Geosciences Node",
     "licence": "NASA data, generally not subject to copyright in the United States; NASA credited as the source.",
     "credit": "NASA / GSFC / MOLA Science Team",
     "citation": "Smith, D. E., Zuber, M. T., Neumann, G. A., Guinness, E. A., & Slavney, S. (2003). Mars Global "
                 "Surveyor Laser Altimeter Mission Experiment Gridded Data Record, MGS-M-MOLA-5-MEGDR-L3-V1.0. "
                 "NASA Planetary Data System. doi:10.17189/1519460",
     "url": "https://pds-geosciences.wustl.edu/mgs/mgs-m-mola-5-megdr-l3-v1/mgsl_300x/meg016/",
     "licence_url": "https://www.nasa.gov/nasa-brand-center/images-and-media/",
     "used_for": "Mars globe relief (product MEGT90N000EB, turned into a 1k normal map)"},
    {"name": "CGI Moon Kit (LRO LROC WAC colour mosaic)",
     "provider": "NASA's Scientific Visualization Studio",
     "licence": "Public domain (NASA SVS: all content is public domain unless otherwise noted).",
     "credit": "NASA's Scientific Visualization Studio",
     "citation": "NASA's Scientific Visualization Studio, CGI Moon Kit (lroc_color_poles_4k.tif).",
     "url": "https://svs.gsfc.nasa.gov/4720",
     "licence_url": "https://svs.gsfc.nasa.gov/help/",
     "used_for": "Moon globe and flat-map colour texture (4096x2048 TIF, downscaled)"},
    {"name": "LRO LOLA Gridded Data Record (topography, 16 px/deg)",
     "provider": "NASA Goddard Space Flight Center, LRO LOLA team, via the PDS Geosciences Node",
     "licence": "NASA data, generally not subject to copyright in the United States; NASA credited as the source.",
     "credit": "NASA / GSFC / LOLA Science Team",
     "citation": "LRO LOLA Team (producer D. E. Smith, NASA Goddard Space Flight Center). Lunar Orbiter Laser Altimeter "
                 "Gridded Data Record, LRO-L-LOLA-4-GDR-V1.0, product LDEM_16, version 3.1 (2019-03-15). "
                 "NASA Planetary Data System, PDS Geosciences Node.",
     "url": "https://pds-geosciences.wustl.edu/lro/lro-l-lola-3-rdr-v1/lrolol_1xxx/data/lola_gdr/cylindrical/img/",
     "licence_url": "https://www.nasa.gov/nasa-brand-center/images-and-media/",
     "used_for": "Moon globe relief (product LDEM_16, turned into a 1k normal map)"},
]


def export_about(written: list[Path]) -> None:
    """About page content: method (restated from the audit) and data sources with licence and citation."""
    about = {
        **stamp([AUDIT]),
        "method": ABOUT_METHOD,
        "sources": ABOUT_SOURCES,
    }
    path = OUT_DIR / "about.json"
    write_json(path, about)
    written.append(path)


def size_report(written: list[Path]) -> None:
    total = 0
    print(f"\n{'file':<48}{'bytes':>12}")
    for p in sorted(written):
        n = p.stat().st_size
        total += n
        print(f"{p.relative_to(OUT_DIR).as_posix():<48}{n:>12,}")
    print(f"{'total (this script)':<48}{total:>12,}")
    everything = sum(p.stat().st_size for p in OUT_DIR.rglob("*") if p.is_file())
    print(f"{'public/data total':<48}{everything:>12,}")


def main() -> None:
    written: list[Path] = []
    export_mars(written)
    export_moon(written)
    export_caveats(written)
    export_about(written)
    manifest = OUT_DIR / "manifest.json"
    write_json(manifest, {
        "schema_version": SCHEMA_VERSION,
        "generated_at": GENERATED_AT,
        "worlds": ["mars", "moon"],
        "files": sorted(p.relative_to(OUT_DIR).as_posix() for p in written),
    })
    written.append(manifest)
    size_report(written)


if __name__ == "__main__":
    main()
