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
    MOON_LOGS_DIR,
    MOON_PROVENANCE_MANIFEST,
    MOON_RECOMPUTED_METRICS_CSV,
    MOON_METRICS_CSV,
)

SEEDS = [42, 43, 44]
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
MOON_DF_PREFIX = {"cnn": "cnn", "hybrid_qcnn": "hybrid", "frozen_embedding_mlp": "mlp"}

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
     "text": "Three seeds on one supplied split are descriptive evidence, not proof of a general advantage."},
    {"id": "moon_seeds", "world": "moon", "title": "Seeds are not independent test samples", "source": "audit",
     "text": "The three-seed mean ± SD is not a confidence interval. The same fixed test partition is reused "
             "across seeds, so seeds reflect initialization/training variation, not independent test samples."},
    {"id": "moon_provenance", "world": "moon", "title": "Unverified provenance", "source": "audit",
     "text": "The provenance audit did not verify class annotation criteria, original split-generation "
             "methodology, source-image identity, or geographic/site identity."},
    {"id": "moon_test_embeddings", "world": "moon", "title": "Test-embedding timing in the completed run",
     "source": "audit",
     "text": "Test embeddings were computed before fitting the QCNN/MLP heads, but were not used for gradient "
             "updates, validation selection, or threshold selection; the saved audit confirms final metrics "
             "came from saved predictions."},
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
            "n_tiles": int(len(group)),
            "n_true_safe": int(group["safe_label"].sum()),
            "native_resolution_m": clean(dataset_meta[site_id]["native_resolution_m"]),
            # dataset_metadata "dimensions" is [rows, cols] of the native ortho/DTM grid.
            "native_height_px": int(dataset_meta[site_id]["dimensions"][0]),
            "native_width_px": int(dataset_meta[site_id]["dimensions"][1]),
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

def export_moon(written: list[Path]) -> None:
    results = {seed: moon_data.load_moon_results(seed) for seed in SEEDS}
    base = results[SEEDS[0]]
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

    recomputed = moon_data.load_moon_recomputed_metrics()
    models = {
        m: {row["metric"]: {"mean": clean(row["mean"]), "sd": clean(row["sample_sd"])}
            for _, row in recomputed[recomputed["model"] == m].iterrows()}
        for m in MOON_MODELS + ["majority_train_baseline"]
    }
    per_seed_df = moon_data.load_moon_metrics()
    per_seed = [{c: clean(v) for c, v in r.items() if c not in ("test_class_counts",)}
                for _, r in per_seed_df.iterrows()]

    pc = base["parameter_counts"]
    summary = {
        **stamp([MOON_RECOMPUTED_METRICS_CSV, MOON_METRICS_CSV, MOON_PROVENANCE_MANIFEST]
                + [MOON_LOGS_DIR / f"seed_{s}" / "results.json" for s in SEEDS]),
        "classes": classes,
        "models": MOON_MODELS,
        "split_counts": split_counts,
        "split_sizes": {k: sum(v.values()) for k, v in split_counts.items()},
        "metrics": models,
        "per_seed": per_seed,
        "parameter_counts": {
            "cnn": {
                "trainable_during_training": int(pc["cnn"]["trainable_during_training"]),
                "trainable_recorded_post_freeze": int(pc["cnn"]["trainable_recorded_post_freeze"]),
                "note": pc["cnn"]["correction_note"],
            },
            "hybrid_head": int(pc["hybrid_head"]["trainable"]),
            "mlp_head": int(pc["frozen_mlp_head"]["trainable"]),
        },
        "timing": {
            "comparable": False,
            "caveat_id": "moon_timing",
            "milliseconds_per_image": {
                m: {str(s): clean(results[s]["test"]["inference"][m]["milliseconds_per_image"]) for s in SEEDS}
                for m in MOON_MODELS
            },
        },
    }
    path = OUT_DIR / "moon" / "summary.json"
    write_json(path, summary)
    written.append(path)

    frames = {seed: moon_data.get_moon_predictions_df(seed) for seed in SEEDS}
    images = image_fields()["moon"]
    for cls in classes:
        sub_ids = sorted(s for s, t in zip(base["test"]["sample_ids"], base["test"]["ground_truth"])
                         if moon_data.CLASS_IDX_TO_NAME[t] == cls)
        crops = []
        for i, sample_id in enumerate(sub_ids):
            entry: dict[str, Any] = {"sample_id": sample_id, "img": i, "truth": cls, "pred": {}}
            for model in MOON_MODELS:
                p = MOON_DF_PREFIX[model]
                entry["pred"][model] = {}
                for seed in SEEDS:
                    row = frames[seed].set_index("sample_id").loc[sample_id]
                    probs = [round(float(row[f"{p}_prob_{c}"]), 4) for c in classes]
                    entry["pred"][model][str(seed)] = {"label": row[f"{p}_pred_name"],
                                                      "conf": round(float(row[f"{p}_confidence"]), 4),
                                                      "probs": probs}
            crops.append(entry)
        path = OUT_DIR / "moon" / "classes" / f"{cls}.json"
        write_json(path, {**stamp([MOON_LOGS_DIR / f"seed_{s}" / "results.json" for s in SEEDS]),
                          "class": cls, "classes": classes, "seeds": SEEDS,
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


def export_about(written: list[Path]) -> None:
    """Skeleton for the About page; licence/citation text is completed and confirmed in build step 10."""
    todo = "TO CONFIRM"
    about = {
        **stamp([AUDIT]),
        "method": [],
        "sources": [
            {"name": "HiRISE orthoimages and DTMs", "provider": "NASA/JPL/University of Arizona",
             "licence": todo, "citation": todo, "url": todo,
             "used_for": "Mars strips, terrain targets (slope, roughness, hazard) and tiles"},
            {"name": "LROCNet", "provider": todo, "licence": todo, "citation": todo, "url": todo,
             "used_for": "Moon crater classification crops (ejecta, oldcrater, none)"},
        ],
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
