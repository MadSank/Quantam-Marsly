"""Parity tests: committed website/public/data JSON vs the source experiment artifacts.

Run from the repository root with the Streamlit app's venv:

    visualization_app/.venv/Scripts/python.exe -m pytest website/scripts/tests

Skipped when the gitignored experiment artifacts are not on disk.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))

import export_data as ex  # noqa: E402

pytestmark = pytest.mark.skipif(not ex.MARS_METRICS_DIR.exists() or not ex.MOON_LOGS_DIR.exists(),
                                reason="experiment artifacts (runs/, data/raw/) not present")

DATA = ex.OUT_DIR
TILE_TOL = 0.5 * 10 ** -ex.TILE_DECIMALS + 1e-9
PROB_TOL = 1e-4  # 4-decimal rounding of probabilities the app computes from saved logits


def load(rel: str) -> dict:
    def no_constants(name: str):
        raise ValueError(f"non-finite constant {name} in {rel}")
    return json.loads((DATA / rel).read_text(encoding="utf-8"), parse_constant=no_constants)


def same(a, b) -> bool:
    """Exact equality with NaN in the source matching null in the export."""
    if b is None:
        return a is None or (isinstance(a, float) and np.isnan(a))
    if isinstance(a, float) and np.isnan(a):
        return False
    return a == b


@pytest.fixture(scope="module")
def sites() -> list[dict]:
    return load("mars/sites.json")["sites"]


def test_manifest_lists_every_file():
    manifest = load("manifest.json")
    for rel in manifest["files"]:
        assert (DATA / rel).is_file(), rel
        load(rel)  # parses, no NaN/Infinity


def test_sites_cover_all_tiles(sites):
    meta = pd.read_parquet(ex.MARS_METADATA_PARQUET)
    assert sum(s["n_tiles"] for s in sites) == len(meta) == 15233
    assert {s["site_id"]: s["n_tiles"] for s in sites}["site_01"] == 2195
    for s in sites:
        g = meta[meta.site_id == s["site_id"]]
        assert s["n_true_safe"] == int(g.safe_label.sum())
        assert s["name"] == ex.SITE_NAMES[s["site_id"]]
        assert -90 <= s["center"]["lat"] <= 90 and 0 <= s["center"]["lon_e"] < 360


@pytest.mark.parametrize("regime,stem", list(ex.REGIMES.items()))
def test_headline_matches_aggregate_csv(regime, stem):
    exported = load("mars/headline.json")[regime]
    df = pd.read_csv(ex.MARS_METRICS_DIR / f"{stem}_aggregate.csv")
    for _, row in df.iterrows():
        cell = exported[row["model"]][row["metric"]]
        assert same(row["mean_site_metric"], cell["mean"])
        assert same(row["sd_across_sites"], cell["sd"])
        assert cell["n_sites"] == row["n_sites"]
        assert set(cell) == {"mean", "sd", "n_sites"}  # no interval fields


def test_mars_parameter_counts_match_runs():
    pc = load("mars/headline.json")["parameter_counts"]
    for model, run_dir in (("CNN", "model1_resnet18"), ("CNN_QCNN", "model2_qcnn")):
        for f in (ex.MARS_RUNS_DIR / run_dir).glob("fold_*_seed_*/param_counts.json"):
            assert json.loads(f.read_text(encoding="utf-8")) == pc[model], f
    assert pc["CNN"]["total_trainable"] == 11242563 and pc["CNN_QCNN"]["total_trainable"] == 4152


def test_no_interval_or_paired_fields_anywhere():
    for path in DATA.rglob("*.json"):
        text = path.read_text(encoding="utf-8")
        for banned in ("t95", "paired_differences", "Paired Site Differences"):
            assert banned not in text, f"{banned!r} in {path.name}"


@pytest.mark.parametrize("regime,stem", list(ex.REGIMES.items()))
def test_site_files_match_csvs(sites, regime, stem):
    mean_df = pd.read_csv(ex.MARS_METRICS_DIR / f"{stem}_site_mean.csv")
    seed_df = pd.read_csv(ex.MARS_METRICS_DIR / f"{stem}_per_site_seed.csv")
    for s in sites:
        sid = s["site_id"]
        exported = load(f"mars/sites/{sid}.json")
        for model in ex.MARS_MODELS:
            src = mean_df[(mean_df.model == model) & (mean_df.test_site == sid)].iloc[0]
            for col, val in src.items():
                if col not in ("model", "test_site"):
                    assert same(val, exported["mean"][regime][model][col]), (sid, model, col)
        rows = [r for r in exported["per_seed"] if r["regime"] == regime]
        src_rows = seed_df[seed_df.test_site == sid]
        assert len(rows) == len(src_rows) == 2 * len(ex.SEEDS)
        for r in rows:
            src = src_rows[(src_rows.model == r["model"]) & (src_rows.seed == r["seed"])].iloc[0]
            for col, val in src.items():
                if col != "test_site":
                    assert same(val, r[col]), (sid, r["model"], r["seed"], col)
        # fixed display order: CNN rows before CNN_QCNN rows
        assert [r["model"] for r in rows] == sorted((r["model"] for r in rows), key=ex.MARS_MODELS.index)


def test_tiles_match_predictions(sites):
    meta = pd.read_parquet(ex.MARS_METADATA_PARQUET).set_index("tile_id")
    for s in sites:
        tiles = load(f"mars/tiles/{s['site_id']}.json")
        n = s["n_tiles"]
        assert len(tiles["tile_id"]) == len(tiles["row"]) == len(tiles["col"]) == n
        m = meta.loc[tiles["tile_id"]]
        assert (m["row"].values == tiles["row"]).all() and (m["col"].values == tiles["col"]).all()
        assert (m["site_id"] == s["site_id"]).all()
        for model_dir, key in (("model1_resnet18", "cnn"), ("model2_qcnn", "qcnn")):
            for seed in ex.SEEDS:
                src = pd.read_parquet(ex.MARS_RUNS_DIR / model_dir / f"fold_{s['fold']}_seed_{seed}"
                                      / "test_predictions.parquet")
                assert src["tile_id"].tolist() == tiles["tile_id"]
                p = tiles["pred"][key][str(seed)]
                for src_col, k in (("pred_mean_slope_5m", "slope"), ("pred_roughness_m", "rough"),
                                   ("pred_hazard_fraction", "hazard")):
                    assert np.allclose(src[src_col].values, p[k], atol=TILE_TOL, rtol=0), (s["site_id"], key, seed, k)
                assert (src["pred_safe"].values == p["safe"]).all()
                assert (src["true_safe"].values == tiles["true"]["safe"]).all()
                assert np.allclose(src["true_mean_slope_5m"].values, tiles["true"]["slope"], atol=TILE_TOL, rtol=0)


def test_moon_summary_matches_sources():
    summary = load("moon/summary.json")
    recomputed = pd.read_csv(ex.MOON_RECOMPUTED_METRICS_CSV)
    for _, row in recomputed.iterrows():
        cell = summary["metrics"][row["model"]][row["metric"]]
        assert same(row["mean"], cell["mean"]) and same(row["sample_sd"], cell["sd"])
    assert summary["split_sizes"] == {"train": 3608, "val": 613, "test": 779}
    pc = summary["parameter_counts"]
    assert pc["cnn"]["trainable_during_training"] == 59347
    assert pc["cnn"]["trainable_recorded_post_freeze"] == 195
    assert (pc["hybrid_head"], pc["mlp_head"]) == (568, 547)
    assert summary["timing"]["comparable"] is False


def test_moon_summary_is_the_ten_seed_run():
    summary = load("moon/summary.json")
    assert summary["run"].endswith("runs_deterministic_10seed_20261028")
    assert summary["seeds"] == ex.moon_seeds() == list(range(42, 52))
    per_seed = pd.read_csv(ex.MOON_METRICS_CSV)
    assert len(summary["per_seed"]) == len(per_seed) == 40
    for row, cell in zip(per_seed.to_dict("records"), summary["per_seed"]):
        for k in ("seed", "model", "accuracy", "balanced_accuracy", "macro_f1"):
            assert same(row[k], cell[k])
    for model, g in per_seed.groupby("model"):
        for k in ("accuracy", "balanced_accuracy", "macro_f1"):
            assert same(g[k].min(), summary["ranges"][model][k]["min"])
            assert same(g[k].max(), summary["ranges"][model][k]["max"])


def test_moon_export_has_no_paired_statistics():
    text = (ex.OUT_DIR / "moon" / "summary.json").read_text(encoding="utf-8").lower()
    assert "paired" not in text and "diff" not in text


def test_moon_crops_match_results():
    classes = load("moon/summary.json")["classes"]
    seeds = ex.moon_seeds()
    results = {seed: json.loads((ex.MOON_LOGS_DIR / f"seed_{seed}" / "results.json").read_text(encoding="utf-8"))
               for seed in seeds}
    test = results[seeds[0]]["test"]
    truth = dict(zip(test["sample_ids"], test["ground_truth"]))
    index = {sid: i for i, sid in enumerate(test["sample_ids"])}
    total = 0
    for cls in classes:
        crops = load(f"moon/classes/{cls}.json")["crops"]
        total += len(crops)
        assert [c["img"] for c in crops] == list(range(len(crops)))
        for c in crops:
            assert classes[truth[c["sample_id"]]] == cls == c["truth"]
            for model in ex.MOON_MODELS:
                for seed in seeds:
                    p = c["pred"][model][str(seed)]
                    logits = np.array(results[seed]["test"]["logits"][model][index[c["sample_id"]]])
                    probs = np.exp(logits - logits.max())
                    probs /= probs.sum()
                    assert np.allclose(probs, p["probs"], atol=PROB_TOL, rtol=0)
                    assert 0 <= p["conf"] <= 1 and abs(p["conf"] - probs.max()) <= PROB_TOL
                    saved = results[seed]["test"]["predictions"][model][index[c["sample_id"]]]
                    assert p["label"] == classes[saved] == classes[int(np.argmax(probs))]
    assert total == 779


def test_caveats_are_verbatim():
    caveats = load("caveats.json")["caveats"]
    assert [c["id"] for c in caveats] == [c["id"] for c in ex.CAVEATS]
    sources = {c["source"]: ex.normalise((ex.REPO_ROOT / c["source"]).read_text(encoding="utf-8")) for c in caveats}
    for c, spec in zip(caveats, ex.CAVEATS):
        for part in spec.get("parts", [spec.get("text")]):
            assert ex.normalise(part) in sources[c["source"]], c["id"]
        assert "cross zero" not in c["text"] and "t interval" not in c["text"]
