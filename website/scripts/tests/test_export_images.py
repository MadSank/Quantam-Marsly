"""Checks on the exported web images: size budget, no raw rasters, geometry, tile alignment.

Run from the repository root with the Streamlit app's venv:

    visualization_app/.venv/Scripts/python.exe -m pytest website/scripts/tests
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))

import export_images as ex  # noqa: E402

DATA = ex.OUT_DIR
METADATA = ex.REPO_ROOT / "runs" / "controlled_evaluation" / "processed" / "metadata.parquet"
RAW_SUFFIXES = {".tif", ".tiff", ".jp2", ".img", ".lbl", ".npy", ".pt"}


def load(rel: str) -> dict:
    return json.loads((DATA / rel).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def sites() -> list[dict]:
    return load("mars/sites.json")["sites"]


def test_no_raw_rasters_shipped():
    shipped = [p for p in (ex.REPO_ROOT / "website" / "public").rglob("*") if p.suffix.lower() in RAW_SUFFIXES]
    assert shipped == []


def test_public_data_within_budget():
    total = sum(p.stat().st_size for p in DATA.rglob("*") if p.is_file())
    assert total <= ex.SIZE_BUDGET_BYTES, f"{total:,} bytes"


def test_mars_previews_match_metadata(sites):
    for s in sites:
        p, t = s["preview"], s["thumb"]
        with Image.open(DATA / p["src"]) as im:
            assert im.size == (p["width"], p["height"])
            assert im.mode in ("LA", "RGBA"), "nodata must be transparent"
        with Image.open(DATA / t["src"]) as im:
            assert im.size == (t["width"], t["height"]) and max(im.size) == ex.THUMB_LONG_EDGE
        target = ex.PREVIEW_LONG_EDGE[s["site_id"]]
        native = (s["native_width_px"], s["native_height_px"])
        if target is None:
            assert (p["width"], p["height"]) == native
        else:  # never upsampled: strips smaller than the target keep their native size
            assert max(p["width"], p["height"]) == min(target, max(native))
        assert p["scale_from_native"]["x"] == pytest.approx(p["width"] / native[0])
        assert p["scale_from_native"]["y"] == pytest.approx(p["height"] / native[1])


def test_moon_sprites_match_crops():
    for cls in load("moon/summary.json")["classes"]:
        doc = load(f"moon/classes/{cls}.json")
        sp = doc["sprite"]
        assert sp["count"] == len(doc["crops"]) <= sp["cols"] * sp["rows"]
        with Image.open(DATA / sp["src"]) as im:
            assert im.size == (sp["width"], sp["height"]) == (sp["cols"] * sp["cell"], sp["rows"] * sp["cell"])
        with Image.open(DATA / doc["thumb"]["src"]) as im:
            assert im.size == (doc["thumb"]["width"], doc["thumb"]["height"])


@pytest.mark.skipif(not METADATA.exists(), reason="processed Mars shards not present")
def test_tile_positions_align_with_previews(sites):
    """Per-tile brightness in the preview must track the model input patches at the stated
    positions almost exactly, and clearly better than one tile to the right. Neighbouring
    tiles are similar, so the shifted correlation stays high; compare unexplained variance."""
    meta = pd.read_parquet(METADATA)
    for s in sites:
        g = meta[meta.site_id == s["site_id"]]
        with Image.open(DATA / s["preview"]["src"]) as im:
            la = np.asarray(im.convert("LA"), dtype=np.float32)
        lum, alpha = la[..., 0], la[..., 1] > 0
        sx, sy = s["preview"]["scale_from_native"]["x"], s["preview"]["scale_from_native"]["y"]
        size = s["tile_size_native_px"]

        def preview_mean(row: int, col: int) -> float:
            y0, y1 = int(row * sy), max(int((row + size) * sy), int(row * sy) + 1)
            x0, x1 = int(col * sx), max(int((col + size) * sx), int(col * sx) + 1)
            if y1 > lum.shape[0] or x1 > lum.shape[1]:
                return np.nan
            m = alpha[y0:y1, x0:x1]
            return float(lum[y0:y1, x0:x1][m].mean()) if m.mean() > 0.9 else np.nan

        shards: dict[str, np.ndarray] = {}
        truth, here, shifted = [], [], []
        for r in g.itertuples():
            shard = shards.setdefault(r.shard_path, np.load(r.shard_path, mmap_mode="r"))
            patch = np.asarray(shard[r.shard_idx], dtype=np.float32)
            valid = patch[2] > 0
            a, b = preview_mean(r.row, r.col), preview_mean(r.row, r.col + size)
            if valid.mean() > 0.9 and np.isfinite(a) and np.isfinite(b):
                truth.append(float(patch[0][valid].mean()))
                here.append(a)
                shifted.append(b)
        assert len(truth) > 100, s["site_id"]
        r_here = np.corrcoef(truth, here)[0, 1]
        r_shift = np.corrcoef(truth, shifted)[0, 1]
        print(f"{s['site_id']}: n={len(truth)} r={r_here:.4f} shifted r={r_shift:.4f}")
        assert r_here > 0.98, (s["site_id"], r_here)
        assert (1 - r_here**2) < 0.5 * (1 - r_shift**2), (s["site_id"], r_here, r_shift)
