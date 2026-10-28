"""Checks on the exported DTM contours (page decoration for /mars).

Run from the repository root with the root venv, which has rasterio:

    .venv/Scripts/python.exe -m pytest website/scripts/tests/test_export_contours.py
"""

from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

import pytest

pytest.importorskip("rasterio")
pytest.importorskip("contourpy")

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))

import export_contours as ec  # noqa: E402

DATA = ec.REPO_ROOT / "website" / "public" / "data"
SITES = json.loads((DATA / "mars" / "sites.json").read_text(encoding="utf-8"))["sites"]
NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


def load(site_id: str) -> dict:
    return json.loads((ec.OUT_DIR / f"{site_id}.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("site", SITES, ids=lambda s: s["site_id"])
def test_contours_match_site_and_stay_in_frame(site):
    c = load(site["site_id"])
    assert c["dtm_product_id"] == site["dtm_product_id"]
    assert any(src.endswith(f"{site['dtm_product_id']}.IMG") for src in c["source"])
    elevations = [lv["elevation_m"] for lv in c["levels"]]
    assert elevations == sorted(elevations) and len(elevations) >= 5
    mantissa = c["interval_m"] / 10 ** math.floor(math.log10(c["interval_m"]))
    assert round(mantissa, 6) in (1, 2, 5)
    for lv in c["levels"]:
        assert lv["index"] == (round(lv["elevation_m"] / c["interval_m"]) % ec.INDEX_EVERY == 0)
        xy = [float(v) for v in NUMBER.findall(lv["d"])]
        xs, ys = xy[0::2], xy[1::2]
        assert min(xs) >= -0.5 and max(xs) <= c["width"] - 0.5
        assert min(ys) >= -0.5 and max(ys) <= c["height"] - 0.5


def test_nice_interval():
    assert ec.nice_interval(1700) == 100
    assert ec.nice_interval(300) == 20
    assert ec.nice_interval(18) == 1


@pytest.mark.parametrize("site_id", ["site_04"])
def test_contours_reproduce_from_dtm(site_id):
    manifest = __import__("pandas").read_csv(ec.MANIFEST)
    dtm = ec.REPO_ROOT / manifest.set_index("site_id").loc[site_id, "dtm_path"]
    if not dtm.exists():
        pytest.skip("DTM not on this machine")
    interval, levels = ec.contours_for(ec.read_decimated(dtm))
    saved = load(site_id)
    assert interval == saved["interval_m"]
    assert [lv["elevation_m"] for lv in levels] == [lv["elevation_m"] for lv in saved["levels"]]
    assert [lv["d"] for lv in levels] == [lv["d"] for lv in saved["levels"]]
