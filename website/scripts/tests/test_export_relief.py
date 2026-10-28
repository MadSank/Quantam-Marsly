"""Checks on the globes' relief (steps 8b and 9b): MOLA- and LOLA-derived normal maps.

Run from the repository root with the Streamlit app's venv:

    visualization_app/.venv/Scripts/python.exe -m pytest website/scripts/tests/test_export_relief.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))

import export_relief as er  # noqa: E402

RELIEF = {w: json.loads((er.GLOBE_DIR / f"{w}_relief.json").read_text(encoding="utf-8")) for w in er.WORLDS}

# Known extremes (lat, lon E in -180..180), each within 2 degrees.
EXTREMES = {
    "mars": {"max": (18.6, -133.8), "min_box": ((-45, -30), (55, 75))},  # Olympus Mons; Hellas floor
    "moon": {"max": (5.4, -158.6), "min_box": ((-72, -68), (-176, -168))},  # Selenean summit; Antoniadi
}


@pytest.mark.parametrize("world", list(er.WORLDS))
def test_relief_files_match_their_record(world):
    relief = RELIEF[world]
    assert relief["vertical_exaggeration"] == er.WORLDS[world].exaggeration
    for key, tex in relief["textures"].items():
        with Image.open(er.OUT_DIR / tex["src"]) as im:
            assert im.size == (tex["width"], tex["height"]) and tex["width"] == 2 * tex["height"]
            rgb = np.asarray(im.convert("RGB"), dtype=np.float64) / 255 * 2 - 1
        # Unit normals pointing up out of the surface (blue dominant), within 8-bit rounding.
        assert rgb[..., 2].min() > 0
        assert np.abs(np.linalg.norm(rgb, axis=2) - 1).mean() < 0.03, key


@pytest.mark.parametrize("world", list(er.WORLDS))
def test_heights_are_rolled_to_the_colour_texture_layout(world):
    if not er.WORLDS[world].img.exists():
        pytest.skip(f"{world} elevation grid not on this machine")
    h = er.read_heights(world)
    lat = lambda r: 90 - (r + 0.5) / 16  # noqa: E731
    lon = lambda c: -180 + (c + 0.5) / 16  # noqa: E731
    r, c = np.unravel_index(np.argmax(h), h.shape)
    want = EXTREMES[world]["max"]
    assert abs(lat(r) - want[0]) < 2 and abs(lon(c) - want[1]) < 2
    r, c = np.unravel_index(np.argmin(h), h.shape)
    (lat0, lat1), (lon0, lon1) = EXTREMES[world]["min_box"]
    assert lat0 < lat(r) < lat1 and lon0 < lon(c) < lon1
    assert RELIEF[world]["elevation_range_m"] == [int(h.min()), int(h.max())]
