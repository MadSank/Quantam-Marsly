"""Checks on the Mars globe relief (step 8b): MOLA-derived normal maps.

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

RELIEF = json.loads((er.GLOBE_DIR / "mars_relief.json").read_text(encoding="utf-8"))


def test_relief_files_match_their_record():
    assert RELIEF["vertical_exaggeration"] == er.EXAGGERATION
    for key, tex in RELIEF["textures"].items():
        with Image.open(er.OUT_DIR / tex["src"]) as im:
            assert im.size == (tex["width"], tex["height"]) and tex["width"] == 2 * tex["height"]
            rgb = np.asarray(im.convert("RGB"), dtype=np.float64) / 255 * 2 - 1
        # Unit normals pointing up out of the surface (blue dominant), within WebP rounding.
        assert rgb[..., 2].min() > 0
        assert np.abs(np.linalg.norm(rgb, axis=2) - 1).mean() < 0.03, key


@pytest.mark.skipif(not er.MOLA_IMG.exists(), reason="MOLA MEGDR not on this machine")
def test_mola_is_rolled_to_the_colour_texture_layout():
    h = er.read_mola()
    lat = lambda r: 90 - (r + 0.5) / 16  # noqa: E731
    lon = lambda c: -180 + (c + 0.5) / 16  # noqa: E731
    r, c = np.unravel_index(np.argmax(h), h.shape)
    assert abs(lat(r) - 18.6) < 2 and abs(lon(c) - (-133.8)) < 2  # Olympus Mons
    r, c = np.unravel_index(np.argmin(h), h.shape)
    assert -45 < lat(r) < -30 and 55 < lon(c) < 75  # Hellas basin floor
    assert RELIEF["elevation_range_m"] == [int(h.min()), int(h.max())]
