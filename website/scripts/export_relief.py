"""Export the Mars globe's relief: tangent-space normal maps from MOLA topography.

Run from the repository root with the Streamlit app's venv (numpy + Pillow):

    visualization_app/.venv/Scripts/python.exe website/scripts/export_relief.py

Source: MGS MOLA MEGDR topography at 16 px/deg (megt90n000eb.img + .lbl, NASA PDS
Geosciences Node, public domain), downloaded with approval into gitignored
data/raw/textures/mola/. The grid runs 0..360 deg E; it is rolled to -180..180 deg so it
lines up with the Viking MDIM colour texture (export_images.py), then area-averaged to
each texture size. Slopes are taken on the sphere: north-south spacing R*dphi, east-west
spacing R*cos(lat)*dlambda (cos clamped near the poles). Normals use the OpenGL/three.js
tangent convention: red = east (+u), green = north (+v), blue = up.

Writes img/globe/mars_normal_1k.webp (lossless) and img/globe/mars_relief.json. Decoration only:
no number on the site is read from these files.
"""

from __future__ import annotations

import io
import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from stable_io import write_bytes, write_json

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "website" / "public" / "data"
GLOBE_DIR = OUT_DIR / "img" / "globe"
MOLA_DIR = REPO_ROOT / "data" / "raw" / "textures" / "mola"
MOLA_IMG = MOLA_DIR / "megt90n000eb.img"
MOLA_LBL = MOLA_DIR / "megt90n000eb.lbl"
SOURCE_URL = "https://pds-geosciences.wustl.edu/mgs/mgs-m-mola-5-megdr-l3-v1/mgsl_300x/meg016/"
CREDIT = "MGS MOLA MEGDR topography, 16 px/deg (NASA / GSFC / MOLA Science Team, via the PDS Geosciences Node; public domain)"

MARS_RADIUS_M = 3_396_190.0  # MOLA reference radius
# One 1k map: about the planet's on-screen resolution at overview zoom (close-ups show the draped
# HiRISE strip instead). Lossless, because lossy WebP's chroma subsampling bends the encoded normals
# (mean error 0.04, some steep polar normals flipped inward); lossless 2k would cost 3.3 MB.
WIDTHS = (1024,)
# Relief is exaggerated so it reads at globe scale; at 10 km/px real slopes are a few degrees.
EXAGGERATION = 12.0
MIN_COS_LAT = 0.05


def rel(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT).as_posix()


def label_value(text: str, key: str) -> str:
    m = re.search(rf"^\s*{key}\s*=\s*([^\s<]+)", text, re.MULTILINE)
    if not m:
        raise ValueError(f"{key} missing from {MOLA_LBL.name}")
    return m.group(1)


def read_mola() -> np.ndarray:
    lbl = MOLA_LBL.read_text(encoding="ascii", errors="replace")
    lines, samples = int(label_value(lbl, "LINES")), int(label_value(lbl, "LINE_SAMPLES"))
    assert label_value(lbl, "SAMPLE_TYPE") == "MSB_INTEGER" and int(label_value(lbl, "SAMPLE_BITS")) == 16
    assert float(label_value(lbl, "WESTERNMOST_LONGITUDE")) == 0.0, "expected a grid starting at 0 deg E"
    data = np.fromfile(MOLA_IMG, dtype=">i2")
    assert data.size == lines * samples, f"{MOLA_IMG.name}: {data.size} samples, expected {lines * samples}"
    heights = data.reshape(lines, samples).astype(np.float64)
    # 0..360 E -> -180..180 E, the colour texture's layout.
    return np.roll(heights, samples // 2, axis=1)


def area_downsample(h: np.ndarray, width: int) -> np.ndarray:
    lines, samples = h.shape
    k = max(1, samples // width)
    w = samples // k
    return h[: (lines // k) * k, : w * k].reshape(lines // k, k, w, k).mean(axis=(1, 3))


def normal_map(h: np.ndarray, width: int) -> Image.Image:
    rows, cols = h.shape
    dphi = np.pi / rows
    dlam = 2 * np.pi / cols
    lat = (0.5 - (np.arange(rows) + 0.5) / rows) * np.pi
    cos_lat = np.maximum(np.cos(lat), MIN_COS_LAT)[:, None]
    # East: wrap around the dateline. North: row 0 is the north edge, so north = -row.
    dh_east = (np.roll(h, -1, axis=1) - np.roll(h, 1, axis=1)) / (2 * MARS_RADIUS_M * cos_lat * dlam)
    padded = np.pad(h, ((1, 1), (0, 0)), mode="edge")
    dh_north = (padded[:-2] - padded[2:]) / (2 * MARS_RADIUS_M * dphi)
    n = np.dstack([-EXAGGERATION * dh_east, -EXAGGERATION * dh_north, np.ones_like(h)])
    n /= np.linalg.norm(n, axis=2, keepdims=True)
    rgb = np.clip((n * 0.5 + 0.5) * 255.0 + 0.5, 0, 255).astype(np.uint8)
    img = Image.fromarray(rgb, "RGB")
    return img if img.width == width else img.resize((width, width // 2), Image.Resampling.LANCZOS)


def main() -> None:
    heights = read_mola()
    textures = {}
    for width in WIDTHS:
        img = normal_map(area_downsample(heights, width), width)
        path = GLOBE_DIR / f"mars_normal_{width // 1024}k.webp"
        buf = io.BytesIO()
        img.save(buf, "WEBP", lossless=True, quality=100, method=6)
        changed = write_bytes(path, buf.getvalue())
        textures[f"{width // 1024}k"] = {"src": path.resolve().relative_to(OUT_DIR).as_posix(), "width": img.width, "height": img.height}
        print(f"{path.name}: {img.width}x{img.height}, {path.stat().st_size / 1024:.0f} KB{'' if changed else ' (unchanged)'}")
    write_json(GLOBE_DIR / "mars_relief.json", {
        "source": [rel(MOLA_IMG), rel(MOLA_LBL)],
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"),
        "url": SOURCE_URL,
        "credit": CREDIT,
        "layout": "equirectangular, longitude -180..180 E left to right, latitude 90..-90 top to bottom",
        "convention": "tangent space: red = east (+u), green = north (+v), blue = up",
        "vertical_exaggeration": EXAGGERATION,
        "elevation_range_m": [int(heights.min()), int(heights.max())],
        "textures": textures,
    })


if __name__ == "__main__":
    main()
