"""Export the globes' relief: tangent-space normal maps from laser-altimeter topography.

Run from the repository root with the Streamlit app's venv (numpy + Pillow):

    visualization_app/.venv/Scripts/python.exe website/scripts/export_relief.py

Sources (NASA PDS Geosciences Node, public domain), downloaded with approval into gitignored
data/raw/textures/:
  - Mars: MGS MOLA MEGDR topography at 16 px/deg (mola/megt90n000eb.img + .lbl).
  - Moon: LRO LOLA GDR topography at 16 px/deg (lola/ldem_16.img + .lbl).
Both grids run 0..360 deg E; each is rolled to -180..180 deg so it lines up with that world's
colour texture (export_images.py), then area-averaged to the texture size. Slopes are taken on
the sphere: north-south spacing R*dphi, east-west spacing R*cos(lat)*dlambda (cos clamped near
the poles). Normals use the OpenGL/three.js tangent convention: red = east (+u), green = north
(+v), blue = up.

Writes img/globe/{world}_normal_1k.webp (lossless) and img/globe/{world}_relief.json.
Decoration only: no number on the site is read from these files.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from stable_io import write_bytes, write_json

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "website" / "public" / "data"
GLOBE_DIR = OUT_DIR / "img" / "globe"
TEXTURE_DIR = REPO_ROOT / "data" / "raw" / "textures"


@dataclass(frozen=True)
class Source:
    img: Path
    lbl: Path
    url: str
    credit: str
    dtype: str  # numpy dtype of one sample, from the label's SAMPLE_TYPE
    radius_m: float  # reference radius for slopes
    # Relief is exaggerated so it reads at globe scale; at ~10 km/px real slopes are a few degrees.
    exaggeration: float

    @property
    def lbl_text(self) -> str:
        return self.lbl.read_text(encoding="ascii", errors="replace")


WORLDS: dict[str, Source] = {
    "mars": Source(
        img=TEXTURE_DIR / "mola" / "megt90n000eb.img",
        lbl=TEXTURE_DIR / "mola" / "megt90n000eb.lbl",
        url="https://pds-geosciences.wustl.edu/mgs/mgs-m-mola-5-megdr-l3-v1/mgsl_300x/meg016/",
        credit="MGS MOLA MEGDR topography, 16 px/deg (NASA / GSFC / MOLA Science Team, via the PDS Geosciences Node; public domain)",
        dtype=">i2",
        radius_m=3_396_190.0,  # MOLA reference radius
        exaggeration=12.0,
    ),
    "moon": Source(
        img=TEXTURE_DIR / "lola" / "ldem_16.img",
        lbl=TEXTURE_DIR / "lola" / "ldem_16.lbl",
        url="https://pds-geosciences.wustl.edu/lro/lro-l-lola-3-rdr-v1/lrolol_1xxx/data/lola_gdr/cylindrical/img/",
        credit="LRO LOLA GDR topography, 16 px/deg (NASA / GSFC / LOLA Science Team, via the PDS Geosciences Node; public domain)",
        dtype="<i2",
        radius_m=1_737_400.0,  # LOLA reference sphere (OFFSET)
        exaggeration=6.0,  # lunar crater walls are steeper at this scale than Mars's plains
    ),
}

# One 1k map per world: about the planet's on-screen resolution at overview zoom. Lossless,
# because lossy WebP's chroma subsampling bends the encoded normals (mean error 0.04, some steep
# polar normals flipped inward); a lossless 2k map would cost about 3.3 MB.
WIDTHS = (1024,)
MIN_COS_LAT = 0.05


def rel(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT).as_posix()


def label_value(text: str, key: str) -> str:
    m = re.search(rf"^\s*{key}\s*=\s*([^\s<]+)", text, re.MULTILINE)
    if not m:
        raise ValueError(f"{key} missing from the label")
    return m.group(1)


def read_heights(world: str) -> np.ndarray:
    """Heights in metres relative to the reference radius, rolled to -180..180 deg E."""
    src = WORLDS[world]
    lbl = src.lbl_text
    lines, samples = int(label_value(lbl, "LINES")), int(label_value(lbl, "LINE_SAMPLES"))
    sample_type = label_value(lbl, "SAMPLE_TYPE")
    assert int(label_value(lbl, "SAMPLE_BITS")) == 16
    assert sample_type == {">i2": "MSB_INTEGER", "<i2": "LSB_INTEGER"}[src.dtype], sample_type
    assert float(label_value(lbl, "WESTERNMOST_LONGITUDE")) == 0.0, "expected a grid starting at 0 deg E"
    scale = float(label_value(lbl, "SCALING_FACTOR")) if re.search(r"^\s*SCALING_FACTOR\s*=", lbl, re.MULTILINE) else 1.0
    data = np.fromfile(src.img, dtype=src.dtype)
    assert data.size == lines * samples, f"{src.img.name}: {data.size} samples, expected {lines * samples}"
    heights = data.reshape(lines, samples).astype(np.float64) * scale
    return np.roll(heights, samples // 2, axis=1)


def read_mola() -> np.ndarray:
    """Mars heights (kept for the export tests)."""
    return read_heights("mars")


def area_downsample(h: np.ndarray, width: int) -> np.ndarray:
    lines, samples = h.shape
    k = max(1, samples // width)
    w = samples // k
    return h[: (lines // k) * k, : w * k].reshape(lines // k, k, w, k).mean(axis=(1, 3))


def normal_map(h: np.ndarray, width: int, radius_m: float, exaggeration: float) -> Image.Image:
    rows, cols = h.shape
    dphi = np.pi / rows
    dlam = 2 * np.pi / cols
    lat = (0.5 - (np.arange(rows) + 0.5) / rows) * np.pi
    cos_lat = np.maximum(np.cos(lat), MIN_COS_LAT)[:, None]
    # East: wrap around the dateline. North: row 0 is the north edge, so north = -row.
    dh_east = (np.roll(h, -1, axis=1) - np.roll(h, 1, axis=1)) / (2 * radius_m * cos_lat * dlam)
    padded = np.pad(h, ((1, 1), (0, 0)), mode="edge")
    dh_north = (padded[:-2] - padded[2:]) / (2 * radius_m * dphi)
    n = np.dstack([-exaggeration * dh_east, -exaggeration * dh_north, np.ones_like(h)])
    n /= np.linalg.norm(n, axis=2, keepdims=True)
    rgb = np.clip((n * 0.5 + 0.5) * 255.0 + 0.5, 0, 255).astype(np.uint8)
    img = Image.fromarray(rgb, "RGB")
    return img if img.width == width else img.resize((width, width // 2), Image.Resampling.LANCZOS)


def export_world(world: str) -> None:
    src = WORLDS[world]
    heights = read_heights(world)
    textures = {}
    for width in WIDTHS:
        img = normal_map(area_downsample(heights, width), width, src.radius_m, src.exaggeration)
        path = GLOBE_DIR / f"{world}_normal_{width // 1024}k.webp"
        buf = io.BytesIO()
        img.save(buf, "WEBP", lossless=True, quality=100, method=6)
        changed = write_bytes(path, buf.getvalue())
        textures[f"{width // 1024}k"] = {"src": path.resolve().relative_to(OUT_DIR).as_posix(), "width": img.width, "height": img.height}
        print(f"{path.name}: {img.width}x{img.height}, {path.stat().st_size / 1024:.0f} KB{'' if changed else ' (unchanged)'}")
    write_json(GLOBE_DIR / f"{world}_relief.json", {
        "source": [rel(src.img), rel(src.lbl)],
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"),
        "url": src.url,
        "credit": src.credit,
        "layout": "equirectangular, longitude -180..180 E left to right, latitude 90..-90 top to bottom",
        "convention": "tangent space: red = east (+u), green = north (+v), blue = up",
        "vertical_exaggeration": src.exaggeration,
        "elevation_range_m": [int(heights.min()), int(heights.max())],
        "textures": textures,
    })


def main() -> None:
    for world in WORLDS:
        export_world(world)


if __name__ == "__main__":
    main()
