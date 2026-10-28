"""Export topographic contour lines for the /mars page background (never the raw DTMs).

Run from the repository root with the root venv (rasterio, contourpy via matplotlib):

    .venv/Scripts/python.exe website/scripts/export_contours.py

For each site, the HiRISE DTM named in data/real_manifest.csv is read decimated
(average resampling, nodata masked), and contours are traced at a round interval
chosen so the site's relief gives about CONTOURS_TARGET lines. Every fifth line is
an index contour. Writes mars/contours/<site_id>.json:

    { source, generated_at, site_id, dtm_product_id, width, height, interval_m,
      levels: [{ elevation_m, index, d }] }

``d`` is an SVG path in a viewBox of width x height (the decimated DTM grid).
Decoration only: no number on the page is read from these files.
"""

from __future__ import annotations

import math
from pathlib import Path

import contourpy
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling

from stable_io import write_json

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "website" / "public" / "data" / "mars" / "contours"
MANIFEST = REPO_ROOT / "data" / "real_manifest.csv"

LONG_EDGE = 320  # decimated DTM grid, long edge in cells
CONTOURS_TARGET = 18
INDEX_EVERY = 5
RELIEF_PERCENTILES = (1.0, 99.0)
MIN_POINTS = 6  # drop tiny closed loops (noise at the decimated scale)
MIN_STEP = 1.5  # thin vertices closer than this (grid cells) to the last one kept


def rel(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT).as_posix()


def nice_interval(relief: float, target: int = CONTOURS_TARGET) -> float:
    """A 1-2-5 interval giving at most about ``target`` contours over ``relief`` metres."""
    raw = relief / target
    power = 10 ** math.floor(math.log10(raw))
    for m in (1, 2, 5, 10):
        if m * power >= raw:
            return float(m * power)
    return float(10 * power)


def read_decimated(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        scale = LONG_EDGE / max(src.width, src.height)
        shape = (max(2, round(src.height * scale)), max(2, round(src.width * scale)))
        band = src.read(1, out_shape=shape, resampling=Resampling.average, masked=True)
    data = band.astype(np.float64).filled(np.nan)
    # PDS DTMs flag nodata with a huge negative float even when the mask misses it.
    data[~np.isfinite(data) | (data < -1e30)] = np.nan
    return data


def thin(line: np.ndarray) -> np.ndarray:
    kept = [line[0]]
    for p in line[1:-1]:
        if np.hypot(*(p - kept[-1])) >= MIN_STEP:
            kept.append(p)
    kept.append(line[-1])
    return np.asarray(kept)


def path_d(lines: list[np.ndarray]) -> str:
    parts = []
    for line in lines:
        if len(line) < MIN_POINTS:
            continue
        pts = " ".join(f"{x:.1f} {y:.1f}" for x, y in thin(line))
        parts.append(f"M{pts}")
    return "".join(parts)


def contours_for(dem: np.ndarray) -> tuple[float, list[dict]]:
    lo, hi = np.nanpercentile(dem, RELIEF_PERCENTILES)
    interval = nice_interval(float(hi - lo))
    start = math.ceil(lo / interval) * interval
    gen = contourpy.contour_generator(z=np.ma.masked_invalid(dem), line_type=contourpy.LineType.Separate)
    levels = []
    k = 0
    elevation = start
    while elevation <= hi:
        d = path_d(gen.lines(elevation))
        if d:
            levels.append({"elevation_m": round(elevation, 3), "index": round(elevation / interval) % INDEX_EVERY == 0, "d": d})
        k += 1
        elevation = start + k * interval
    return interval, levels


def main() -> None:
    manifest = pd.read_csv(MANIFEST)
    for row in manifest.itertuples():
        dtm = REPO_ROOT / row.dtm_path
        dem = read_decimated(dtm)
        interval, levels = contours_for(dem)
        payload = {
            "source": [rel(dtm), rel(MANIFEST)],
            "generated_at": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"),
            "site_id": row.site_id,
            "dtm_product_id": dtm.stem,
            "width": int(dem.shape[1]),
            "height": int(dem.shape[0]),
            "interval_m": interval,
            "levels": levels,
        }
        out = OUT_DIR / f"{row.site_id}.json"
        changed = write_json(out, payload, compact=True)
        print(f"{row.site_id}: {len(levels)} contours at {interval:g} m, {out.stat().st_size / 1024:.1f} KB{'' if changed else ' (unchanged)'}")


if __name__ == "__main__":
    main()
