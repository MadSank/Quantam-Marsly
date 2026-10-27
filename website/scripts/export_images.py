"""Export web image derivatives for the website (never the raw rasters).

Run from the repository root with the Streamlit app's venv, after export_data.py:

    visualization_app/.venv/Scripts/python.exe website/scripts/export_images.py

Mars: one WebP preview per HiRISE strip (display stretch only; nodata transparent)
plus a rail thumbnail. Moon: one sprite sheet of test crops per class plus a class
thumbnail. Writes img/manifest.json and fills the preview/thumb/sprite fields in
mars/sites.json and moon/classes/<class>.json. See website/PLAN.md "Images".
"""

from __future__ import annotations

import io
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from stable_io import write_bytes, write_json

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "website" / "public" / "data"
IMG_DIR = OUT_DIR / "img"
MANIFEST = REPO_ROOT / "data" / "real_manifest.csv"
MOON_ROOT = REPO_ROOT / "data" / "raw" / "moon" / "lrocnet"
SIZE_BUDGET_BYTES = 25 * 1024 * 1024

# Long edge of each Mars preview in pixels; None keeps the native resolution.
PREVIEW_LONG_EDGE: dict[str, int | None] = {
    "site_01": None,
    "site_02": 4000,
    "site_03": 4000,
    "site_04": 4000,
    "site_05": 8000,
}
THUMB_LONG_EDGE = 320
WEBP_QUALITY = 80
STRETCH_PERCENTILES = (2.0, 98.0)
CORE_NULL_DN = 0  # HiRISE ortho nodata DN (dataset_metadata.json "core_null")
MOON_CELL_PX = 128
MOON_THUMB_GRID = 3
MOON_THUMB_CELL_PX = 108

Image.MAX_IMAGE_PIXELS = None  # HiRISE strips exceed Pillow's decompression-bomb default


def rel(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT).as_posix()


def data_rel(path: Path) -> str:
    return path.resolve().relative_to(OUT_DIR).as_posix()


def save_webp(img: Image.Image, path: Path) -> None:
    buf = io.BytesIO()
    img.save(buf, "WEBP", quality=WEBP_QUALITY, method=6)
    write_bytes(path, buf.getvalue())


# ---------------------------------------------------------------- Mars

def decode_band(path: Path, band: int, native_long: int, target_long: int | None) -> Image.Image:
    """Decode one band of a JP2 at the coarsest resolution level still >= the target size.

    Some products expose fewer resolution levels than their size suggests, so on a
    decode error we retry one level finer.
    """
    wanted = 0 if target_long is None else max(0, int(math.floor(math.log2(native_long / target_long))))
    for level in range(wanted, -1, -1):
        try:
            im = Image.open(path)
            im.reduce = level
            im.load()
            break
        except OSError:
            if level == 0:
                raise
    if im.mode == "L":
        if band != 1:
            raise ValueError(f"{path.name}: single-band product but band {band} requested")
        return im
    return im.getchannel(band - 1)  # multi-band products decode as RGB in band order (IRB: IR, RED, BG)


def stretch(gray: Image.Image) -> Image.Image:
    """Per-strip 2–98% display stretch over valid pixels; nodata becomes transparent."""
    dn = np.asarray(gray, dtype=np.float32)
    valid = dn != CORE_NULL_DN
    lo, hi = np.percentile(dn[valid], STRETCH_PERCENTILES)
    out = np.clip((dn - lo) / max(hi - lo, 1e-6), 0.0, 1.0) * 255.0
    lum = Image.fromarray(out.round().astype(np.uint8), "L")
    alpha = Image.fromarray(np.where(valid, 255, 0).astype(np.uint8), "L")
    return Image.merge("LA", (lum, alpha))


def resize_long(img: Image.Image, long_edge: int) -> Image.Image:
    w, h = img.size
    scale = long_edge / max(w, h)
    if scale >= 1:
        return img
    return img.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.Resampling.LANCZOS)


def export_mars(written: list[Path]) -> dict[str, Any]:
    sites_path = OUT_DIR / "mars" / "sites.json"
    sites_doc = json.loads(sites_path.read_text(encoding="utf-8"))
    manifest = pd.read_csv(MANIFEST).set_index("site_id")
    out: dict[str, Any] = {}
    for site in sites_doc["sites"]:
        sid = site["site_id"]
        row = manifest.loc[sid]
        src = REPO_ROOT / row["ortho_path"]
        native_w, native_h = site["native_width_px"], site["native_height_px"]
        with Image.open(src) as probe:
            if probe.size != (native_w, native_h):
                raise ValueError(f"{sid}: JP2 is {probe.size}, metadata says {(native_w, native_h)}")
        target = PREVIEW_LONG_EDGE[sid]
        gray = decode_band(src, int(row["ortho_band"]), max(native_w, native_h), target)
        la = stretch(gray)
        preview = la if target is None else resize_long(la, target)
        if target is None and preview.size != (native_w, native_h):
            preview = preview.resize((native_w, native_h), Image.Resampling.LANCZOS)
        thumb = resize_long(preview, THUMB_LONG_EDGE)

        preview_path = IMG_DIR / "mars" / f"{sid}_preview.webp"
        thumb_path = IMG_DIR / "mars" / f"{sid}_thumb.webp"
        save_webp(preview, preview_path)
        save_webp(thumb, thumb_path)
        written += [preview_path, thumb_path]

        pw, ph = preview.size
        out[sid] = {
            "source": rel(src),
            "band": int(row["ortho_band"]),
            "display_stretch": {"percentiles": list(STRETCH_PERCENTILES), "note": "display stretch only"},
            "preview": {"src": data_rel(preview_path), "width": pw, "height": ph,
                        "scale_from_native": {"x": pw / native_w, "y": ph / native_h}},
            "thumb": {"src": data_rel(thumb_path), "width": thumb.size[0], "height": thumb.size[1]},
        }
        site["preview"] = out[sid]["preview"]
        site["thumb"] = out[sid]["thumb"]
        print(f"{sid}: {native_w}x{native_h} -> preview {pw}x{ph}")
    write_json(sites_path, sites_doc, compact=False)
    return out


# ---------------------------------------------------------------- Moon

def export_moon(written: list[Path]) -> dict[str, Any]:
    summary = json.loads((OUT_DIR / "moon" / "summary.json").read_text(encoding="utf-8"))
    out: dict[str, Any] = {}
    for cls in summary["classes"]:
        class_path = OUT_DIR / "moon" / "classes" / f"{cls}.json"
        doc = json.loads(class_path.read_text(encoding="utf-8"))
        crops = doc["crops"]
        cols = math.ceil(math.sqrt(len(crops)))
        rows = math.ceil(len(crops) / cols)
        sheet = Image.new("L", (cols * MOON_CELL_PX, rows * MOON_CELL_PX), 0)
        cells = []
        for crop in crops:
            with Image.open(MOON_ROOT / crop["sample_id"]) as im:
                # LROCNet crops are grayscale stored as RGB (identical channels).
                cell = im.convert("L").resize((MOON_CELL_PX, MOON_CELL_PX), Image.Resampling.LANCZOS)
            i = crop["img"]
            sheet.paste(cell, ((i % cols) * MOON_CELL_PX, (i // cols) * MOON_CELL_PX))
            cells.append(cell)

        n = MOON_THUMB_GRID
        thumb = Image.new("L", (n * MOON_THUMB_CELL_PX, n * MOON_THUMB_CELL_PX), 0)
        for i, cell in enumerate(cells[: n * n]):
            small = cell.resize((MOON_THUMB_CELL_PX, MOON_THUMB_CELL_PX), Image.Resampling.LANCZOS)
            thumb.paste(small, ((i % n) * MOON_THUMB_CELL_PX, (i // n) * MOON_THUMB_CELL_PX))

        sprite_path = IMG_DIR / "moon" / f"{cls}_sprite.webp"
        thumb_path = IMG_DIR / "moon" / f"{cls}_thumb.webp"
        save_webp(sheet, sprite_path)
        save_webp(thumb, thumb_path)
        written += [sprite_path, thumb_path]

        out[cls] = {
            "sprite": {"src": data_rel(sprite_path), "cell": MOON_CELL_PX, "cols": cols, "rows": rows,
                       "count": len(crops), "width": sheet.size[0], "height": sheet.size[1]},
            "thumb": {"src": data_rel(thumb_path), "width": thumb.size[0], "height": thumb.size[1]},
        }
        doc["sprite"] = out[cls]["sprite"]
        doc["thumb"] = out[cls]["thumb"]
        write_json(class_path, doc, compact=True)
        print(f"{cls}: {len(crops)} crops -> sprite {cols}x{rows} cells")
    return out


# ---------------------------------------------------------------- landing

LANDING_SIZE = (1600, 800)  # door image, 2:1 like its frame
LANDING_MARS_SITE = "site_01"
LANDING_MOON_PER_CLASS = 6   # crops per class in the Moon mosaic (first in sample-id order)
LANDING_MARS_CROP = 0.7      # share of the preview width kept, so the crop stays inside the imaged area
LANDING_MOON_COLS = 6


def export_landing(written: list[Path], mars: dict[str, Any]) -> dict[str, Any]:
    """Door images for the landing page, cut from the derivatives above (real data only)."""
    out: dict[str, Any] = {}
    w, h = LANDING_SIZE

    # Mars: the centre of one strip's preview, cropped to the door's aspect (display stretch as in the preview).
    with Image.open(OUT_DIR / mars[LANDING_MARS_SITE]["preview"]["src"]) as im:
        preview = im.convert("LA")
    pw, ph = preview.size
    cw = min(round(pw * LANDING_MARS_CROP), round(ph * w / h))
    ch = round(cw * h / w)
    box = ((pw - cw) // 2, (ph - ch) // 2, (pw - cw) // 2 + cw, (ph - ch) // 2 + ch)
    mars_img = preview.crop(box).resize((w, h), Image.Resampling.LANCZOS)
    mars_path = IMG_DIR / "landing" / "mars.webp"
    save_webp(mars_img, mars_path)
    out["mars"] = {"src": data_rel(mars_path), "width": w, "height": h, "site_id": LANDING_MARS_SITE,
                   "crop_from_preview": list(box)}

    # Moon: a mosaic of real test crops, the same number from each class, classes interleaved.
    summary = json.loads((OUT_DIR / "moon" / "summary.json").read_text(encoding="utf-8"))
    per_class = [json.loads((OUT_DIR / "moon" / "classes" / f"{c}.json").read_text(encoding="utf-8"))["crops"]
                 [:LANDING_MOON_PER_CLASS] for c in summary["classes"]]
    crops = [c for group in zip(*per_class) for c in group]
    cols = LANDING_MOON_COLS
    rows = math.ceil(len(crops) / cols)
    cell = w // cols
    sheet = Image.new("L", (cols * cell, rows * cell), 0)
    for i, crop in enumerate(crops):
        with Image.open(MOON_ROOT / crop["sample_id"]) as im:
            tile = im.convert("L").resize((cell, cell), Image.Resampling.LANCZOS)
        sheet.paste(tile, ((i % cols) * cell, (i // cols) * cell))
    moon_path = IMG_DIR / "landing" / "moon.webp"
    save_webp(sheet, moon_path)
    out["moon"] = {"src": data_rel(moon_path), "width": sheet.size[0], "height": sheet.size[1],
                   "sample_ids": [c["sample_id"] for c in crops]}
    written += [mars_path, moon_path]
    return out


# ---------------------------------------------------------------- report

def size_report(written: list[Path]) -> None:
    def total(paths) -> int:
        return sum(p.stat().st_size for p in paths)

    print(f"\n{'file':<40}{'bytes':>12}")
    for p in sorted(written):
        print(f"{data_rel(p):<40}{p.stat().st_size:>12,}")
    files = [p for p in OUT_DIR.rglob("*") if p.is_file()]
    groups = {
        "images": [p for p in files if p.is_relative_to(IMG_DIR)],
        "lab": [p for p in files if p.is_relative_to(OUT_DIR / "lab")],
    }
    groups["data"] = [p for p in files if p not in groups["images"] and p not in groups["lab"]]
    for name in ("data", "images", "lab"):
        print(f"{'public/data ' + name:<40}{total(groups[name]):>12,}")
    grand = total(files)
    print(f"{'public/data total':<40}{grand:>12,}  (budget {SIZE_BUDGET_BYTES:,})")
    if grand > SIZE_BUDGET_BYTES:
        sys.exit("public/data exceeds the 25 MB budget")


def main() -> None:
    written: list[Path] = []
    manifest = {"mars": export_mars(written), "moon": export_moon(written)}
    manifest["landing"] = export_landing(written, manifest["mars"])
    path = IMG_DIR / "manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, manifest, compact=False)
    written.append(path)
    size_report(written)


if __name__ == "__main__":
    os.chdir(REPO_ROOT)
    main()
