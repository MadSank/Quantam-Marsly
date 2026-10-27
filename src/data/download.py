"""Ingestion and verification of HiRISE orthoimages and DTMs."""

import argparse
import logging
import os
import shutil
import urllib.request
from pathlib import Path
from typing import Any

import pandas as pd
import rasterio
import rasterio.windows

from src.utils import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def parse_solar_angles_from_metadata(raster_path: Path) -> tuple[float, float]:
    """Extracts solar incidence and azimuth angles from rasterio tags or sidecar label files."""
    inc_angle: float = float("nan")
    az_angle: float = float("nan")

    inc_keys = {"incidence_angle", "solar_incidence", "solar_incidence_angle", "sun_inc"}
    az_keys = {
        "sub_solar_azimuth",
        "solar_azimuth",
        "solar_azimuth_angle",
        "sun_azimuth",
        "sun_az",
    }

    try:
        with rasterio.open(raster_path) as src:
            tags_to_check: list[dict[str, Any]] = [src.tags()]
            for ns in ["ISIS3", "PDS", "DEFAULT"]:
                try:
                    t = src.tags(ns=ns)
                    if t:
                        tags_to_check.append(t)
                except Exception:
                    pass

            for tags in tags_to_check:
                for k, v in tags.items():
                    k_lower = k.lower()
                    if k_lower in inc_keys and pd.isna(inc_angle):
                        try:
                            inc_angle = float(str(v).split()[0].replace("<DEGREE>", ""))
                        except (ValueError, TypeError):
                            pass
                    if k_lower in az_keys and pd.isna(az_angle):
                        try:
                            az_angle = float(str(v).split()[0].replace("<DEGREE>", ""))
                        except (ValueError, TypeError):
                            pass
    except Exception as e:
        logging.warning("Could not read tags from raster %s: %s", raster_path, e)

    # Check sidecar files if angles are still missing
    if pd.isna(inc_angle) or pd.isna(az_angle):
        parent_dir = raster_path.parent
        stem = raster_path.stem
        candidates = [
            parent_dir / f"{stem}.LBL",
            parent_dir / f"{stem}.lbl",
            parent_dir / f"{stem}.xml",
            parent_dir / f"{stem}.aux.xml",
            raster_path.with_suffix(".LBL"),
            raster_path.with_suffix(".lbl"),
        ]
        for candidate in candidates:
            if candidate.exists() and candidate.is_file():
                try:
                    with open(candidate, "r", encoding="utf-8", errors="ignore") as f:
                        for line in f:
                            if "=" in line:
                                parts = line.split("=", 1)
                                k_lower = parts[0].strip().lower()
                                v_val = parts[1].strip()
                                if k_lower in inc_keys and pd.isna(inc_angle):
                                    try:
                                        inc_angle = float(v_val.split()[0].replace("<DEGREE>", ""))
                                    except (ValueError, TypeError):
                                        pass
                                if k_lower in az_keys and pd.isna(az_angle):
                                    try:
                                        az_angle = float(v_val.split()[0].replace("<DEGREE>", ""))
                                    except (ValueError, TypeError):
                                        pass
                except Exception as e:
                    logging.warning("Could not parse sidecar %s: %s", candidate, e)

    if pd.isna(inc_angle) or pd.isna(az_angle):
        logging.warning("Solar angles missing or incomplete for %s: inc=%s, az=%s", raster_path, inc_angle, az_angle)

    return inc_angle, az_angle


def verify_raster(path: Path) -> dict[str, Any]:
    """Verifies that a raster file opens and decodes valid pixels via rasterio/GDAL."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Raster file does not exist: {path}")

    with rasterio.open(path) as src:
        if src.width <= 0 or src.height <= 0:
            raise ValueError(f"Invalid raster dimensions for {path}: ({src.width}, {src.height})")
        if src.count < 1:
            raise ValueError(f"Raster has no bands: {path}")

        w_read = min(16, src.width)
        h_read = min(16, src.height)
        data = src.read(1, window=rasterio.windows.Window(0, 0, w_read, h_read))
        if data is None or data.shape != (h_read, w_read):
            raise RuntimeError(f"Failed to read test window from raster {path}")

        return {
            "width": src.width,
            "height": src.height,
            "crs": str(src.crs),
            "nodata": src.nodata,
        }


def fetch_or_link_file(source: str, dest_path: Path) -> Path:
    """Downloads remote URL or symlinks/copies local file to dest_path."""
    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    if source.startswith("http://") or source.startswith("https://"):
        part_path = dest_path.with_name(f"{dest_path.name}.part")
        downloaded_bytes = 0
        if part_path.exists():
            downloaded_bytes = part_path.stat().st_size

        req = urllib.request.Request(source)
        if downloaded_bytes > 0:
            req.add_header("Range", f"bytes={downloaded_bytes}-")

        mode = "ab" if downloaded_bytes > 0 else "wb"
        try:
            with urllib.request.urlopen(req) as resp, open(part_path, mode) as out_f:
                shutil.copyfileobj(resp, out_f)
            os.replace(part_path, dest_path)
        except Exception as e:
            if part_path.exists() and downloaded_bytes == 0:
                part_path.unlink(missing_ok=True)
            raise RuntimeError(f"Failed to download {source} to {dest_path}: {e}") from e

        return dest_path

    # Local file path
    src_p = Path(source).resolve()
    if not src_p.exists():
        raise FileNotFoundError(f"Source file does not exist: {source}")

    if dest_path.resolve() == src_p:
        return dest_path

    if dest_path.exists():
        dest_path.unlink()

    try:
        os.symlink(src_p, dest_path)
    except (OSError, NotImplementedError):
        shutil.copy2(src_p, dest_path)

    return dest_path


def process_manifest(config: dict[str, Any]) -> pd.DataFrame:
    """Processes manifest.csv, fetches/links rasters, verifies them, and writes verified_manifest.csv."""
    manifest_path = Path(config["paths"]["manifest_path"])
    raw_dir = Path(config["paths"]["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)

    columns = ["site_id", "ortho_local_path", "dtm_local_path", "sun_inc", "sun_az"]
    if not manifest_path.exists():
        logging.warning("Manifest file %s not found. Creating empty verified manifest.", manifest_path)
        out_df = pd.DataFrame(columns=columns)
        out_df.to_csv(raw_dir / "verified_manifest.csv", index=False)
        return out_df

    manifest_df = pd.read_csv(manifest_path)
    if len(manifest_df) == 0:
        logging.info("Manifest %s contains 0 rows.", manifest_path)
        out_df = pd.DataFrame(columns=columns)
        out_df.to_csv(raw_dir / "verified_manifest.csv", index=False)
        return out_df

    verified_rows: list[dict[str, Any]] = []
    for _, row in manifest_df.iterrows():
        site_id = str(row["site_id"]).strip()
        site_dir = raw_dir / site_id
        site_dir.mkdir(parents=True, exist_ok=True)

        ortho_src = str(row["ortho_path_or_url"]).strip()
        dtm_src = str(row["dtm_path_or_url"]).strip()

        ortho_name = Path(ortho_src.split("?")[0]).name
        dtm_name = Path(dtm_src.split("?")[0]).name

        ortho_dst = site_dir / ortho_name
        dtm_dst = site_dir / dtm_name

        logging.info("Processing site %s...", site_id)
        ortho_local = fetch_or_link_file(ortho_src, ortho_dst)
        dtm_local = fetch_or_link_file(dtm_src, dtm_dst)

        verify_raster(ortho_local)
        verify_raster(dtm_local)

        # Check if manifest already provided sun_inc and sun_az
        sun_inc = float(row["sun_inc"]) if "sun_inc" in row and pd.notna(row["sun_inc"]) else float("nan")
        sun_az = float(row["sun_az"]) if "sun_az" in row and pd.notna(row["sun_az"]) else float("nan")

        if pd.isna(sun_inc) or pd.isna(sun_az):
            meta_inc, meta_az = parse_solar_angles_from_metadata(ortho_local)
            if pd.isna(sun_inc):
                sun_inc = meta_inc
            if pd.isna(sun_az):
                sun_az = meta_az

        verified_rows.append(
            {
                "site_id": site_id,
                "ortho_local_path": str(ortho_local.resolve()),
                "dtm_local_path": str(dtm_local.resolve()),
                "sun_inc": sun_inc,
                "sun_az": sun_az,
            }
        )

    out_df = pd.DataFrame(verified_rows, columns=columns)
    verified_csv = raw_dir / "verified_manifest.csv"
    out_df.to_csv(verified_csv, index=False)
    logging.info("Saved %d verified sites to %s", len(out_df), verified_csv)
    return out_df


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Download and verify rasters in manifest.")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file.")
    parser.add_argument("--override", nargs="*", default=None, help="Config overrides key=value.")
    args = parser.parse_args(argv)

    cfg = load_config(args.config, overrides=args.override)
    process_manifest(cfg)


if __name__ == "__main__":
    main()
