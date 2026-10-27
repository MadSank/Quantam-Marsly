"""Synthetic terrain generator for unit tests and smoke verification."""

import copy
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rasterio
import rasterio.transform
from scipy.ndimage import gaussian_filter

from src.data.preprocess import preprocess_site

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def create_tilted_plane_geotiffs(
    out_dir: Path,
    site_id: str,
    slope_deg: float,
    azimuth_deg: float = 90.0,
    size_px: int = 512,
    gsd_m: float = 0.5,
    sun_inc_deg: float = 45.0,
    sun_az_deg: float = 135.0,
) -> tuple[Path, Path]:
    """Creates synthetic analytical tilted plane DTM and hillshaded ortho GeoTIFFs."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    x = np.arange(size_px, dtype=np.float64) * gsd_m
    y = np.arange(size_px, dtype=np.float64) * gsd_m
    xx, yy = np.meshgrid(x, y)

    theta = np.radians(slope_deg)
    alpha = np.radians(azimuth_deg)

    p = float(np.tan(theta) * np.sin(alpha))
    q = float(-np.tan(theta) * np.cos(alpha))

    dtm = (100.0 + p * xx + q * yy).astype(np.float32)

    norm_denom = np.sqrt(1.0 + p * p + q * q)
    nx = -p / norm_denom
    ny = -q / norm_denom
    nz = 1.0 / norm_denom

    inc_rad = np.radians(sun_inc_deg)
    az_rad = np.radians(sun_az_deg)
    sx = np.sin(inc_rad) * np.sin(az_rad)
    sy = np.sin(inc_rad) * np.cos(az_rad)
    sz = np.cos(inc_rad)

    cos_i = max(0.0, float(nx * sx + ny * sy + nz * sz))
    ortho = np.full(
        (size_px, size_px),
        float(np.clip(0.15 + 0.70 * cos_i, 0.05, 0.95)),
        dtype=np.float32,
    )

    transform = rasterio.transform.from_origin(500000.0, 1000000.0, gsd_m, gsd_m)
    crs = "EPSG:32631"

    dtm_path = out_dir / f"{site_id}_dtm.tif"
    ortho_path = out_dir / f"{site_id}_ortho.tif"

    tags = {
        "INCIDENCE_ANGLE": str(sun_inc_deg),
        "SOLAR_AZIMUTH_ANGLE": str(sun_az_deg),
    }

    with rasterio.open(
        dtm_path,
        "w",
        driver="GTiff",
        height=size_px,
        width=size_px,
        count=1,
        dtype=dtm.dtype,
        crs=crs,
        transform=transform,
        nodata=-9999.0,
    ) as dst:
        dst.write(dtm, 1)
        dst.update_tags(**tags)

    with rasterio.open(
        ortho_path,
        "w",
        driver="GTiff",
        height=size_px,
        width=size_px,
        count=1,
        dtype=ortho.dtype,
        crs=crs,
        transform=transform,
        nodata=-9999.0,
    ) as dst:
        dst.write(ortho, 1)
        dst.update_tags(**tags)

    return ortho_path, dtm_path


def generate_synthetic_site_geotiffs(
    out_dir: Path,
    site_id: str,
    base_slope_deg: float,
    roughness_scale: float,
    size_px: int,
    gsd_m: float,
    sun_inc_deg: float,
    sun_az_deg: float,
    rng: np.random.Generator,
) -> tuple[Path, Path]:
    """Generates synthetic site with planar ramp, random craters, roughness, and Lambertian hillshade."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    x = np.arange(size_px, dtype=np.float64) * gsd_m
    y = np.arange(size_px, dtype=np.float64) * gsd_m
    xx, yy = np.meshgrid(x, y)

    ramp_azimuth = rng.uniform(0.0, 360.0)
    theta = np.radians(base_slope_deg)
    alpha = np.radians(ramp_azimuth)
    p = float(np.tan(theta) * np.sin(alpha))
    q = float(-np.tan(theta) * np.cos(alpha))

    dtm = 200.0 + p * xx + q * yy

    n_craters = rng.integers(4, 9)
    for _ in range(n_craters):
        cx = rng.uniform(60.0, (size_px - 60) * gsd_m)
        cy = rng.uniform(60.0, (size_px - 60) * gsd_m)
        radius = rng.uniform(20.0, 60.0)
        depth = rng.uniform(5.0, 15.0)

        dist = np.hypot(xx - cx, yy - cy)
        crater_mask = dist < radius
        dtm[crater_mask] -= depth * (1.0 - (dist[crater_mask] / radius) ** 2)

        rim_mask = (dist >= radius) & (dist < 1.3 * radius)
        rim_dist = (dist[rim_mask] - radius) / (0.3 * radius)
        dtm[rim_mask] += 0.25 * depth * (1.0 - rim_dist)

    noise = rng.normal(0.0, 1.0, size=(size_px, size_px))
    filtered_noise = gaussian_filter(noise, sigma=3.0)
    filtered_noise = (filtered_noise - np.mean(filtered_noise)) / (np.std(filtered_noise) + 1e-8)
    dtm += roughness_scale * filtered_noise

    dz_dy, dz_dx = np.gradient(dtm, gsd_m, gsd_m)
    norm = np.sqrt(1.0 + dz_dx * dz_dx + dz_dy * dz_dy)
    nx = -dz_dx / norm
    ny = -dz_dy / norm
    nz = 1.0 / norm

    inc_rad = np.radians(sun_inc_deg)
    az_rad = np.radians(sun_az_deg)
    sx = np.sin(inc_rad) * np.sin(az_rad)
    sy = np.sin(inc_rad) * np.cos(az_rad)
    sz = np.cos(inc_rad)

    cos_i = np.clip(nx * sx + ny * sy + nz * sz, 0.0, 1.0)
    albedo = 1.0 + 0.03 * rng.normal(0.0, 1.0, size=(size_px, size_px))
    ortho = np.clip((0.15 + 0.70 * cos_i) * albedo, 0.05, 0.95).astype(np.float32)
    dtm = dtm.astype(np.float32)

    transform = rasterio.transform.from_origin(500000.0, 1000000.0, gsd_m, gsd_m)
    crs = "EPSG:32631"

    dtm_path = out_dir / f"{site_id}_dtm.tif"
    ortho_path = out_dir / f"{site_id}_ortho.tif"

    tags = {
        "INCIDENCE_ANGLE": str(sun_inc_deg),
        "SOLAR_AZIMUTH_ANGLE": str(sun_az_deg),
    }

    with rasterio.open(
        dtm_path,
        "w",
        driver="GTiff",
        height=size_px,
        width=size_px,
        count=1,
        dtype=dtm.dtype,
        crs=crs,
        transform=transform,
        nodata=-9999.0,
    ) as dst:
        dst.write(dtm, 1)
        dst.update_tags(**tags)

    with rasterio.open(
        ortho_path,
        "w",
        driver="GTiff",
        height=size_px,
        width=size_px,
        count=1,
        dtype=ortho.dtype,
        crs=crs,
        transform=transform,
        nodata=-9999.0,
    ) as dst:
        dst.write(ortho, 1)
        dst.update_tags(**tags)

    return ortho_path, dtm_path


def generate_smoke_dataset(config: dict[str, Any], seed: int = 42) -> pd.DataFrame:
    """Generates 5 synthetic sites and preprocesses them into 200 tiles."""
    rng = np.random.default_rng(seed)
    smoke_cfg = copy.deepcopy(config)

    n_sites = smoke_cfg.get("smoke", {}).get("n_synthetic_images", 5)
    image_size_px = smoke_cfg.get("smoke", {}).get("image_size_px", 640)
    n_tiles_total = smoke_cfg.get("smoke", {}).get("n_tiles_total", 200)
    per_site_target = n_tiles_total // n_sites

    raw_dir = Path(smoke_cfg["paths"]["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)

    base_slopes = [3.0, 5.5, 7.5, 10.0, 14.0]
    sun_incs = [35.0, 45.0, 55.0, 40.0, 50.0]
    sun_azs = [110.0, 135.0, 160.0, 200.0, 225.0]

    verified_rows: list[dict[str, Any]] = []

    for i in range(n_sites):
        site_id = f"syn_site_{i}"
        site_dir = raw_dir / site_id
        slope = base_slopes[i % len(base_slopes)]
        s_inc = sun_incs[i % len(sun_incs)]
        s_az = sun_azs[i % len(sun_azs)]
        r_scale = 0.4 + 0.1 * i

        ortho_p, dtm_p = generate_synthetic_site_geotiffs(
            site_dir,
            site_id=site_id,
            base_slope_deg=slope,
            roughness_scale=r_scale,
            size_px=image_size_px,
            gsd_m=smoke_cfg["preprocessing"]["gsd_m"],
            sun_inc_deg=s_inc,
            sun_az_deg=s_az,
            rng=rng,
        )

        verified_rows.append(
            {
                "site_id": site_id,
                "ortho_local_path": str(ortho_p.resolve()),
                "dtm_local_path": str(dtm_p.resolve()),
                "sun_inc": s_inc,
                "sun_az": s_az,
            }
        )

    verified_df = pd.DataFrame(verified_rows)
    verified_csv = raw_dir / "verified_manifest.csv"
    verified_df.to_csv(verified_csv, index=False)

    smoke_cfg["preprocessing"]["train_stride_px"] = 64
    smoke_cfg["preprocessing"]["max_tiles_per_image"] = per_site_target

    all_records: list[dict[str, Any]] = []
    for _, row in verified_df.iterrows():
        records = preprocess_site(row, smoke_cfg, seed=seed)
        all_records.extend(records)

    processed_dir = Path(smoke_cfg["paths"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = processed_dir / "metadata.parquet"

    columns = [
        "tile_id",
        "image_id",
        "shard_path",
        "shard_idx",
        "row",
        "col",
        "lat",
        "lon",
        "sun_inc",
        "sun_az",
        "mean_slope_5m",
        "mean_slope_15m",
        "roughness_m",
        "hazard_fraction",
        "safe_label",
    ]
    if all_records:
        metadata_df = pd.DataFrame(all_records)[columns]
    else:
        metadata_df = pd.DataFrame(columns=columns)

    metadata_df.to_parquet(metadata_path, index=False)
    logging.info("Generated smoke dataset with %d tiles across %d sites.", len(metadata_df), n_sites)
    return metadata_df
