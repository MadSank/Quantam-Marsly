"""Co-registration, masking, Horn slope calculation, tiling, and normalization."""

import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rasterio
import rasterio.transform
import rasterio.warp
from rasterio.crs import CRS
from rasterio.enums import Resampling

from src.utils import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def get_metric_crs_and_transform(src: rasterio.DatasetReader, mars_radius_m: float = 3396190.0) -> CRS:
    """Returns metric CRS for raster, falling back to Mars equirectangular if non-metric."""
    if src.crs and src.crs.is_projected and src.crs.linear_units in {"m", "metre", "meter"}:
        return src.crs
    proj4_str = (
        f"+proj=eqc +lat_ts=0 +lat_0=0 +lon_0=0 +a={mars_radius_m} +b={mars_radius_m} +units=m +no_defs"
    )
    return CRS.from_string(proj4_str)


def coregister_and_resample(
    ortho_path: Path, dtm_path: Path, gsd_m: float = 0.5, mars_radius_m: float = 3396190.0
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Resamples orthoimage and DTM onto a shared metric grid."""
    with rasterio.open(ortho_path) as ortho_src, rasterio.open(dtm_path) as dtm_src:
        target_crs = get_metric_crs_and_transform(dtm_src, mars_radius_m=mars_radius_m)

        ortho_bounds = rasterio.warp.transform_bounds(ortho_src.crs, target_crs, *ortho_src.bounds)
        dtm_bounds = rasterio.warp.transform_bounds(dtm_src.crs, target_crs, *dtm_src.bounds)

        left = max(ortho_bounds[0], dtm_bounds[0])
        bottom = max(ortho_bounds[1], dtm_bounds[1])
        right = min(ortho_bounds[2], dtm_bounds[2])
        top = min(ortho_bounds[3], dtm_bounds[3])

        if right <= left or top <= bottom:
            raise ValueError(f"No spatial intersection between ortho ({ortho_path}) and DTM ({dtm_path})")

        width = int(np.floor((right - left) / gsd_m))
        height = int(np.floor((top - bottom) / gsd_m))
        if width <= 0 or height <= 0:
            raise ValueError(f"Intersection dimensions too small: width={width}, height={height}")

        dst_transform = rasterio.transform.from_origin(left, top, gsd_m, gsd_m)

        ortho_dst = np.full((height, width), np.nan, dtype=np.float32)
        dtm_dst = np.full((height, width), np.nan, dtype=np.float32)

        rasterio.warp.reproject(
            source=rasterio.band(ortho_src, 1),
            destination=ortho_dst,
            src_transform=ortho_src.transform,
            src_crs=ortho_src.crs,
            dst_transform=dst_transform,
            dst_crs=target_crs,
            resampling=Resampling.average,
            src_nodata=ortho_src.nodata,
            dst_nodata=np.nan,
        )

        rasterio.warp.reproject(
            source=rasterio.band(dtm_src, 1),
            destination=dtm_dst,
            src_transform=dtm_src.transform,
            src_crs=dtm_src.crs,
            dst_transform=dst_transform,
            dst_crs=target_crs,
            resampling=Resampling.bilinear,
            src_nodata=dtm_src.nodata,
            dst_nodata=np.nan,
        )

        meta_info = {
            "crs": str(target_crs),
            "transform": dst_transform,
            "bounds": (left, bottom, right, top),
            "width": width,
            "height": height,
            "ortho_nodata": ortho_src.nodata,
            "dtm_nodata": dtm_src.nodata,
        }

        return ortho_dst, dtm_dst, meta_info


def build_invalid_mask(
    ortho: np.ndarray,
    dtm: np.ndarray,
    ortho_nodata: float | None = None,
    dtm_nodata: float | None = None,
    seam_mult: float = 6.0,
    gsd_m: float = 0.5,
) -> np.ndarray:
    """Detects invalid, nodata, saturated, and seam artifact pixels."""
    invalid = ~np.isfinite(ortho) | ~np.isfinite(dtm)

    if ortho_nodata is not None and np.isfinite(ortho_nodata):
        invalid |= np.isclose(ortho, ortho_nodata)
    if dtm_nodata is not None and np.isfinite(dtm_nodata):
        invalid |= np.isclose(dtm, dtm_nodata)

    invalid |= dtm <= -1e30
    invalid |= ortho <= -1e30

    valid_ortho = ortho[~invalid]
    if len(valid_ortho) > 0:
        v_min = float(np.min(valid_ortho))
        v_max = float(np.max(valid_ortho))
        if v_max > v_min:
            v_9995 = float(np.percentile(valid_ortho, 99.95))
            if (
                np.isclose(v_9995, v_max, atol=1e-4)
                and (v_max >= 0.99 or v_max >= 254.0)
                and np.sum(valid_ortho >= v_max) > 10
            ):
                invalid |= ortho >= v_max

            if v_min <= 0.0:
                v_0005 = float(np.percentile(valid_ortho, 0.05))
                if np.isclose(v_0005, v_min, atol=1e-4) and np.sum(valid_ortho <= v_min) > 10:
                    invalid |= ortho <= v_min

    # Stitching-seam artifact detection on DTM via 1-pixel step jumps that immediately revert along a line segment
    dtm_clean = np.where(invalid, np.nan, dtm)
    if dtm_clean.shape[0] > 16 and dtm_clean.shape[1] > 16:
        diff_h = np.diff(dtm_clean, axis=1)
        diff_v = np.diff(dtm_clean, axis=0)

        valid_diff_h = np.abs(diff_h[np.isfinite(diff_h)])
        valid_diff_v = np.abs(diff_v[np.isfinite(diff_v)])
        if len(valid_diff_h) > 0 and len(valid_diff_v) > 0:
            all_diffs = np.concatenate([valid_diff_h, valid_diff_v])
        elif len(valid_diff_h) > 0:
            all_diffs = valid_diff_h
        else:
            all_diffs = valid_diff_v

        if len(all_diffs) > 100:
            med_diff = float(np.median(all_diffs))
            thresh = max(seam_mult * med_diff, 2.0 * gsd_m)

            if dtm_clean.shape[1] > 3:
                revert_h = (
                    (diff_h[:, 1:] * diff_h[:, :-1] < 0)
                    & (np.abs(diff_h[:, 1:]) > thresh)
                    & (np.abs(diff_h[:, :-1]) > thresh)
                )
                col_seam_active = np.sum(revert_h, axis=0) >= 16
                if np.any(col_seam_active):
                    for c_idx in np.where(col_seam_active)[0]:
                        invalid[:, c_idx + 1] = True

            if dtm_clean.shape[0] > 3:
                revert_v = (
                    (diff_v[1:, :] * diff_v[:-1, :] < 0)
                    & (np.abs(diff_v[1:, :]) > thresh)
                    & (np.abs(diff_v[:-1, :]) > thresh)
                )
                row_seam_active = np.sum(revert_v, axis=1) >= 16
                if np.any(row_seam_active):
                    for r_idx in np.where(row_seam_active)[0]:
                        invalid[r_idx + 1, :] = True

    return invalid


def compute_horn_slope_deg(
    dtm: np.ndarray, invalid_mask: np.ndarray, gsd_m: float, baseline_m: float
) -> np.ndarray:
    """Computes Horn's 3x3 weighted slope in degrees at a given baseline distance."""
    k = max(1, int(round(baseline_m / gsd_m)))
    slope_deg = np.full(dtm.shape, np.nan, dtype=np.float32)

    dtm_clean = np.where(invalid_mask, np.nan, dtm.astype(np.float64))
    h, w = dtm.shape
    if h < 2 * k + 1 or w < 2 * k + 1:
        return slope_deg

    z_nw = dtm_clean[:-2 * k, :-2 * k]
    z_n = dtm_clean[:-2 * k, k:-k]
    z_ne = dtm_clean[:-2 * k, 2 * k :]
    z_w = dtm_clean[k:-k, :-2 * k]
    z_e = dtm_clean[k:-k, 2 * k :]
    z_sw = dtm_clean[2 * k :, :-2 * k]
    z_s = dtm_clean[2 * k :, k:-k]
    z_se = dtm_clean[2 * k :, 2 * k :]

    denom = 8.0 * k * gsd_m
    dz_dx = ((z_ne + 2.0 * z_e + z_se) - (z_nw + 2.0 * z_w + z_sw)) / denom
    dz_dy = ((z_nw + 2.0 * z_n + z_ne) - (z_sw + 2.0 * z_s + z_se)) / denom

    slope_center = np.degrees(np.arctan(np.hypot(dz_dx, dz_dy)))
    slope_deg[k:-k, k:-k] = slope_center.astype(np.float32)
    return slope_deg


def compute_tile_roughness(dtm_tile: np.ndarray, valid_mask: np.ndarray, gsd_m: float) -> float:
    """Computes detrended RMS height residual relative to best-fit plane."""
    rows, cols = np.where(valid_mask)
    if len(rows) < 3:
        return 0.0

    x = cols * gsd_m
    y = rows * gsd_m
    z = dtm_tile[valid_mask].astype(np.float64)

    a_mat = np.column_stack([np.ones_like(x, dtype=np.float64), x.astype(np.float64), y.astype(np.float64)])
    try:
        coeffs, _, _, _ = np.linalg.lstsq(a_mat, z, rcond=None)
        residual = z - (a_mat @ coeffs)
        return float(np.sqrt(np.mean(residual ** 2)))
    except np.linalg.LinAlgError:
        return 0.0


def compute_tile_targets(
    dtm_tile: np.ndarray,
    slope_5m_tile: np.ndarray,
    slope_15m_tile: np.ndarray,
    valid_mask: np.ndarray,
    config: dict[str, Any],
) -> dict[str, float | int] | None:
    """Computes targets for a single tile; returns None if masked fraction exceeds threshold."""
    prep_cfg = config["preprocessing"]
    max_masked = prep_cfg["max_masked_fraction"]
    gsd_m = prep_cfg["gsd_m"]
    hazard_slope_deg = prep_cfg["hazard_slope_deg"]
    safe_mean_slope_deg = prep_cfg["safe_mean_slope_deg"]
    safe_hazard_fraction = prep_cfg["safe_hazard_fraction"]

    tile_size = dtm_tile.size
    valid_count = np.count_nonzero(valid_mask)
    masked_fraction = 1.0 - (valid_count / float(tile_size))

    if masked_fraction > max_masked:
        return None

    valid_slope_5m = slope_5m_tile[valid_mask]
    finite_mask_5m = np.isfinite(valid_slope_5m)
    if np.mean(finite_mask_5m) < (1.0 - max_masked):
        return None

    clean_slopes_5m = valid_slope_5m[finite_mask_5m]
    if len(clean_slopes_5m) == 0:
        return None

    mean_slope_5m = float(np.mean(clean_slopes_5m))

    valid_slope_15m = slope_15m_tile[valid_mask]
    finite_mask_15m = np.isfinite(valid_slope_15m)
    if np.any(finite_mask_15m):
        mean_slope_15m = float(np.mean(valid_slope_15m[finite_mask_15m]))
    else:
        mean_slope_15m = mean_slope_5m

    roughness_m = compute_tile_roughness(dtm_tile, valid_mask, gsd_m)
    hazard_fraction = float(np.mean(clean_slopes_5m > hazard_slope_deg))
    safe_label = int(
        (mean_slope_5m < safe_mean_slope_deg) and (hazard_fraction < safe_hazard_fraction)
    )

    return {
        "mean_slope_5m": mean_slope_5m,
        "mean_slope_15m": mean_slope_15m,
        "roughness_m": roughness_m,
        "hazard_fraction": hazard_fraction,
        "safe_label": safe_label,
    }


def normalize_ortho_percentile(
    ortho: np.ndarray, invalid_mask: np.ndarray, p_low: float = 2.0, p_high: float = 98.0
) -> np.ndarray:
    """Normalizes ortho intensities with 2-98 percentile stretch clipped to [0, 1]."""
    valid_pixels = ortho[~invalid_mask & np.isfinite(ortho)]
    if len(valid_pixels) == 0:
        return np.zeros_like(ortho, dtype=np.float32)

    vmin, vmax = np.percentile(valid_pixels, [p_low, p_high])
    scale = max(float(vmax - vmin), 1e-6)
    norm = np.clip((ortho - vmin) / scale, 0.0, 1.0).astype(np.float32)
    norm[invalid_mask] = 0.0
    return norm


def stratified_subsample_tiles(
    candidates: list[dict[str, Any]], max_tiles: int, n_quantiles: int = 10, seed: int = 42
) -> list[dict[str, Any]]:
    """Subsamples tiles across slope quantiles to preserve hazardous/steep tails."""
    if len(candidates) <= max_tiles:
        return candidates

    slopes = [c["mean_slope_5m"] for c in candidates]
    s_series = pd.Series(slopes)
    try:
        bins = pd.qcut(s_series, q=n_quantiles, labels=False, duplicates="drop")
    except ValueError:
        bins = pd.Series(0, index=s_series.index)

    rng = np.random.default_rng(seed)
    bin_indices: dict[int, list[int]] = {}
    for idx, b in enumerate(bins):
        bin_indices.setdefault(b, []).append(idx)

    selected_indices: list[int] = []
    unique_bins = sorted(bin_indices.keys())
    per_bin_quota = max(1, max_tiles // len(unique_bins))

    for b in unique_bins:
        indices = bin_indices[b]
        rng.shuffle(indices)
        selected_indices.extend(indices[:per_bin_quota])

    remaining_budget = max_tiles - len(selected_indices)
    if remaining_budget > 0:
        unused = list(set(range(len(candidates))) - set(selected_indices))
        rng.shuffle(unused)
        selected_indices.extend(unused[:remaining_budget])
    elif len(selected_indices) > max_tiles:
        rng.shuffle(selected_indices)
        selected_indices = selected_indices[:max_tiles]

    selected_indices.sort()
    return [candidates[i] for i in selected_indices]


def preprocess_site(
    site_row: pd.Series, config: dict[str, Any], seed: int = 42
) -> list[dict[str, Any]]:
    """Runs complete preprocessing pipeline for one site and writes tiles shard."""
    site_id = str(site_row["site_id"]).strip()
    ortho_path = Path(site_row["ortho_local_path"])
    dtm_path = Path(site_row["dtm_local_path"])
    sun_inc = float(site_row.get("sun_inc", float("nan")))
    sun_az = float(site_row.get("sun_az", float("nan")))

    prep_cfg = config["preprocessing"]
    gsd_m = prep_cfg["gsd_m"]
    tile_px = prep_cfg["tile_px"]
    stride_px = prep_cfg.get("train_stride_px", tile_px // 2)
    p_low = prep_cfg.get("percentile_low", 2.0)
    p_high = prep_cfg.get("percentile_high", 98.0)
    seam_mult = prep_cfg.get("seam_gradient_multiplier", 6.0)
    mars_radius_m = prep_cfg.get("mars_radius_m", 3396190.0)
    max_tiles = prep_cfg.get("max_tiles_per_image", 1500)
    n_quantiles = prep_cfg.get("stratify_quantiles", 10)

    ortho, dtm, meta = coregister_and_resample(
        ortho_path, dtm_path, gsd_m=gsd_m, mars_radius_m=mars_radius_m
    )
    invalid_mask = build_invalid_mask(
        ortho,
        dtm,
        ortho_nodata=meta["ortho_nodata"],
        dtm_nodata=meta["dtm_nodata"],
        seam_mult=seam_mult,
        gsd_m=gsd_m,
    )

    slope_5m = compute_horn_slope_deg(dtm, invalid_mask, gsd_m=gsd_m, baseline_m=5.0)
    slope_15m = compute_horn_slope_deg(dtm, invalid_mask, gsd_m=gsd_m, baseline_m=15.0)
    norm_ortho = normalize_ortho_percentile(ortho, invalid_mask, p_low=p_low, p_high=p_high)

    h, w = ortho.shape
    candidates: list[dict[str, Any]] = []
    tile_arrays: list[np.ndarray] = []

    transform = meta["transform"]

    for r in range(0, h - tile_px + 1, stride_px):
        for c in range(0, w - tile_px + 1, stride_px):
            dtm_tile = dtm[r : r + tile_px, c : c + tile_px]
            slope_5m_tile = slope_5m[r : r + tile_px, c : c + tile_px]
            slope_15m_tile = slope_15m[r : r + tile_px, c : c + tile_px]
            valid_mask_tile = ~invalid_mask[r : r + tile_px, c : c + tile_px]

            targets = compute_tile_targets(
                dtm_tile, slope_5m_tile, slope_15m_tile, valid_mask_tile, config
            )
            if targets is None:
                continue

            ortho_tile = norm_ortho[r : r + tile_px, c : c + tile_px]

            center_coords = transform * (c + tile_px / 2.0, r + tile_px / 2.0)
            center_x, center_y = center_coords[0], center_coords[1]

            tile_id = f"{site_id}_r{r}_c{c}"
            record = {
                "tile_id": tile_id,
                "image_id": site_id,
                "row": r,
                "col": c,
                "lat": center_y,
                "lon": center_x,
                "sun_inc": sun_inc,
                "sun_az": sun_az,
                **targets,
            }
            candidates.append(record)
            tile_arrays.append(ortho_tile)

    if not candidates:
        logging.warning("Site %s produced 0 valid tiles.", site_id)
        return []

    if len(candidates) > max_tiles:
        indexed_candidates = [{"orig_idx": idx, **cand} for idx, cand in enumerate(candidates)]
        subsampled = stratified_subsample_tiles(
            indexed_candidates, max_tiles=max_tiles, n_quantiles=n_quantiles, seed=seed
        )
        selected_indices = [item["orig_idx"] for item in subsampled]
        final_candidates = [candidates[i] for i in selected_indices]
        final_tiles = [tile_arrays[i] for i in selected_indices]
    else:
        final_candidates = candidates
        final_tiles = tile_arrays

    processed_dir = Path(config["paths"]["processed_dir"])
    shards_dir = processed_dir / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)
    shard_path = shards_dir / f"{site_id}_tiles.npy"

    stacked_tiles = np.stack(final_tiles, axis=0).astype(np.float16)
    np.save(shard_path, stacked_tiles)

    rel_shard_path = str(shard_path)
    for idx, cand in enumerate(final_candidates):
        cand["shard_path"] = rel_shard_path
        cand["shard_idx"] = idx

    logging.info("Site %s saved %d tiles to shard %s", site_id, len(final_candidates), shard_path)
    return final_candidates


def run_preprocessing(config: dict[str, Any]) -> pd.DataFrame:
    """Runs preprocessing for all sites in verified manifest and creates metadata.parquet."""
    raw_dir = Path(config["paths"]["raw_dir"])
    verified_csv = raw_dir / "verified_manifest.csv"
    if not verified_csv.exists():
        raise FileNotFoundError(f"Verified manifest not found at {verified_csv}. Run download first.")

    manifest_df = pd.read_csv(verified_csv)
    all_tile_records: list[dict[str, Any]] = []

    for _, site_row in manifest_df.iterrows():
        records = preprocess_site(site_row, config)
        all_tile_records.extend(records)

    processed_dir = Path(config["paths"]["processed_dir"])
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
    if all_tile_records:
        metadata_df = pd.DataFrame(all_tile_records)[columns]
    else:
        metadata_df = pd.DataFrame(columns=columns)

    metadata_df.to_parquet(metadata_path, index=False)
    logging.info("Wrote %d tiles metadata to %s", len(metadata_df), metadata_path)
    return metadata_df


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Preprocess Mars terrain rasters into tiles and metadata.")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file.")
    parser.add_argument("--override", nargs="*", default=None, help="Config overrides key=value.")
    args = parser.parse_args(argv)

    cfg = load_config(args.config, overrides=args.override)
    run_preprocessing(cfg)


if __name__ == "__main__":
    main()
