"""Tests for analytical tilted planes, Horn's slope, roughness, and nodata masking."""

from pathlib import Path

import numpy as np

from src.data.preprocess import (
    build_invalid_mask,
    compute_horn_slope_deg,
    compute_tile_roughness,
    compute_tile_targets,
    coregister_and_resample,
)
from src.data.synthetic import create_tilted_plane_geotiffs
from src.utils import load_config


def test_tilted_plane_slope_and_roughness(tmp_path: Path) -> None:
    """Verifies that an analytical tilted plane recovers exact slope within 0.01 deg and zero roughness."""
    target_slope = 12.5
    azimuth = 90.0
    size_px = 512
    gsd_m = 0.5

    ortho_p, dtm_p = create_tilted_plane_geotiffs(
        out_dir=tmp_path,
        site_id="test_plane",
        slope_deg=target_slope,
        azimuth_deg=azimuth,
        size_px=size_px,
        gsd_m=gsd_m,
    )

    ortho, dtm, meta = coregister_and_resample(ortho_p, dtm_p, gsd_m=gsd_m)
    invalid_mask = build_invalid_mask(ortho, dtm, ortho_nodata=meta["ortho_nodata"], dtm_nodata=meta["dtm_nodata"])

    slope_5m = compute_horn_slope_deg(dtm, invalid_mask, gsd_m=gsd_m, baseline_m=5.0)
    slope_15m = compute_horn_slope_deg(dtm, invalid_mask, gsd_m=gsd_m, baseline_m=15.0)

    # Valid interior pixels where Horn stencils did not touch boundary
    valid_5m = slope_5m[np.isfinite(slope_5m)]
    valid_15m = slope_15m[np.isfinite(slope_15m)]

    assert len(valid_5m) > 0, "No valid 5m slope pixels computed."
    assert len(valid_15m) > 0, "No valid 15m slope pixels computed."

    mean_slope_5m = float(np.mean(valid_5m))
    mean_slope_15m = float(np.mean(valid_15m))

    assert abs(mean_slope_5m - target_slope) < 1e-2, (
        f"Horn 5m slope {mean_slope_5m} deviated from target {target_slope}"
    )
    assert abs(mean_slope_15m - target_slope) < 1e-2, (
        f"Horn 15m slope {mean_slope_15m} deviated from target {target_slope}"
    )

    valid_mask = ~invalid_mask
    roughness = compute_tile_roughness(dtm, valid_mask, gsd_m=gsd_m)
    assert roughness < 1e-3, f"Expected roughness < 1e-3 on perfect plane, got {roughness}"


def test_masked_tile_dropping() -> None:
    """Verifies that injecting > 5% nodata into a tile causes compute_tile_targets to drop it (return None)."""
    cfg = load_config("configs/default.yaml")
    tile_size = 256
    dtm_tile = np.ones((tile_size, tile_size), dtype=np.float32) * 100.0
    slope_5m_tile = np.ones((tile_size, tile_size), dtype=np.float32) * 5.0
    slope_15m_tile = np.ones((tile_size, tile_size), dtype=np.float32) * 5.0

    valid_mask = np.ones((tile_size, tile_size), dtype=bool)

    # Inject a 10% nodata patch (exceeds max_masked_fraction = 0.05)
    patch_rows = int(tile_size * 0.35)
    patch_cols = int(tile_size * 0.35)
    valid_mask[:patch_rows, :patch_cols] = False

    masked_frac = 1.0 - (np.count_nonzero(valid_mask) / (tile_size * tile_size))
    assert masked_frac > 0.05, f"Masked fraction {masked_frac} is not > 0.05"

    targets = compute_tile_targets(
        dtm_tile=dtm_tile,
        slope_5m_tile=slope_5m_tile,
        slope_15m_tile=slope_15m_tile,
        valid_mask=valid_mask,
        config=cfg,
    )

    assert targets is None, f"Expected tile to be dropped (None), but got {targets}"
