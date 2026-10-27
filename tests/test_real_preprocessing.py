"""Focused checks for the isolated real-data preprocessing helpers."""

import numpy as np
import pandas as pd
from pathlib import Path

from src.data.preprocess_real import horn_window, valid_dtm
from src.data.dataset import MarsTerrainDataset
from src.evaluate import compute_run_metrics


def test_real_valid_mask_excludes_pds_nodata_and_nonfinite() -> None:
    nodata = -3.4028226550889045e38
    values = np.array([[1.0, nodata, np.nan], [2.0, -1e31, 3.0]], dtype=np.float32)
    mask = valid_dtm(values, nodata)
    assert mask.tolist() == [[True, False, False], [True, False, True]]


def test_native_horn_window_uses_physical_pixel_scale_and_propagates_nodata() -> None:
    scale = 1.0
    slope = 12.5
    z = np.tile(np.arange(41, dtype=np.float32) * np.tan(np.radians(slope)), (41, 1))
    valid = np.ones_like(z, dtype=bool)
    result = horn_window(z, valid, scale, baseline_m=5.0)
    finite = result[np.isfinite(result)]
    assert finite.size > 0
    assert abs(float(finite.mean()) - slope) < 1e-3
    valid[20, 20] = False
    result = horn_window(z, valid, scale, baseline_m=5.0)
    assert np.isnan(result[20, 20])
    assert np.isnan(result[15:26, 15:26]).any()


def test_undefined_safe_metric_denominators_are_nan() -> None:
    pred = pd.DataFrame(
        {
            "true_mean_slope_5m": [1.0], "pred_mean_slope_5m": [1.2],
            "true_roughness_m": [0.1], "pred_roughness_m": [0.2],
            "true_hazard_fraction": [0.0], "pred_hazard_fraction": [0.1],
            "true_safe": [0], "pred_safe": [0],
        }
    )
    metrics = compute_run_metrics(pred, ["mean_slope_5m", "roughness_m", "hazard_fraction"])
    assert np.isnan(metrics["safe_recall"])
    assert np.isnan(metrics["false_safe_rate"])


def test_real_mask_channels_are_preserved_by_photometric_augmentation(tmp_path: Path) -> None:
    shard = np.zeros((1, 3, 8, 8), dtype=np.float16)
    shard[0, 0] = 0.5
    shard[0, 1] = 1.0
    shard[0, 2] = 1.0
    shard[0, 1, 0, 0] = 0.0
    shard[0, 2, 0, 1] = 0.0
    shard_path = tmp_path / "masked.npy"
    np.save(shard_path, shard)
    meta = pd.DataFrame([{"tile_id":"p0","image_id":"site_01","shard_path":str(shard_path),"shard_idx":0}])
    cfg = {
        "preprocessing":{"target_cols":["mean_slope_5m","roughness_m","hazard_fraction"]},
        "augmentation":{"enable_photometric":True,"brightness_jitter":0.1,"contrast_jitter":0.1,"gaussian_noise_std":0.02},
        "features":{"use_sun_features":False},
    }
    sample = MarsTerrainDataset(meta, config=cfg, is_train=True)[0]
    image = sample["image"].numpy()
    assert image.shape == (3,8,8)
    assert image[1,0,0] == 0.0 and image[2,0,1] == 0.0
    assert np.isclose(image[1].mean(), 63.0/64.0)
    assert np.isclose(image[2].mean(), 63.0/64.0)
