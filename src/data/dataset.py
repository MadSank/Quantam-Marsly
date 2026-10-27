"""PyTorch Dataset, train-only TargetScaler, and sun-consistent augmentations."""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class TargetScaler:
    """Standardizes multi-output regression targets using log1p + training-only z-score."""

    def __init__(self, target_cols: list[str], log1p_cols: list[str]) -> None:
        self.target_cols = list(target_cols)
        self.log1p_cols = list(log1p_cols)
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, train_df: pd.DataFrame) -> "TargetScaler":
        y = train_df[self.target_cols].to_numpy(dtype=np.float64).copy()
        for idx, col in enumerate(self.target_cols):
            if col in self.log1p_cols:
                y[:, idx] = np.log1p(np.maximum(y[:, idx], 0.0))

        self.mean_ = y.mean(axis=0)
        std = y.std(axis=0)
        self.std_ = np.where(std < 1e-8, 1.0, std)
        return self

    def transform(self, df_or_array: pd.DataFrame | np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("TargetScaler must be fit before calling transform.")

        if isinstance(df_or_array, pd.DataFrame):
            y = df_or_array[self.target_cols].to_numpy(dtype=np.float64).copy()
        else:
            y = np.array(df_or_array, dtype=np.float64).copy()

        for idx, col in enumerate(self.target_cols):
            if col in self.log1p_cols:
                y[:, idx] = np.log1p(np.maximum(y[:, idx], 0.0))

        y_std = (y - self.mean_) / self.std_
        return y_std.astype(np.float32)

    def inverse_transform(self, y_std: np.ndarray | torch.Tensor) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("TargetScaler must be fit before calling inverse_transform.")

        if isinstance(y_std, torch.Tensor):
            arr = y_std.detach().cpu().numpy().astype(np.float64).copy()
        else:
            arr = np.array(y_std, dtype=np.float64).copy()

        y_log = arr * self.std_ + self.mean_
        for idx, col in enumerate(self.target_cols):
            if col in self.log1p_cols:
                y_log[:, idx] = np.expm1(y_log[:, idx])

        # Clamping non-negative physical values
        y_log[:, 0] = np.maximum(y_log[:, 0], 0.0)  # slope
        y_log[:, 1] = np.maximum(y_log[:, 1], 0.0)  # roughness
        if y_log.shape[1] > 2:
            y_log[:, 2] = np.clip(y_log[:, 2], 0.0, 1.0)  # hazard fraction

        return y_log.astype(np.float32)


def derive_safe_decisions(
    preds_raw: np.ndarray, safe_slope_deg: float = 8.0, safe_hazard_frac: float = 0.05
) -> np.ndarray:
    """Derives binary landing safety decision from physical-unit predictions."""
    if preds_raw.ndim == 1:
        preds = preds_raw.reshape(1, -1)
    else:
        preds = preds_raw

    is_safe = (preds[:, 0] < safe_slope_deg) & (preds[:, 2] < safe_hazard_frac)
    return is_safe.astype(np.int64)


def transform_sun_azimuth(
    sun_az_deg: float, rot90_k: int = 0, flip_h: bool = False, flip_v: bool = False
) -> float:
    """Updates solar azimuth in lockstep with geometric 2D image transformations."""
    if np.isnan(sun_az_deg):
        return float("nan")

    az = (float(sun_az_deg) - 90.0 * (rot90_k % 4)) % 360.0
    if flip_h:
        az = (360.0 - az) % 360.0
    if flip_v:
        az = (180.0 - az) % 360.0

    return float(az)


class MarsTerrainDataset(Dataset):
    """PyTorch Dataset loading float16 ortho shards with augmentation and sun features."""

    def __init__(
        self,
        metadata_df: pd.DataFrame,
        scaler: TargetScaler | None = None,
        config: dict[str, Any] | None = None,
        is_train: bool = False,
        embeddings: np.ndarray | None = None,
        quanv_maps: np.ndarray | None = None,
    ) -> None:
        self.metadata = metadata_df.reset_index(drop=True)
        self.scaler = scaler
        self.config = config or {}
        self.is_train = is_train
        self.embeddings = embeddings
        self.quanv_maps = quanv_maps

        # Shard caching per unique shard_path
        self._shard_cache: dict[str, np.ndarray] = {}

        # Target preprocessing
        prep_cfg = self.config.get("preprocessing", {})
        self.target_cols = prep_cfg.get(
            "target_cols", ["mean_slope_5m", "roughness_m", "hazard_fraction"]
        )
        self.has_targets = all(col in self.metadata.columns for col in self.target_cols)

        if self.has_targets and self.scaler is not None:
            self.scaled_targets = self.scaler.transform(self.metadata).copy()
        elif self.has_targets:
            self.scaled_targets = self.metadata[self.target_cols].to_numpy(dtype=np.float32).copy()
        else:
            self.scaled_targets = None

        # Augmentation settings
        aug_cfg = self.config.get("augmentation", {})
        self.enable_photometric = bool(aug_cfg.get("enable_photometric", True))
        self.brightness_jitter = float(aug_cfg.get("brightness_jitter", 0.10))
        self.contrast_jitter = float(aug_cfg.get("contrast_jitter", 0.10))
        self.gaussian_noise_std = float(aug_cfg.get("gaussian_noise_std", 0.02))
        self.enable_geometric = bool(aug_cfg.get("enable_geometric", False))
        # Optional fold-fitted radiometric normalization for controlled
        # evaluations. Existing configs omit this and retain legacy behavior.
        norm_cfg = self.config.get("input_normalization", {})
        self.image_if_low = norm_cfg.get("if_low")
        self.image_if_high = norm_cfg.get("if_high")
        # Backward compatibility: historical fold configs omit `clip` and
        # retain their original clipped behavior. New controlled folds set
        # clip=False to preserve valid out-of-training-range I/F contrast.
        self.image_if_clip = bool(norm_cfg.get("clip", True))
        if (self.image_if_low is None) != (self.image_if_high is None):
            raise ValueError("input_normalization requires both if_low and if_high")
        if self.image_if_low is not None and float(self.image_if_high) <= float(self.image_if_low):
            raise ValueError("input_normalization.if_high must exceed if_low")

        # Solar feature verification
        features_cfg = self.config.get("features", {})
        use_sun = bool(features_cfg.get("use_sun_features", True))

        has_nan_sun = False
        if "sun_inc" in self.metadata.columns and "sun_az" in self.metadata.columns:
            has_nan_sun = bool(
                self.metadata["sun_inc"].isna().any() or self.metadata["sun_az"].isna().any()
            )
        else:
            has_nan_sun = True

        if use_sun and has_nan_sun:
            logging.warning(
                "Solar metadata contains NaNs. Disabling effective_use_sun for this dataset."
            )
            self.effective_use_sun = False
        else:
            self.effective_use_sun = use_sun

    def _get_shard(self, shard_path: str) -> np.ndarray:
        if shard_path not in self._shard_cache:
            p = Path(shard_path)
            if not p.exists():
                raise FileNotFoundError(f"Shard array not found: {p}")
            self._shard_cache[shard_path] = np.load(p, mmap_mode="r")
        return self._shard_cache[shard_path]

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.metadata.iloc[idx]
        shard_arr = self._get_shard(str(row["shard_path"]))
        tile_data = shard_arr[int(row["shard_idx"])].astype(np.float32).copy()

        if self.image_if_low is not None and tile_data.ndim == 3 and tile_data.shape[0] >= 3:
            image_valid = tile_data[2] > 0.0
            normalized = (tile_data[0] - float(self.image_if_low)) / (
                float(self.image_if_high) - float(self.image_if_low)
            )
            if self.image_if_clip:
                normalized = np.clip(normalized, 0.0, 1.0)
            tile_data[0] = normalized
            tile_data[0, ~image_valid] = 0.0

        sun_inc = float(row.get("sun_inc", float("nan")))
        sun_az = float(row.get("sun_az", float("nan")))

        # Augmentation (training only)
        if self.is_train:
            if self.enable_geometric:
                rot90_k = int(np.random.randint(0, 4))
                flip_h = bool(np.random.rand() > 0.5)
                flip_v = bool(np.random.rand() > 0.5)

                if rot90_k > 0:
                    tile_data = np.rot90(tile_data, k=rot90_k).copy()
                if flip_h:
                    tile_data = np.fliplr(tile_data).copy()
                if flip_v:
                    tile_data = np.flipud(tile_data).copy()

                sun_az = transform_sun_azimuth(
                    sun_az, rot90_k=rot90_k, flip_h=flip_h, flip_v=flip_v
                )

            if self.enable_photometric:
                b_offset = np.random.uniform(-self.brightness_jitter, self.brightness_jitter)
                c_factor = np.random.uniform(
                    1.0 - self.contrast_jitter, 1.0 + self.contrast_jitter
                )
                if tile_data.ndim == 3:
                    image_channel = tile_data[0]
                    noise = np.random.normal(0.0, self.gaussian_noise_std, size=image_channel.shape)
                    mean_val = np.mean(image_channel)
                    tile_data[0] = np.clip(
                        (image_channel - mean_val) * c_factor + mean_val + b_offset + noise,
                        0.0, 1.0,
                    ).astype(np.float32)
                else:
                    noise = np.random.normal(0.0, self.gaussian_noise_std, size=tile_data.shape)
                    mean_val = np.mean(tile_data)
                    tile_data = np.clip(
                        (tile_data - mean_val) * c_factor + mean_val + b_offset + noise,
                        0.0, 1.0,
                    ).astype(np.float32)

        img_tensor = torch.from_numpy(tile_data).float()
        if img_tensor.ndim == 2:
            img_tensor = img_tensor.unsqueeze(0)

        # Sun features
        if self.effective_use_sun and not np.isnan(sun_inc) and not np.isnan(sun_az):
            sun_features = torch.tensor(
                [
                    sun_inc / 90.0,
                    np.sin(np.radians(sun_az)),
                    np.cos(np.radians(sun_az)),
                ],
                dtype=torch.float32,
            )
            sun_angles_rad = torch.tensor(
                [
                    np.radians(sun_inc),
                    np.radians(sun_az) % np.pi,
                ],
                dtype=torch.float32,
            )
        else:
            sun_features = torch.zeros(3, dtype=torch.float32)
            sun_angles_rad = torch.zeros(2, dtype=torch.float32)

        sample: dict[str, Any] = {
            "image": img_tensor,
            "sun_features": sun_features,
            "sun_angles_rad": sun_angles_rad,
            "tile_id": str(row["tile_id"]),
            "image_id": str(row["image_id"]),
        }

        if self.scaled_targets is not None:
            sample["target"] = torch.from_numpy(self.scaled_targets[idx].copy()).float()
        if "safe_label" in row:
            sample["safe_label"] = int(row["safe_label"])

        if self.embeddings is not None:
            sample["embedding"] = torch.from_numpy(self.embeddings[idx].copy()).float()
        if self.quanv_maps is not None:
            sample["quanv_map"] = torch.from_numpy(self.quanv_maps[idx].copy()).float()

        return sample
