from pathlib import Path
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "runs" / "pipeline_repair_audit"
META = ROOT / "runs" / "controlled_evaluation" / "processed" / "metadata.parquet"
sys.path.insert(0, str(ROOT))
from src.controlled_evaluation import fit_training_image_bounds
from src.data.splits import get_fold_indices
from src.data.dataset import MarsTerrainDataset

metadata = pd.read_parquet(META)
rows = []
normalizers = {}
preview_data = []
for fold in range(5):
    splits = get_fold_indices(metadata, test_fold=fold, n_splits=5)
    bounds = fit_training_image_bounds(metadata, splits["train"])
    normalizers[fold] = bounds
    cfg = {"input_normalization": bounds, "features": {"use_sun_features": False}, "augmentation": {"enable_photometric": False}}
    test = splits["test"]
    site = str(test.site_id.iloc[0])
    assert test.site_id.astype(str).nunique() == 1
    ds = MarsTerrainDataset(test, config=cfg, is_train=False)
    norm_values = [[], [], []]
    raw_valid = []
    raw_below = raw_above = raw_n = invalid_n = 0
    invalid_nonzero = 0
    preview = None
    for i in range(len(ds)):
        sample = ds[i]["image"].numpy()
        raw = ds._get_shard(str(test.iloc[i].shard_path))[int(test.iloc[i].shard_idx)].astype(np.float32)
        valid = raw[2] > 0
        img = raw[0][valid]
        if img.size:
            raw_valid.append(img)
            raw_below += int((img <= bounds["if_low"]).sum())
            raw_above += int((img >= bounds["if_high"]).sum())
            raw_n += int(img.size)
            norm_values[0].append(sample[0][valid])
        invalid_n += int((~valid).sum())
        invalid_nonzero += int(np.count_nonzero(sample[0][~valid]))
        norm_values[1].append(sample[1].reshape(-1))
        norm_values[2].append(sample[2].reshape(-1))
        if preview is None:
            preview = (raw[0].copy(), sample[0].copy(), raw[1].copy(), raw[2].copy(), str(test.iloc[i].tile_id))
    raw_values = np.concatenate(raw_valid)
    normalized_image = np.concatenate(norm_values[0])
    row = {
        "site": site, "fold": fold, "test_patches": len(test),
        "training_sites": ",".join(bounds["fit_sites"]),
        "training_patches_for_fit": bounds["fit_patch_count"],
        "training_valid_pixels_for_fit": bounds["fit_valid_pixel_count"],
        "if_low": bounds["if_low"], "if_high": bounds["if_high"],
        "valid_image_pixels": raw_n, "invalid_image_pixels": invalid_n,
        "invalid_image_fraction": invalid_n / (invalid_n + raw_n),
        "below_low_fraction_valid": raw_below / raw_n,
        "above_high_fraction_valid": raw_above / raw_n,
        "outside_bounds_fraction_valid": (raw_below + raw_above) / raw_n,
        "old_clip_fraction_exact_0_or_1": (raw_below + raw_above) / raw_n,
        "new_out_of_0_1_fraction_valid": float(((normalized_image < 0) | (normalized_image > 1)).mean()),
        "new_valid_if_min": float(normalized_image.min()), "new_valid_if_max": float(normalized_image.max()),
        "new_valid_if_mean": float(normalized_image.mean()), "new_valid_if_std": float(normalized_image.std()),
        "new_valid_if_q01": float(np.quantile(normalized_image, .01)),
        "new_valid_if_q50": float(np.quantile(normalized_image, .50)),
        "new_valid_if_q99": float(np.quantile(normalized_image, .99)),
        "invalid_pixels_nonzero_after_normalization": invalid_nonzero,
    }
    for ch in (1, 2):
        arr = np.concatenate(norm_values[ch])
        row.update({f"channel{ch}_min": float(arr.min()), f"channel{ch}_max": float(arr.max()),
                    f"channel{ch}_mean": float(arr.mean()), f"channel{ch}_std": float(arr.std()),
                    f"channel{ch}_valid_fraction": float((arr > 0).mean())})
    rows.append(row)
    preview_data.append((site, preview))

pd.DataFrame(rows).sort_values("site").to_csv(OUT / "preprocessing_audit.csv", index=False)
with (OUT / "fold_normalization_parameters.yaml").open("w", encoding="utf-8") as f:
    yaml.safe_dump({str(k): v for k, v in normalizers.items()}, f, sort_keys=False)
fig, axes = plt.subplots(5, 3, figsize=(12, 17), constrained_layout=True)
for r, (site, pack) in enumerate(sorted(preview_data)):
    raw_if, normalized, dtm_mask, image_mask, tile_id = pack
    raw_valid_mask = image_mask > 0
    raw_vmin, raw_vmax = np.quantile(raw_if[raw_valid_mask], [.02, .98])
    norm_vmin, norm_vmax = np.quantile(normalized[raw_valid_mask], [.02, .98])
    axes[r, 0].imshow(np.ma.masked_where(~raw_valid_mask, raw_if), cmap="gray", vmin=raw_vmin, vmax=raw_vmax)
    axes[r, 0].set_title(f"{site} calibrated I/F\n{tile_id}")
    axes[r, 1].imshow(np.ma.masked_where(~raw_valid_mask, normalized), cmap="gray", vmin=norm_vmin, vmax=norm_vmax)
    axes[r, 1].set_title("New normalized I/F\n(display-only valid-pixel q02–q98 stretch)")
    axes[r, 2].imshow(image_mask, cmap="gray", vmin=0, vmax=1)
    axes[r, 2].contour(dtm_mask, levels=[.5], colors=["cyan"], linewidths=.4)
    axes[r, 2].set_title("Image-valid mask; cyan = DTM-valid edge")
    for ax in axes[r]:
        ax.set_xticks([]); ax.set_yticks([])
fig.savefig(OUT / "preprocessing_previews.png", dpi=130)
plt.close(fig)
print(pd.DataFrame(rows).sort_values("site").to_string(index=False))
print(f"Saved {OUT / 'preprocessing_audit.csv'}")
print(f"Saved {OUT / 'fold_normalization_parameters.yaml'}")
print(f"Saved {OUT / 'preprocessing_previews.png'}")


