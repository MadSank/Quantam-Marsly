"""Grouped stratified cross-validation splits by image_id."""

import numpy as np
import pandas as pd


def assign_grouped_folds(
    metadata_df: pd.DataFrame, n_splits: int = 5, seed: int = 42
) -> pd.DataFrame:
    """Assigns fold indices (0 to n_splits-1) grouped by image_id and stratified by mean slope."""
    df = metadata_df.copy()
    if len(df) == 0:
        df["fold"] = pd.Series(dtype=int)
        return df

    # Compute mean slope per image
    image_means = df.groupby("image_id")["mean_slope_5m"].mean().reset_index()
    image_means = image_means.sort_values(by="mean_slope_5m").reset_index(drop=True)

    rng = np.random.default_rng(seed)
    n_images = len(image_means)

    fold_assignments: dict[str, int] = {}
    for i in range(0, n_images, n_splits):
        chunk = image_means.iloc[i : i + n_splits]
        k = len(chunk)
        perm = rng.permutation(n_splits)
        for idx, (_, row) in enumerate(chunk.iterrows()):
            fold_assignments[str(row["image_id"])] = int(perm[idx])

    df["fold"] = df["image_id"].astype(str).map(fold_assignments).astype(int)
    return df


def get_fold_indices(
    metadata_df: pd.DataFrame, test_fold: int, n_splits: int = 5
) -> dict[str, pd.DataFrame]:
    """Partitions metadata into train, val, and test splits with strict image_id separation."""
    df = metadata_df.copy()
    if "fold" not in df.columns:
        df = assign_grouped_folds(df, n_splits=n_splits)

    val_fold = (test_fold + 1) % n_splits

    test_df = df[df["fold"] == test_fold].reset_index(drop=True)
    val_df = df[df["fold"] == val_fold].reset_index(drop=True)
    train_df = df[~df["fold"].isin([test_fold, val_fold])].reset_index(drop=True)

    # Mandatory runtime assertions: zero image_id leakage across partitions
    train_images = set(train_df["image_id"])
    val_images = set(val_df["image_id"])
    test_images = set(test_df["image_id"])

    assert train_images.isdisjoint(val_images), (
        f"Data leakage detected! Shared images between train and val: {train_images & val_images}"
    )
    assert train_images.isdisjoint(test_images), (
        f"Data leakage detected! Shared images between train and test: {train_images & test_images}"
    )
    assert val_images.isdisjoint(test_images), (
        f"Data leakage detected! Shared images between val and test: {val_images & test_images}"
    )

    return {
        "train": train_df,
        "val": val_df,
        "test": test_df,
    }
