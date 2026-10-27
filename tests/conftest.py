"""Shared pytest fixtures for Mars QCNN testing."""

import copy
from pathlib import Path
from typing import Any, Generator

import pytest

from src.data.synthetic import generate_smoke_dataset
from src.utils import load_config


@pytest.fixture(scope="session")
def synthetic_workspace(tmp_path_factory: pytest.TempPathFactory) -> Generator[dict[str, Any], None, None]:
    """Session fixture creating a temporary workspace with 5 synthetic sites and 200 tiles."""
    tmp_dir = tmp_path_factory.mktemp("mars_smoke_workspace")

    base_cfg = load_config("configs/default.yaml", smoke=True)
    cfg = copy.deepcopy(base_cfg)

    cfg["paths"]["manifest_path"] = str(tmp_dir / "manifest.csv")
    cfg["paths"]["raw_dir"] = str(tmp_dir / "raw")
    cfg["paths"]["processed_dir"] = str(tmp_dir / "processed")
    cfg["paths"]["cache_dir"] = str(tmp_dir / "cache")
    cfg["paths"]["runs_dir"] = str(tmp_dir / "runs")
    cfg["paths"]["results_dir"] = str(tmp_dir / "results")

    # Generate synthetic smoke dataset (5 sites, 200 tiles)
    metadata_df = generate_smoke_dataset(cfg, seed=42)

    yield {
        "config": cfg,
        "metadata_df": metadata_df,
        "tmp_dir": tmp_dir,
    }
