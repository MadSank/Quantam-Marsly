"""Centralized path definitions and validation for Mars and Moon artifacts."""

import os
from pathlib import Path
from typing import Dict

# Root resolution: allow environment variable override or infer relative to this file
_env_root = os.getenv("MARS_QCNN_ROOT")
if _env_root:
    PROJECT_ROOT = Path(_env_root).resolve()
else:
    # visualization_app/data_loading/paths.py -> parents[2] is PROJECT_ROOT
    PROJECT_ROOT = Path(__file__).resolve().parents[2]

APP_ROOT = PROJECT_ROOT / "visualization_app"

# Mars paths (Repaired retraining seedset)
MARS_REPAIRED_ROOT = PROJECT_ROOT / "runs" / "pipeline_repair_audit" / "retraining_seedset"
MARS_METRICS_DIR = MARS_REPAIRED_ROOT / "metrics"
MARS_RUNS_DIR = MARS_REPAIRED_ROOT / "runs"
MARS_CONFIGS_DIR = MARS_REPAIRED_ROOT / "configs"
MARS_PROTOCOL_DIR = MARS_REPAIRED_ROOT / "protocol"
MARS_PROGRESS_FILE = MARS_REPAIRED_ROOT / "progress.json"

MARS_CONTROLLED_PROCESSED = PROJECT_ROOT / "runs" / "controlled_evaluation" / "processed"
MARS_METADATA_PARQUET = MARS_CONTROLLED_PROCESSED / "metadata.parquet"
MARS_DATASET_METADATA = MARS_CONTROLLED_PROCESSED / "dataset_metadata.json"
MARS_SHARDS_DIR = MARS_CONTROLLED_PROCESSED / "shards"

# Moon paths (LROCNet experiment)
MOON_ROOT = PROJECT_ROOT / "data" / "raw" / "moon" / "lrocnet"
MOON_EXPERIMENT_ROOT = MOON_ROOT / "experiment_lunar_cnn_qcnn"
MOON_RUNS_DIR = MOON_EXPERIMENT_ROOT / "runs"
MOON_LOGS_DIR = MOON_RUNS_DIR / "logs"
MOON_METRICS_CSV = MOON_RUNS_DIR / "metrics.csv"
MOON_RECOMPUTED_METRICS_CSV = MOON_RUNS_DIR / "results_verification" / "recomputed_metrics.csv"
MOON_AUDIT_REPORT = MOON_RUNS_DIR / "results_verification" / "results_audit.md"
MOON_PROVENANCE_MANIFEST = MOON_ROOT / "provenance_manifest.csv"
MOON_CONFIG_YAML = MOON_EXPERIMENT_ROOT / "config.yaml"


def verify_critical_paths() -> Dict[str, bool]:
    """Verify presence of key input artifacts and return a status map."""
    paths_to_check = {
        "PROJECT_ROOT": PROJECT_ROOT,
        "MARS_REPAIRED_ROOT": MARS_REPAIRED_ROOT,
        "MARS_METRICS_DIR": MARS_METRICS_DIR,
        "MARS_RUNS_DIR": MARS_RUNS_DIR,
        "MARS_METADATA_PARQUET": MARS_METADATA_PARQUET,
        "MARS_SHARDS_DIR": MARS_SHARDS_DIR,
        "MOON_ROOT": MOON_ROOT,
        "MOON_LOGS_DIR": MOON_LOGS_DIR,
        "MOON_PROVENANCE_MANIFEST": MOON_PROVENANCE_MANIFEST,
    }
    return {name: p.exists() for name, p in paths_to_check.items()}
