"""Milestone 3 Acceptance Tests: Training Data Explorer."""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from data_loading.mars_data import (
    load_mars_metadata,
    load_mars_patch_image,
    FOLD_SITE_MAPPING,
)
from data_loading.moon_data import (
    load_moon_provenance,
    load_moon_image,
)


def test_mars_site_selection_and_stretching():
    df_mars = load_mars_metadata()
    site1_patches = df_mars[df_mars["site_id"] == "site_01"]
    assert len(site1_patches) == 2195
    
    # Test patch image stretch
    sample_row = site1_patches.iloc[0]
    img = load_mars_patch_image(sample_row["shard_path"], int(sample_row["shard_idx"]))
    assert img.shape == (64, 64)
    assert img.dtype == np.uint8
    # Stretch produces values spanning the display spectrum
    assert img.max() > img.min()


def test_mars_fold_mapping():
    # Verify fold 0: test=site_01, val=site_02, train=[site_03, site_04, site_05]
    f0 = FOLD_SITE_MAPPING[0]
    assert f0["test"] == "site_01"
    assert f0["val"] == "site_02"
    assert set(f0["train"]) == {"site_03", "site_04", "site_05"}
    
    # Verify all 5 folds cover all 5 sites uniquely as test
    test_sites = [FOLD_SITE_MAPPING[f]["test"] for f in range(5)]
    assert set(test_sites) == {"site_01", "site_02", "site_03", "site_04", "site_05"}


def test_moon_provenance_and_images():
    df_prov = load_moon_provenance()
    assert len(df_prov) == 5000
    
    # Check provenance status
    assert (df_prov["provenance_status"] == "FILE_FACTS_CONFIRMED_SOURCE_ID_UNVERIFIED").all()
    
    # Verify unverified sites/obs IDs are indeed NaN/unverified
    assert df_prov["source_observation_id"].isna().all()
    assert df_prov["site_id"].isna().all()
    
    # Verify test ejecta images load cleanly
    ejecta_test = df_prov[(df_prov["existing_split"] == "test") & (df_prov["class_label"] == "ejecta")]
    assert len(ejecta_test) == 89
    
    first_ejecta = ejecta_test.iloc[0]["relative_image_path"]
    img = load_moon_image(first_ejecta)
    assert img.size == (227, 227)


def test_training_explorer_page_execution():
    """Verify that pages/2_Training_Data_Explorer.py compiles and runs without unhandled exceptions."""
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(APP_DIR / "pages" / "2_Training_Data_Explorer.py"), default_timeout=30)
    at.run()
    assert not at.exception, f"AppTest raised an exception: {at.exception}"
