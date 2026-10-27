"""Milestone 7 Acceptance Tests: Methodology & Limitations Content Verification."""

import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))


def test_methodology_file_content():
    methodology_path = APP_DIR / "pages" / "6_Methodology.py"
    assert methodology_path.exists()
    
    content = methodology_path.read_text(encoding="utf-8")
    
    # Check for crucial honest guardrails and disclosures
    assert "Not an operational landing safety" in content or "Not a Validated Landing Safety" in content
    assert "No Quantum Speedup" in content or "No quantum speedup" in content
    assert "FILE_FACTS_CONFIRMED_SOURCE_ID_UNVERIFIED" in content
    assert "59,347" in content  # Corrected CNN parameter count
    assert "568" in content  # Moon Hybrid QCNN parameter count
    assert "547" in content  # Moon Frozen MLP parameter count
    assert "degrees of freedom" in content or "df = 4" in content
    assert "crosses zero" in content or "cross zero" in content
    assert "Leave-One-Site-Out" in content
    
    # Phase 2 Mandatory Disclosures:
    assert "Moon Test Embedding Computation Disclosure" in content
    assert "test labels were never used" in content
    assert "39.65%" in content  # Mars CNN domain shift test FPR
    assert "33.94%" in content  # Mars QCNN domain shift test FPR
    assert "render_walkthrough_footer" in content


def test_methodology_page_execution():
    """Verify that pages/6_Methodology.py compiles and runs without unhandled exceptions."""
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(APP_DIR / "pages" / "6_Methodology.py"), default_timeout=30)
    at.run()
    assert not at.exception, f"AppTest raised an exception: {at.exception}"
