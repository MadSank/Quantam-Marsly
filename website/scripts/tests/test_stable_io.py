"""stable_io: unchanged content (apart from generated_at) is never rewritten."""

from __future__ import annotations

import json
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))

from stable_io import write_bytes, write_json  # noqa: E402


def test_json_keeps_original_stamp_when_content_unchanged(tmp_path):
    path = tmp_path / "a.json"
    assert write_json(path, {"generated_at": "2026-01-01T00:00:00+00:00", "x": [1, 2.5, None]})
    assert not write_json(path, {"generated_at": "2026-02-02T00:00:00+00:00", "x": [1, 2.5, None]})
    assert json.loads(path.read_text(encoding="utf-8"))["generated_at"] == "2026-01-01T00:00:00+00:00"


def test_json_rewritten_when_content_changes(tmp_path):
    path = tmp_path / "a.json"
    write_json(path, {"generated_at": "t1", "x": 1})
    assert write_json(path, {"generated_at": "t2", "x": 2})
    assert json.loads(path.read_text(encoding="utf-8")) == {"generated_at": "t2", "x": 2}


def test_compact_and_indented_are_the_same_content(tmp_path):
    path = tmp_path / "a.json"
    write_json(path, {"x": [1, 2]}, compact=True)
    assert not write_json(path, {"x": [1, 2]}, compact=False)


def test_bytes_written_only_when_different(tmp_path):
    path = tmp_path / "a.bin"
    assert write_bytes(path, b"abc")
    assert not write_bytes(path, b"abc")
    assert write_bytes(path, b"abd")
