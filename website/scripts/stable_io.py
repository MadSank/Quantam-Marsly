"""Write export files only when their content changes, so re-exports don't churn git.

A JSON file whose content is unchanged apart from its top-level ``generated_at``
stamp is left as is, keeping the original stamp.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

STAMP_KEY = "generated_at"


def _without_stamp(obj: Any) -> Any:
    return {k: v for k, v in obj.items() if k != STAMP_KEY} if isinstance(obj, dict) else obj


def write_json(path: Path, payload: Any, *, compact: bool = False) -> bool:
    """Serialise ``payload``; write only if the content differs. Returns True if written."""
    text = json.dumps(payload, ensure_ascii=False, allow_nan=False,
                      separators=(",", ":") if compact else None, indent=None if compact else 1) + "\n"
    if path.exists():
        old = path.read_text(encoding="utf-8")
        if old == text or _without_stamp(json.loads(old)) == _without_stamp(json.loads(text)):
            return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return True


def write_bytes(path: Path, data: bytes) -> bool:
    """Write ``data`` only if the file is missing or different. Returns True if written."""
    if path.exists() and path.read_bytes() == data:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return True
