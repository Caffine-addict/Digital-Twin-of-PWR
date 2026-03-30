"""Save structured results under runs/*.json.

Lightweight:
- stdlib only
Deterministic:
- file naming is stable for the same scenario/config; if a file exists, version suffix is added.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional
import json


def _allocate_versioned(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    parent = path.parent
    for i in range(2, 10_000):
        p = parent / f"{stem}_v{i}{suffix}"
        if not p.exists():
            return p
    raise RuntimeError("failed to allocate versioned filename")


def save_json(*, runs_dir: str, base_name: str, payload: Dict[str, Any]) -> str:
    out_dir = Path(runs_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = _allocate_versioned(out_dir / f"{base_name}.json")
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return str(path)
