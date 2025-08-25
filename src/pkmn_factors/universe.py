from __future__ import annotations

from pathlib import Path
import csv
from typing import Set


def load_universe(path: Path | str) -> Set[str]:
    """
    Load a simple CSV with columns:
      - card_key (required)
      - enabled (optional; "true"/"false", defaults to true if missing)
    Returns a set of enabled card keys.
    """
    p = Path(path)
    if not p.exists():
        # empty universe if file missing; callers may decide how to handle
        return set()

    enabled: set[str] = set()
    with p.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row.get("card_key") or "").strip()
            if not key:
                continue
            flag = (row.get("enabled") or "true").strip().lower()
            if flag in {"1", "true", "yes", "y"}:
                enabled.add(key)
    return enabled
