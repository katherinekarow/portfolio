"""Load config.yaml (project root by default)."""
from __future__ import annotations

from pathlib import Path

import yaml

DEFAULT_PATH = Path(__file__).resolve().parents[2] / "config.yaml"


def load(path: str | Path | None = None) -> dict:
    with open(path or DEFAULT_PATH) as f:
        return yaml.safe_load(f)
