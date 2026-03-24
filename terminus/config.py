"""Configuration loading from YAML."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"


def load_config(path: Optional[Path] = None) -> dict:
    """Load pipeline config from a YAML file."""
    path = path or DEFAULT_CONFIG
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
