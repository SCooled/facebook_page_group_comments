"""Load config.yaml."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    cfg_path = Path(path) if path else ROOT / "config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    cfg["_root"] = str(ROOT)
    return cfg


def resolve(cfg: dict[str, Any], key: str) -> Path:
    """Resolve a *_dir config entry relative to the project root."""
    p = Path(cfg[key])
    return p if p.is_absolute() else Path(cfg["_root"]) / p
