"""YAML config loading utilities."""

from pathlib import Path

import yaml


def load_config(path: str | Path) -> dict:
    """Load a YAML config file."""
    with open(path) as f:
        return yaml.safe_load(f)
