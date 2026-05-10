"""Load a named selection variant from ``selection_p200_datasets.yaml``."""

from __future__ import annotations

from pathlib import Path

import yaml


def load_selection_variant(path: str | Path, variant: str) -> dict:
    """Load a named selection variant from a multi-variant YAML file.

    Parameters
    ----------
    path : str | Path
        Path to a YAML file with top-level keys as variant names.
    variant : str
        Which variant to load (e.g. ``"core"``, ``"core_kf_matched"``).
    """
    p = Path(path)
    with open(p) as f:
        cfg = yaml.safe_load(f)
    if variant not in cfg:
        available = ", ".join(sorted(cfg.keys()))
        raise KeyError(f"Variant '{variant}' not found in {p}. Available: {available}")
    return dict(cfg[variant])
