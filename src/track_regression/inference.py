"""Load a trained model for inference from its training config and checkpoint."""

from __future__ import annotations

import importlib
from pathlib import Path

import torch
import yaml


def _instantiate(node):
    """Build the objects of a ``class_path`` / ``init_args`` config tree."""
    if isinstance(node, dict):
        if "class_path" in node:
            mod, _, cls = node["class_path"].rpartition(".")
            kwargs = {k: _instantiate(v) for k, v in (node.get("init_args") or {}).items()}
            return getattr(importlib.import_module(mod), cls)(**kwargs)
        return {k: _instantiate(v) for k, v in node.items()}
    if isinstance(node, list):
        return [_instantiate(v) for v in node]
    return node


def load_model(config: str | Path, ckpt: str | Path, device: str = "cuda",
               encoder_dtype: str | None = None):
    """``TrackParameterRegressor`` from a training config + Lightning checkpoint, in eval mode.

    ``encoder_dtype`` overrides the encoder autocast dtype (the paper's
    inference runs the encoder in ``float16``; the seed stays float64 and the
    heads fp32 regardless).
    """
    cfg = yaml.safe_load(open(config))
    model = _instantiate(cfg["model"]["model"])
    state = torch.load(ckpt, map_location="cpu", weights_only=False)["state_dict"]
    model.load_state_dict({k[len("model."):]: v for k, v in state.items() if k.startswith("model.")})
    if encoder_dtype is not None:
        model.encoder_autocast_dtype = getattr(torch, encoder_dtype)
    return model.eval().to(device)
