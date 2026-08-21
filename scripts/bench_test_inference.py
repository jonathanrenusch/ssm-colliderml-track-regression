"""Standalone test-mode forward-pass timer using cuda_timer.

Bypasses the vendored ``InferenceTimer`` callback (which assumes every value
in the inputs dict has ``.shape[1]`` — breaks on packed mode's 1-D
``cu_seqlens`` / ``track_lengths``). Loads the same YAML config Lightning
would, instantiates the data module + model, runs over the *test* split,
and reports mean +/- std forward-pass time after a 10-step warmup.

Usage::

    cd /shared/tracking/ssm-colliderml-track-regression/src/track_regression
    CUDA_VISIBLE_DEVICES=N pixi run -e default python \\
        ../../scripts/bench_test_inference.py \\
        --config config/ssm_cls/pretrain_ssm_cls_packed.yaml \\
        --batch-size 30000 --num-workers 0 --warmup 10
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Triton cache redirect (matches train.py).
os.environ.setdefault("TRITON_CACHE_DIR", "/tmp/triton_cache")

import numpy as np
import torch
import yaml

HERE = Path(__file__).resolve().parent
SRC = (HERE.parent / "src").resolve()
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

torch.set_float32_matmul_precision(__import__("os").environ.get("TRK_MATMUL_PRECISION", "highest"))
if __import__("os").environ.get("TRK_MATMUL_PRECISION", "highest") == "highest":
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False

from track_regression._lib.cuda_timer import cuda_timer
from track_regression.data import ColliderMLRegrDataModule


def _deep_merge(a: dict, b: dict) -> dict:
    if not isinstance(a, dict) or not isinstance(b, dict):
        return b
    # A dict carrying class_path is a polymorphic node: the leaf wins
    # outright. Otherwise merging init_args from two different classes
    # (e.g. SSM base placeholder + transformer leaf) yields garbage.
    if "class_path" in b:
        return b
    out = dict(a)
    for k, v in b.items():
        out[k] = _deep_merge(out[k], v) if k in out else v
    return out


def _load_merged_config(config_path: Path) -> dict:
    """Merge sibling ``base.yaml`` with the leaf config (leaf wins)."""
    parts: list[dict] = []
    base = config_path.parent / "base.yaml"
    if base.exists():
        with open(base) as f:
            parts.append(yaml.safe_load(f) or {})
    with open(config_path) as f:
        parts.append(yaml.safe_load(f) or {})
    merged: dict = {}
    for p in parts:
        merged = _deep_merge(merged, p)
    return merged


def _instantiate(node):
    """Recursively turn ``class_path`` / ``init_args`` dicts into objects."""
    if isinstance(node, dict):
        if "class_path" in node:
            cls = _import(node["class_path"])
            init_args = node.get("init_args", {}) or {}
            init_args = {k: _instantiate(v) for k, v in init_args.items()}
            return cls(**init_args)
        return {k: _instantiate(v) for k, v in node.items()}
    if isinstance(node, list):
        return [_instantiate(v) for v in node]
    return node


def _import(dotted: str):
    mod, _, cls = dotted.rpartition(".")
    import importlib
    return getattr(importlib.import_module(mod), cls)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--batch-size", type=int, default=30000)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--max-batches", type=int, default=0,
                    help="0 = use the whole test set")
    args = ap.parse_args()

    cfg = _load_merged_config(args.config.resolve())

    # ---- Model ----
    model_cfg = cfg["model"]["model"]
    inner_model = _instantiate(model_cfg)  # TrackParameterRegressor
    inner_model = inner_model.cuda().eval()
    n_params = sum(p.numel() for p in inner_model.parameters())
    print(f"[bench] config: {args.config}")
    print(f"[bench] params: {n_params/1e6:.3f} M")
    print(f"[bench] packed_batches: {cfg['data'].get('packed_batches', True)}")
    print(f"[bench] encoder_autocast_dtype: {model_cfg['init_args'].get('encoder_autocast_dtype')}")

    # ---- Data module ----
    data_cfg = dict(cfg["data"])
    data_cfg["batch_size"] = args.batch_size
    data_cfg["num_workers"] = args.num_workers
    # num_train_shards isn't relevant for test stage but the constructor expects it.
    if data_cfg.get("num_train_shards", -1) == -1:
        # Keep default.
        pass
    dm = ColliderMLRegrDataModule(**data_cfg)
    dm.setup("test")
    loader = dm.test_dataloader()

    # ---- Run ----
    times_ms: list[float] = []
    seq_dims: list[int] = []
    n_batches = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            inputs, _ = batch
            inputs = {k: v.cuda(non_blocking=True) for k, v in inputs.items()}
            # Record sequence-length dim summed across the (B,T,...) tensors.
            dim = sum(v.shape[1] for v in inputs.values() if v.ndim >= 2)
            with cuda_timer(times_ms):
                _ = inner_model(inputs)
            seq_dims.append(dim)
            n_batches += 1
            if args.max_batches and n_batches >= args.max_batches:
                break

    torch.cuda.synchronize()
    if len(times_ms) <= args.warmup:
        print(f"[bench] ERROR: only {len(times_ms)} batches, need > {args.warmup}")
        sys.exit(1)

    timed = torch.tensor(times_ms[args.warmup:])
    mean = timed.mean().item()
    std = timed.std().item()
    p50 = float(np.percentile(timed.numpy(), 50))
    p10 = float(np.percentile(timed.numpy(), 10))
    p90 = float(np.percentile(timed.numpy(), 90))

    print("-" * 80)
    print(f"[bench] batches total      : {n_batches}")
    print(f"[bench] batches warmup     : {args.warmup}")
    print(f"[bench] batches measured   : {len(timed)}")
    print(f"[bench] batch_size         : {args.batch_size}")
    print(f"[bench] Mean inference time: {mean:.2f} +/- {std:.2f} ms")
    print(f"[bench] p10 / p50 / p90    : {p10:.2f} / {p50:.2f} / {p90:.2f} ms")
    print(f"[bench] tracks/sec (mean)  : {args.batch_size / (mean/1e3):.0f}")
    print("-" * 80)


if __name__ == "__main__":
    main()
