"""Shared helpers for the GPU kernel-optimization perf harness.

Provides config loading/merging (lifted from ``scripts/bench_test_inference.py``,
semantics preserved: sibling ``base.yaml`` merge, ``class_path``-aware override),
the data-path override for the known-broken preprocessed_dir in shipped configs,
result-sink helpers (jsonl / csv / atomic json), precision-flag pinning matching
production numerics (``train.py:26``), and an environment fingerprint for every
result row.

Import from sibling scripts as::

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import common
"""
from __future__ import annotations

import csv
import json
import os
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"

# Configs reference this dir, which does NOT exist on this node (CLAUDE.md
# "Data-path trap"). The real dataset lives at the replacement path.
DATA_DIR_BROKEN_FRAGMENT = "p200_core_kf_hits_finetune"
DATA_DIR_OVERRIDE = "/scratch/colliderml/arxiv_retraining/p200_core_kf_matched_finetune"


def ensure_src_on_path() -> None:
    """Make ``track_regression`` importable from a script under scripts/perf/."""
    s = str(SRC)
    if s not in sys.path:
        sys.path.insert(0, s)


# ---------------------------------------------------------------------------
# Config loading (lifted from scripts/bench_test_inference.py — keep semantics)
# ---------------------------------------------------------------------------

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


def _import(dotted: str):
    mod, _, cls = dotted.rpartition(".")
    import importlib

    return getattr(importlib.import_module(mod), cls)


def _instantiate(node):
    """Recursively turn ``class_path`` / ``init_args`` dicts into objects.

    Saved Lightning configs from old runs may carry init_args that no longer
    exist on the class (e.g. ``sort_field`` removed after May); those are
    dropped with a loud warning instead of failing the whole bench.
    """
    if isinstance(node, dict):
        if "class_path" in node:
            cls = _import(node["class_path"])
            init_args = node.get("init_args", {}) or {}
            init_args = {k: _instantiate(v) for k, v in init_args.items()}
            import inspect

            try:
                params = inspect.signature(cls.__init__).parameters
            except (TypeError, ValueError):
                params = None
            if params is not None and not any(
                p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()
            ):
                stale = [k for k in init_args if k not in params]
                if stale:
                    print(f"[common] {cls.__name__}: dropping stale init_args {stale}",
                          flush=True)
                    init_args = {k: v for k, v in init_args.items() if k not in stale}
            return cls(**init_args)
        return {k: _instantiate(v) for k, v in node.items()}
    if isinstance(node, list):
        return [_instantiate(v) for v in node]
    return node


def apply_data_dir_override(cfg: dict) -> dict:
    """Fix the broken preprocessed_dir shipped in configs (in place; returns cfg).

    Configs reference ``.../p200_core_kf_hits_finetune`` which does not exist;
    the real dataset is ``p200_core_kf_matched_finetune``. Logs when applied.
    """
    data = cfg.get("data")
    if isinstance(data, dict):
        pdir = data.get("preprocessed_dir")
        if isinstance(pdir, str) and DATA_DIR_BROKEN_FRAGMENT in pdir:
            data["preprocessed_dir"] = DATA_DIR_OVERRIDE
            print(
                f"[common] data-path override: {pdir!r} -> {DATA_DIR_OVERRIDE!r}",
                flush=True,
            )
    return cfg


def load_config(config_path: Path) -> dict:
    """Merged config with the data-path override applied."""
    return apply_data_dir_override(_load_merged_config(Path(config_path).resolve()))


def apply_dotted_override(cfg: dict, dotted_key: str, raw_value: str) -> None:
    """Apply a ``--set a.b.c=val`` style override into a nested config dict.

    The value is parsed with ``yaml.safe_load`` (so ``32`` is an int,
    ``true`` a bool, ``[1,2]`` a list). Intermediate dicts are created
    when missing.
    """
    keys = dotted_key.split(".")
    node = cfg
    for k in keys[:-1]:
        nxt = node.get(k)
        if not isinstance(nxt, dict):
            nxt = {}
            node[k] = nxt
        node = nxt
    node[keys[-1]] = yaml.safe_load(raw_value)
    print(f"[common] config override: {dotted_key} = {node[keys[-1]]!r}", flush=True)


# ---------------------------------------------------------------------------
# Result sinks
# ---------------------------------------------------------------------------

def _flatten_for_csv(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, default=str)
    return value


def append_jsonl(path: Path | str, row: dict) -> None:
    """Append one dict as a JSON line (creates parents)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(row, default=str) + "\n")
        f.flush()
        os.fsync(f.fileno())


def append_csv(path: Path | str, row: dict, field_order: list[str] | None = None) -> None:
    """Append one dict as a CSV row; write the header if the file is new.

    Nested values (dict/list) are flattened via ``json.dumps``. If the file
    already exists its header wins: missing fields are left empty, extra
    fields are dropped (a warning is printed) so the CSV stays rectangular.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    flat = {k: _flatten_for_csv(v) for k, v in row.items()}

    if path.exists() and path.stat().st_size > 0:
        with open(path, newline="") as f:
            header = next(csv.reader(f), None) or []
        extra = [k for k in flat if k not in header]
        if extra:
            print(f"[common] append_csv: dropping fields not in header: {extra}",
                  file=sys.stderr, flush=True)
    else:
        header = list(field_order) if field_order else list(flat.keys())
        for k in flat:
            if k not in header:
                header.append(k)
        with open(path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=header).writeheader()

    with open(path, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=header, extrasaction="ignore").writerow(flat)


def atomic_write_json(path: Path | str, obj: Any) -> None:
    """Write JSON via a temp file + ``os.rename`` (atomic on POSIX same-fs)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, default=str)
        f.flush()
        os.fsync(f.fileno())
    os.rename(tmp, path)


# ---------------------------------------------------------------------------
# Precision + environment fingerprint
# ---------------------------------------------------------------------------

def pin_precision_flags() -> dict:
    """Pin production numerics and return all precision-relevant flags.

    ``train.py:26`` sets ``torch.set_float32_matmul_precision("high")`` →
    production numerics include TF32 linears (scan internals fp32). Benches
    must run under the same flags and record them alongside every result.
    """
    import torch

    torch.set_float32_matmul_precision("high")

    flags: dict[str, Any] = {
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
    }
    # torch >= 2.9 exposes a per-backend fp32 precision knob for cuDNN conv.
    conv = getattr(torch.backends.cudnn, "conv", None)
    fp32_prec = getattr(conv, "fp32_precision", None) if conv is not None else None
    if fp32_prec is not None:
        flags["cudnn_conv_fp32_precision"] = fp32_prec
    matmul_backend = getattr(torch.backends.cuda, "matmul", None)
    mm_fp32 = getattr(matmul_backend, "fp32_precision", None) if matmul_backend else None
    if mm_fp32 is not None:
        flags["cuda_matmul_fp32_precision"] = mm_fp32
    return flags


def _git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=10,
        ).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _pkg_version(name: str) -> str:
    try:
        import importlib.metadata

        return importlib.metadata.version(name)
    except Exception:
        return "unknown"


def env_fingerprint() -> dict:
    """Git SHA, torch/triton/mamba_ssm versions, GPU name, hostname."""
    import torch

    try:
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
    except Exception:
        gpu_name = "unknown"
    return {
        "git_sha": _git_sha(),
        "torch": torch.__version__,
        "triton": _pkg_version("triton"),
        "mamba_ssm": _pkg_version("mamba_ssm"),
        "gpu_name": gpu_name,
        "hostname": socket.gethostname(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    }


def utc_ts() -> str:
    """ISO-8601 UTC timestamp for result rows."""
    import datetime

    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
