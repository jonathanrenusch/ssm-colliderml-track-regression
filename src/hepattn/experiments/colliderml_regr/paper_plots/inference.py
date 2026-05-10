"""Inference helper: ensure the test_predictions.h5 exists for a run.

If a `train.py test` is already running for this run dir (user-launched),
we just wait for the h5 to appear.  Otherwise we spawn it.
"""
from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

from . import COMET_OFFLINE_ROOT, DATA_DIR
from .bundle import _resolve_best_ckpt_and_h5


def _is_inference_running(run_id: str) -> bool:
    try:
        out = subprocess.check_output(
            ["pgrep", "-af", f"train.py.*test.*{run_id}"], text=True
        )
        return bool(out.strip())
    except subprocess.CalledProcessError:
        return False


def ensure_predictions(
    run_id: str,
    *,
    gpu: int = 0,
    data_dir: Path = DATA_DIR,
    wait_timeout_s: int = 60 * 60 * 6,
    poll_s: int = 30,
    skip_inference: bool = False,
) -> Path:
    run_dir = COMET_OFFLINE_ROOT / run_id
    _, h5 = _resolve_best_ckpt_and_h5(run_dir)
    if h5.exists():
        return h5

    if skip_inference:
        raise FileNotFoundError(
            f"--skip-inference set but predictions h5 does not exist: {h5}"
        )

    if _is_inference_running(run_id):
        print(f"[inference] already running for {run_id}, polling for {h5.name}…")
        deadline = time.time() + wait_timeout_s
        while time.time() < deadline:
            if h5.exists():
                return h5
            time.sleep(poll_s)
        raise TimeoutError(f"timed out waiting for {h5}")

    # spawn
    best_ckpt, _ = _resolve_best_ckpt_and_h5(run_dir)
    cfg = run_dir / "config.yaml"
    cmd = [
        "pixi", "run", "python", "train.py", "test",
        "--config", str(cfg),
        "--ckpt_path", str(best_ckpt),
        "--trainer.devices", "1",
        "--data.batch_size", "10000",
        "--data.num_workers", "0",
        "--data.preprocessed_dir", str(data_dir),
    ]
    log = run_dir / "inference.log"
    # Run from the colliderml_regr experiment dir (where train.py lives).
    cwd = str(Path(__file__).resolve().parents[1])
    print(f"[inference] spawning: {' '.join(cmd)}")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    with open(log, "ab") as flog:
        rc = subprocess.call(cmd, cwd=cwd, env=env, stdout=flog, stderr=subprocess.STDOUT)
    if rc != 0:
        raise RuntimeError(f"train.py test failed (rc={rc}); see {log}")

    _, h5 = _resolve_best_ckpt_and_h5(run_dir)
    if not h5.exists():
        raise FileNotFoundError(f"inference completed but h5 missing: {h5}")
    return h5
