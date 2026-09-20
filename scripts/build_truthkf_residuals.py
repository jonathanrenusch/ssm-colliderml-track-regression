#!/usr/bin/env python3
"""Build matched_residuals.npz (absolute truth / SSM / production-truth-KF) for
the approved curve+residual plot designs (acts_rms_curves.py,
acts_legacy_style_plots.py), using the PRODUCTION truth-tracking KF as the
reference instead of the miscalibrated in-pipeline ACTS KF refit
(docs/BUGREPORT_acts_pipeline_kf.md, CLAUDE.md §4.29).

The three arrays are on the same double-matched subset fast_rms_eval uses, so
the numbers match the paper's results table and the fast_rms bundles.

Usage:
  build_truthkf_residuals.py <pred_dir> <store_root> <out_root> [datasets...]
    pred_dir   : dir of <dataset>.h5 SSM predictions (strict fp32)
    store_root : v2 eval farm root (holds <dataset>/test flat stores)
    out_root   : writes <out_root>/<dataset>/matched_residuals.npz
"""
from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "src" / "track_regression" / "scripts"))

from track_regression.eval_utils import PARAMS  # noqa: E402
from fast_rms_eval import load_flat_acts, load_truth_kf  # noqa: E402


def build(h5_path: Path, store_dir: Path) -> dict[str, np.ndarray]:
    with h5py.File(h5_path, "r") as f:
        preds = {p: f["preds"][p][:] for p in PARAMS}
        targets = {p: f["targets"][p][:] for p in PARAMS}
    acts, dm_mask = load_flat_acts(store_dir)
    tkf = load_truth_kf(store_dir)
    if tkf is None:
        raise SystemExit(f"{store_dir} has no production truth-KF side-cars")
    n = len(targets["d0"])
    acts, dm_mask, tkf = acts[:n], dm_mask[:n], np.asarray(tkf)[:n]
    # same double-matched subset as fast_rms_eval (CKF-matched AND truth-KF-finite)
    dm = np.asarray(dm_mask, bool) & np.isfinite(acts[:, 0]) & np.isfinite(tkf[:, 0])
    truth = np.stack([targets[p] for p in PARAMS], axis=1)[dm]
    ssm = np.stack([preds[p] for p in PARAMS], axis=1)[dm]
    kf = tkf[dm]  # production truth-KF, columns aligned with PARAMS
    return {"truth": truth, "ssm": ssm, "kf": kf}


def main() -> None:
    pred_dir = Path(sys.argv[1])
    store_root = Path(sys.argv[2])
    out_root = Path(sys.argv[3])
    names = sys.argv[4:] or sorted(p.stem for p in pred_dir.glob("*.h5"))
    for ds in names:
        h5 = pred_dir / f"{ds}.h5"
        if not h5.exists():
            print(f"  [skip] {ds}: no {h5.name}")
            continue
        d = build(h5, store_root / ds / "test")
        out = out_root / ds
        out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out / "matched_residuals.npz", **d)
        print(f"[truthkf] {ds}: {len(d['truth']):,} matched tracks -> {out}", flush=True)


if __name__ == "__main__":
    main()
