#!/usr/bin/env python3
"""Write matched_residuals.npz per dataset: truth, network ("SSM") and
truth-seeded KF ("kf") perigee parameters on the double-matched tracks.

The subset is the one fast_rms_eval.py uses (CKF-matched and truth-KF fitted),
so the figures and tables built from these files quote the same tracks.
Each array is (N, 5) with columns d0, z0, phi, theta, q/p.

Usage:
  build_residuals.py <pred_dir> <store_root> <out_root> [datasets...]
    pred_dir   : directory of <dataset>.h5 network predictions
    store_root : evaluation root holding <dataset>/test stores with truth-KF side-cars
    out_root   : writes <out_root>/<dataset>/matched_residuals.npz
"""
from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from track_regression.eval_utils import PARAMS  # noqa: E402
from fast_rms_eval import load_flat_acts, load_truth_kf  # noqa: E402


def build(h5_path: Path, store_dir: Path) -> dict[str, np.ndarray]:
    with h5py.File(h5_path, "r") as f:
        preds = {p: f["preds"][p][:] for p in PARAMS}
        targets = {p: f["targets"][p][:] for p in PARAMS}
    acts, dm_mask = load_flat_acts(store_dir)
    tkf = load_truth_kf(store_dir)
    if tkf is None:
        raise SystemExit(f"{store_dir} has no truth-KF side-cars (truth_kf_reco.npy)")
    n = len(targets["d0"])
    if n != len(acts):
        raise SystemExit(f"{h5_path.name}: {n:,} predictions vs {len(acts):,} tracks in {store_dir}")
    tkf = np.asarray(tkf)
    dm = np.asarray(dm_mask, bool) & np.isfinite(acts[:, 0]) & np.isfinite(tkf[:, 0])
    truth = np.stack([targets[p] for p in PARAMS], axis=1)[dm]
    ssm = np.stack([preds[p] for p in PARAMS], axis=1)[dm]
    return {"truth": truth, "ssm": ssm, "kf": tkf[dm]}


def main() -> None:
    if len(sys.argv) < 4:
        raise SystemExit(__doc__)
    pred_dir, store_root, out_root = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
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
        print(f"[residuals] {ds}: {len(d['truth']):,} matched tracks -> {out}", flush=True)


if __name__ == "__main__":
    main()
