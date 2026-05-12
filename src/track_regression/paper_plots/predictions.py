"""Load predictions, build double-matched residuals (SSM + CKF aligned)."""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from hepattn.experiments.colliderml_regr.eval_utils import (
    PARAMS,
    load_acts_augmentation,
)


def _wrap(x: np.ndarray) -> np.ndarray:
    return np.mod(x + np.pi, 2 * np.pi) - np.pi


def load_predictions_with_d0_override(
    main_h5: Path,
    d0_h5: Path | None = None,
) -> dict[str, dict[str, np.ndarray]]:
    """Load preds + targets from main h5; optionally override preds[d0] from a d0-only run.

    Asserts identical targets between the two files for d0 (catches split mismatch).
    """
    with h5py.File(main_h5, "r") as f:
        preds = {p: f["preds"][p][:] for p in PARAMS}
        targets = {p: f["targets"][p][:] for p in PARAMS}

    if d0_h5 is not None:
        with h5py.File(d0_h5, "r") as f:
            d0_preds = f["preds"]["d0"][:]
            d0_targets = f["targets"]["d0"][:]
        if d0_preds.shape != preds["d0"].shape:
            raise ValueError(
                f"d0 override shape mismatch: main {preds['d0'].shape} vs d0 {d0_preds.shape}"
            )
        if not np.array_equal(d0_targets, targets["d0"]):
            raise ValueError(
                "d0 override targets do not match main run targets — different split?"
            )
        preds["d0"] = d0_preds

    return {"preds": preds, "targets": targets}


def build_dm_residuals(
    data: dict,
    data_dir: Path,
) -> dict:
    """Compute SSM + CKF residuals on the double-matched regime.

    Returns a flat dict containing (per param) ``ssm_<p>`` and ``ckf_<p>`` arrays
    of equal length over the DM subset, plus ``eta``, ``pt``, ``count``.
    """
    aug = load_acts_augmentation(data_dir, split="test")
    if aug is None:
        raise FileNotFoundError(f"ACTS augmentation not available under {data_dir}")
    acts_reco, dm_mask, _ = aug
    n_pred = len(data["targets"]["d0"])
    n_acts = len(acts_reco)
    if n_pred > n_acts:
        raise ValueError(
            f"length mismatch: acts {n_acts} vs preds {n_pred} "
            "(predictions cover more tracks than ACTS augmentation has)"
        )
    if n_pred < n_acts:
        # Partial inference (e.g. --limit_test_batches): truncate ACTS to the
        # contiguous prefix that matches the prediction set so the pipeline
        # still produces self-consistent residuals on the partial sample.
        print(f"[predictions] partial h5: truncating ACTS to first {n_pred:,} of {n_acts:,} tracks")
        acts_reco = acts_reco[:n_pred]
        dm_mask = dm_mask[:n_pred]

    dm = dm_mask & ~np.isnan(acts_reco[:, 0])

    out = {"count": int(dm.sum())}
    for i, p in enumerate(PARAMS):
        s = data["preds"][p] - data["targets"][p]
        c = acts_reco[:, i] - data["targets"][p]
        if p == "phi":
            s = _wrap(s)
            c = _wrap(c)
        out[f"ssm_{p}"] = s[dm]
        out[f"ckf_{p}"] = c[dm]

    theta_t = data["targets"]["theta"][dm]
    qop_t = data["targets"]["qop"][dm]
    out["eta"] = -np.log(np.tan(np.clip(theta_t, 1e-8, np.pi - 1e-8) / 2.0))
    out["pt"] = np.sin(np.clip(theta_t, 1e-8, np.pi - 1e-8)) / np.clip(np.abs(qop_t), 1e-8, None)
    out["theta_truth"] = theta_t
    out["qop_truth"] = qop_t

    # also keep target arrays restricted to DM, useful for heatmaps
    for p in PARAMS:
        out[f"truth_{p}"] = data["targets"][p][dm]
        out[f"pred_ssm_{p}"] = data["preds"][p][dm]
        out[f"pred_ckf_{p}"] = acts_reco[dm, i_of(p)]

    return out


def i_of(p: str) -> int:
    return PARAMS.index(p)
