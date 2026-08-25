#!/usr/bin/env python3
"""Reference-fit baselines: RMSE and IQR, pre- and post-iterative-clip.

Characterises the *reconstruction* baselines (truth-tracking KF and CKF) on
every ICLR eval store, independent of any network.  For each dataset x
estimator x parameter it reports

  RMSE_pre    sqrt(mean(r**2))            over the full selected subset
  RMSE_post   iterative 3-sigma clipped   unbinned over the whole dataset
  IQR_pre     p75 - p25                   over the full subset
  IQR_post    p75 - p25 of the survivors of the same clip
  sigma_IQR   IQR / 1.349                 Gaussian-equivalent width

RMSE is bias-sensitive (sqrt(sigma**2 + mean**2)); the IQR is a robust width
that ignores tails entirely.  Quoting both separates "wide core" from
"few catastrophic outliers": if RMSE_pre >> RMSE_post while IQR_pre ~= IQR_post,
the damage is all tail.

Two scopes are reported per estimator:

  dm    double-matched (purity>75% AND efficiency>75%) AND a finite fit from
        *every* estimator in the store -- the apples-to-apples subset, and the
        one every SSM-vs-reference number we quote uses.
  own   every track this estimator produced a finite fit for -- its natural
        coverage.  The truth KF is ~100% efficient and has no seed |d0| window,
        so its `own` scope is much larger than the CKF's and this is where its
        real advantage shows.

Only single_muon_uniform ships a `reco/truth_tracks` table, so the truth KF
rows exist for the uniform stores alone; the fixed-pT sets and ttbar are
CKF-only and cannot be given a truth-KF baseline without a re-reco.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from track_regression.eval_utils import PARAMS, iterative_rms_convergence  # noqa: E402
from track_regression.paper_plots.stats import DISPLAY_SCALE, DISPLAY_UNIT  # noqa: E402
from track_regression.scripts.fast_rms_eval import (  # noqa: E402
    load_flat_acts, load_truth_kf, _wrap,
)

IQR_TO_SIGMA = 1.3489795003921634   # 2 * sqrt(2) * erfinv(0.5), i.e. IQR of N(0,1)


def load_targets(store_dir: Path) -> np.ndarray:
    man = json.loads((store_dir / "manifest.json").read_text())
    return np.concatenate(
        [np.load(store_dir / p["name"] / "targets.npy", mmap_mode="r") for p in man["parts"]],
        axis=0,
    )


def clip_survivors(x: np.ndarray, n_sigma: float = 3.0, max_iter: int = 5) -> np.ndarray:
    """Replicates iterative_rms_convergence's loop but returns the kept values.

    Same window sequence (mean +- n_sigma*std, std recomputed on the survivors,
    same early break on a stationary count), so the RMSE of the return value is
    bit-identical to the helper's ``rms``.  Asserted below.
    """
    data = np.asarray(x, np.float64)
    prev_n = -1
    for _ in range(max_iter):
        mean, sigma = float(np.mean(data)), float(np.std(data))
        mask = (data >= mean - n_sigma * sigma) & (data <= mean + n_sigma * sigma)
        n_kept = int(mask.sum())
        if n_kept == prev_n:
            break
        prev_n = n_kept
        data = data[mask]
    return data


def iqr(x: np.ndarray) -> float:
    lo, hi = np.percentile(np.asarray(x, np.float64), [25.0, 75.0])
    return float(hi - lo)


def metrics(res: np.ndarray, scale: float) -> dict:
    kept = clip_survivors(res)
    ref = iterative_rms_convergence(res)
    rms_post = float(np.sqrt(np.mean(kept ** 2)))
    # the two implementations must agree to float noise or the clip differs
    assert abs(rms_post - ref["rms"]) <= 1e-9 * max(abs(ref["rms"]), 1e-30), \
        f"clip mismatch {rms_post} vs {ref['rms']}"
    iqr_pre, iqr_post = iqr(res), iqr(kept)
    return {
        "n": int(res.size),
        "n_kept": int(kept.size),
        "kept_pct": 100.0 * kept.size / max(res.size, 1),
        "rmse_pre": float(np.sqrt(np.mean(np.asarray(res, np.float64) ** 2))) * scale,
        "rmse_post": rms_post * scale,
        "iqr_pre": iqr_pre * scale,
        "iqr_post": iqr_post * scale,
        "sigma_iqr_pre": iqr_pre / IQR_TO_SIGMA * scale,
        "sigma_iqr_post": iqr_post / IQR_TO_SIGMA * scale,
        "mean_post": float(np.mean(kept)) * scale,
    }


def run_store(name: str, store_dir: Path) -> dict:
    acts, dm = load_flat_acts(store_dir)
    tkf = load_truth_kf(store_dir)
    tg = load_targets(store_dir)
    n = len(tg)
    acts, dm = np.asarray(acts[:n]), np.asarray(dm[:n], bool)
    fin_ckf = np.isfinite(acts).all(axis=1)
    ests = {"CKF": (acts, dm & fin_ckf)}
    common = dm & fin_ckf
    if tkf is not None:
        tkf = np.asarray(tkf[:n])
        fin_tkf = np.isfinite(tkf).all(axis=1)
        ests["truth-KF"] = (tkf, fin_tkf)
        common = common & fin_tkf

    out = {"store": str(store_dir), "n_tracks": int(n),
           "n_dm": int((dm & fin_ckf).sum()), "n_common": int(common.sum()),
           "has_truth_kf": tkf is not None, "estimators": {}}
    for est, (fit, own) in ests.items():
        out["estimators"][est] = {"n_own": int(own.sum()), "params": {}}
        for i, p in enumerate(PARAMS):
            r = fit[:, i] - tg[:, i]
            if p == "phi":
                r = _wrap(r)
            s = DISPLAY_SCALE[p]
            out["estimators"][est]["params"][p] = {
                "unit": DISPLAY_UNIT[p],
                "dm": metrics(r[common], s),
                "own": metrics(r[own], s),
            }
        print(f"  {name:34s} {est:9s} own={own.sum():>10,d} common={common.sum():>10,d}", flush=True)
    return out


def fmt(v: float) -> str:
    a = abs(v)
    if a == 0 or np.isnan(v):
        return f"{v:.4g}"
    return f"{v:.4g}" if a >= 1e-3 else f"{v:.4e}"


def write_report(results: dict, path: Path) -> None:
    L = []
    L.append("Reference-fit baselines on the ICLR eval stores")
    L.append("=" * 118)
    L.append(__doc__.strip())
    L.append("")
    L.append("Clip: iterative 3-sigma, max 5 passes, window = mean +- 3*std recomputed on")
    L.append("survivors; applied UNBINNED over the whole dataset (not per-eta-bin then averaged).")
    L.append(f"sigma_IQR = IQR / {IQR_TO_SIGMA:.6f}  (the IQR of a unit Gaussian).")
    L.append("")
    for scope, blurb in [("dm", "double-matched, finite fit from every estimator in the store"),
                         ("own", "every track this estimator fitted (its own coverage)")]:
        L.append("")
        L.append("#" * 118)
        L.append(f"# SCOPE '{scope}': {blurb}")
        L.append("#" * 118)
        for name, r in results.items():
            L.append("")
            L.append(f"{name}   [{r['n_tracks']:,} tracks, {r['n_dm']:,} double-matched"
                     f", {r['n_common']:,} common"
                     f"{'' if r['has_truth_kf'] else ', NO truth_tracks -> CKF only'}]")
            hdr = (f"  {'est':9s} {'par':6s} {'unit':6s} {'N':>11s} {'kept%':>7s} "
                   f"{'RMSE_pre':>11s} {'RMSE_post':>11s} {'IQR_pre':>11s} {'IQR_post':>11s} "
                   f"{'sIQR_post':>11s} {'mean_post':>11s}")
            L.append(hdr)
            L.append("  " + "-" * (len(hdr) - 2))
            for est, e in r["estimators"].items():
                for p in PARAMS:
                    m = e["params"][p][scope]
                    L.append(f"  {est:9s} {p:6s} {e['params'][p]['unit']:6s} "
                             f"{m['n']:>11,d} {m['kept_pct']:>7.3f} "
                             f"{fmt(m['rmse_pre']):>11s} {fmt(m['rmse_post']):>11s} "
                             f"{fmt(m['iqr_pre']):>11s} {fmt(m['iqr_post']):>11s} "
                             f"{fmt(m['sigma_iqr_post']):>11s} {fmt(m['mean_post']):>11s}")
    path.write_text("\n".join(L) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stores", nargs="+", required=True,
                    help="NAME=/path/to/store/test pairs")
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()

    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for spec in a.stores:
        name, _, path = spec.partition("=")
        d = Path(path)
        if not (d / "manifest.json").exists():
            print(f"  [skip] {name}: no manifest.json at {d}")
            continue
        results[name] = run_store(name, d)

    (out_dir / "kf_baselines.json").write_text(json.dumps(results, indent=2))
    write_report(results, out_dir / "kf_baselines.txt")
    print(f"\nwrote {out_dir/'kf_baselines.txt'}\n      {out_dir/'kf_baselines.json'}")


if __name__ == "__main__":
    main()
