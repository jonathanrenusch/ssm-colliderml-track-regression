#!/usr/bin/env python3
"""Fast RMS-only evaluation: RMSE vs eta, pre-clip and iter-3-sigma, no bootstrap.

The full ``paper_plots`` pipeline spends nearly all of its time in the
bootstrap: 200 resamples per eta bin per parameter per model, plus an 80-sample
unbinned bootstrap for every legend entry.  On a 5 M-track split that dominates
the wall clock while the RMS curves themselves are a single pass.  This routine
drops the bootstrap entirely (no confidence bands) and keeps only the RMSE-vs-eta
figures, in the established design:

  * 2x3 summary grid -- 5 parameters + the eta step-histogram in the 6th cell
  * SSM in C0, CKF in C3; solid = iter-3sigma, dashed = pre-clip
  * per-panel legend carries the unbinned mu; DISPLAY_SCALE / DISPLAY_UNIT units

Reads the ``test_predictions.h5`` written by ``RegressionPredictionWriter`` and
the ACTS columns straight from a flat store (``scripts/preprocess_flat.py``),
which the legacy ``eval_utils.load_acts_augmentation`` cannot read.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from track_regression.eval_utils import PARAMS, iterative_rms_convergence  # noqa: E402
from track_regression.paper_plots import apply_paper_style, save_fig  # noqa: E402
from track_regression.paper_plots.plots._panels import fill_eta_stephist, make_grid  # noqa: E402
from track_regression.paper_plots.stats import DISPLAY_SCALE, DISPLAY_UNIT  # noqa: E402

ETA_EDGES = np.linspace(-3.0, 3.0, 31)


def _save_pdf(fig, out_dir: Path, stem: str) -> None:
    """PDF only — these panels are dense and raster loses detail."""
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{stem}.pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

def load_truth_kf(store_dir: Path) -> np.ndarray | None:
    """Truth-tracking KF fits, if the store carries them.

    Written by scripts/extract_truth_kf.py from `parquet/reco/truth_tracks`, which
    only single_muon_uniform ships. When present this is the PREFERRED reference:
    truth seeding + truth hit assignment, so ~100% efficient and it has no seed
    impact-parameter window (the CKF loses |d0| > 3 mm).
    """
    man_path = store_dir / "manifest.json"
    if not man_path.exists():
        return None            # legacy per-shard store: no truth_tracks anywhere
    man = json.loads(man_path.read_text())
    out = []
    for p in man["parts"]:
        f = store_dir / p["name"] / "truth_kf_reco.npy"
        if not f.exists():
            return None
        out.append(np.load(f, mmap_mode="r"))
    return np.concatenate(out, axis=0)


def load_flat_acts(store_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """(acts_reco (N,5), dm_mask (N,)) in on-disk order, concatenated over parts.

    Handles both stores: a flat store written by ``scripts/preprocess_flat.py``
    (``<dir>/manifest.json`` + parts) and the legacy per-shard layout
    (``<dir>/split.json`` + ``shard_XXXX/selected_tracks/``), so an older
    checkpoint can be evaluated on its own training domain as a control.

    NOTE: this is the CKF (`tracks` table), not the truth-tracking KF.  Only
    single_muon_uniform ships a `truth_tracks` table and the preprocessor does
    not ingest it yet.
    """
    if (store_dir / "manifest.json").exists():
        man = json.loads((store_dir / "manifest.json").read_text())
        reco, dm = [], []
        for p in man["parts"]:
            d = store_dir / p["name"]
            reco.append(np.load(d / "acts_reco.npy", mmap_mode="r"))
            dm.append(np.load(d / "acts_dm.npy", mmap_mode="r"))
        return np.concatenate(reco, axis=0), np.concatenate(dm, axis=0)

    # legacy layout: <root>/split.json + shard_XXXX/selected_tracks/
    root, split = store_dir.parent, store_dir.name
    idxs = json.loads((root / "split.json").read_text())[split]
    reco, dm = [], []
    for i in sorted(idxs):
        sel = root / f"shard_{i:04d}" / "selected_tracks"
        reco.append(np.load(sel / "acts_reco.npy", mmap_mode="r"))
        dm.append(np.load(sel / "acts_dm_mask.npy", mmap_mode="r"))
    return np.concatenate(reco, axis=0), np.concatenate(dm, axis=0)


def _wrap(x):
    return np.mod(x + np.pi, 2 * np.pi) - np.pi


def build_residuals(h5_path: Path, store_dir: Path) -> dict:
    """SSM and CKF residuals over the double-matched subset, plus eta."""
    with h5py.File(h5_path, "r") as f:
        preds = {p: f["preds"][p][:] for p in PARAMS}
        targets = {p: f["targets"][p][:] for p in PARAMS}

    acts, dm_mask = load_flat_acts(store_dir)
    tkf = load_truth_kf(store_dir)
    n = len(targets["d0"])
    if n > len(acts):
        raise ValueError(f"{h5_path.name}: {n:,} predictions vs {len(acts):,} ACTS rows")
    if n < len(acts):
        print(f"  [warn] partial predictions: using first {n:,} of {len(acts):,} tracks")
        acts, dm_mask = acts[:n], dm_mask[:n]
        if tkf is not None:
            tkf = tkf[:n]

    # Reference: the truth-tracking KF when the dataset has it, else the CKF.
    # The CKF is no longer drawn as an extra series when the truth-KF exists
    # (user decision 2026-09-01: every test set now carries the truth-KF); the
    # DM subset definition is unchanged so numbers stay comparable.
    has_ckf = np.asarray(dm_mask, bool) & np.isfinite(acts[:, 0])
    if tkf is not None:
        tkf = np.asarray(tkf)[:n]
        dm = has_ckf & np.isfinite(tkf[:, 0])
        ref, ref_name, second, second_name = tkf, "truth-KF", None, None
    else:
        dm = has_ckf
        ref, ref_name, second, second_name = acts, "CKF", None, None
    out = {"count": int(dm.sum()), "n_total": n,
           "ref_name": ref_name, "second_name": second_name}
    for i, p in enumerate(PARAMS):
        sres = preds[p] - targets[p]
        rres = ref[:, i] - targets[p]
        if p == "phi":
            sres, rres = _wrap(sres), _wrap(rres)
        out[f"ssm_{p}"] = sres[dm]
        out[f"ckf_{p}"] = rres[dm]          # 'ckf_' key kept for compatibility = the REFERENCE
        if second is not None:
            o = second[:, i] - targets[p]
            out[f"second_{p}"] = (_wrap(o) if p == "phi" else o)[dm]
    th = targets["theta"][dm]
    out["eta"] = -np.log(np.tan(np.clip(th, 1e-8, np.pi - 1e-8) / 2.0))
    out["pt"] = np.sin(th) / np.maximum(np.abs(targets["qop"][dm]), 1e-12)
    return out


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

def _raw_rms(x):
    return float(np.sqrt(np.mean(np.asarray(x, np.float64) ** 2)))


def _iter_rms(x):
    return float(iterative_rms_convergence(x)["rms"])


def _profile(eta, res, fn, min_n=30):
    """RMSE per eta bin. One pass, no resampling."""
    centers = 0.5 * (ETA_EDGES[:-1] + ETA_EDGES[1:])
    out = np.full(len(centers), np.nan)
    idx = np.clip(np.digitize(eta, ETA_EDGES) - 1, 0, len(centers) - 1)
    order = np.argsort(idx, kind="stable")
    idx_s, res_s = idx[order], res[order]
    bounds = np.searchsorted(idx_s, np.arange(len(centers) + 1))
    for b in range(len(centers)):
        sel = res_s[bounds[b]:bounds[b + 1]]
        if len(sel) >= min_n:
            out[b] = fn(sel)
    return centers, out


def _fmt(v):
    if v == 0 or not np.isfinite(v):
        return f"{v:.3g}"
    a = abs(v)
    if a >= 100: return f"{v:.0f}"
    if a >= 10:  return f"{v:.1f}"
    if a >= 1:   return f"{v:.2f}"
    if a >= 0.01: return f"{v:.3f}"
    return f"{v:.2e}"


def _fmt_n(n):
    if n >= 1_000_000: return f"{n / 1e6:.2f}M"
    if n >= 1000:      return f"{n / 1e3:.1f}k"
    return str(n)


# ---------------------------------------------------------------------------
# plotting -- same conventions as paper_plots/plots/rms_vs_eta.py
# ---------------------------------------------------------------------------

def _draw(ax, eta, ssm, ckf, p, *, mode, compact, ref_name="CKF", second=None,
          second_name=None):
    scale, unit = DISPLAY_SCALE[p], DISPLAY_UNIT[p]
    show_post = mode in ("both", "postclip")
    show_pre = mode in ("both", "preclip")
    pre_style = "--" if mode == "both" else "-"
    pre_lw = 1.0 if mode == "both" else 1.8
    pre_alpha = 0.7 if mode == "both" else 1.0

    def lbl(name, val, n):
        prefix = "" if compact else f"{name}  "
        body = f"μ = {_fmt(val * scale)}"
        return prefix + body + ("" if compact else f" {unit}  (N={_fmt_n(n)})")

    n_full = len(ssm)
    series = [(ssm, "C0", "SSM"), (ckf, "C3", ref_name)]
    if second is not None:
        series.append((second, "C2", second_name))
    for arr, colour, tag in series:
        if show_post:
            c, y = _profile(eta, arr, _iter_rms)
            cut = iterative_rms_convergence(arr)
            ax.plot(c, y * scale, "-", color=colour, lw=1.8,
                    label=lbl(f"{tag} iter-3σ", cut["rms"], cut["n_kept"]))
        if show_pre:
            c, y = _profile(eta, arr, _raw_rms)
            ax.plot(c, y * scale, pre_style, color=colour, lw=pre_lw, alpha=pre_alpha,
                    label=lbl(f"{tag} pre-clip", _raw_rms(arr), n_full))
    ax.set_xlabel(r"truth $\eta$")
    ax.set_ylabel(f"RMSE({p}) [{unit}]")
    ax.set_xlim(-3, 3)
    ax.set_ylim(bottom=0)
    ax.set_title(p)


def _legend_handles(mode, ref_name="CKF", second_name=None):
    if mode == "both":
        h = [Line2D([0], [0], color="C0", lw=1.8),
             Line2D([0], [0], color="C0", lw=1.0, ls="--", alpha=0.7),
             Line2D([0], [0], color="C3", lw=1.8),
             Line2D([0], [0], color="C3", lw=1.0, ls="--", alpha=0.7)]
        l = ["SSM (iter-3σ)", "SSM (pre-clip)",
             f"{ref_name} (iter-3σ)", f"{ref_name} (pre-clip)"]
        if second_name:
            h += [Line2D([0], [0], color="C2", lw=1.8)]
            l += [f"{second_name} (iter-3σ)"]
        return h, l
    label = "iter-3σ" if mode == "postclip" else "pre-clip"
    h = [Line2D([0], [0], color="C0", lw=1.8), Line2D([0], [0], color="C3", lw=1.8)]
    l = [f"SSM ({label})", f"{ref_name} ({label})"]
    if second_name:
        h += [Line2D([0], [0], color="C2", lw=1.8)]
        l += [f"{second_name} ({label})"]
    return h, l


MODES = [
    ("both", "rms_vs_eta_summary", "RMSE vs η — pre-clip + iter-3σ", False),
    # The pre-clip curve sits 1-2 orders of magnitude above the clipped one on
    # this model, so the linear combined panel flattens every post-clip line
    # onto the axis.  Same figure on a log y is the readable before/after view.
    ("both", "rms_vs_eta_summary_logy", "RMSE vs η — pre-clip + iter-3σ (log y)", True),
    ("preclip", "rms_vs_eta_summary_preclip", "RMSE vs η — pre-clip only (tail-inclusive)", False),
    ("postclip", "rms_vs_eta_summary_postclip", "RMSE vs η — iter-3σ-clipped core only", False),
]


def make_plots(res: dict, out_dir: Path, dataset: str, subtitle: str) -> None:
    eta = res["eta"]
    for mode, stem, title, logy in MODES:
        fig, axes = make_grid()
        for i, p in enumerate(PARAMS):
            _draw(axes[i], eta, res[f"ssm_{p}"], res[f"ckf_{p}"], p, mode=mode, compact=True,
                  ref_name=res.get("ref_name", "CKF"),
                  second=res.get(f"second_{p}"), second_name=res.get("second_name"))
            axes[i].legend(loc="upper left", bbox_to_anchor=(0.02, 0.98), fontsize=7.0,
                           handlelength=1.3, handletextpad=0.4, framealpha=0.85,
                           borderpad=0.25, labelspacing=0.2)
            if logy:
                axes[i].set_yscale("log")
                lo, hi = axes[i].get_ylim()
                axes[i].set_ylim(top=hi * 3.0)
            else:
                lo, hi = axes[i].get_ylim()
                axes[i].set_ylim(bottom=lo, top=hi * 1.20)
        fill_eta_stephist(axes[5], eta)
        h, l = _legend_handles(mode, res.get("ref_name", "CKF"), res.get("second_name"))
        fig.legend(h, l, loc="upper center", ncol=len(l), fontsize=9, frameon=False,
                   bbox_to_anchor=(0.5, 0.985))
        fig.suptitle(f"{dataset} — {title} — reference: {res.get('ref_name','CKF')} — "
                     f"N={res['count']:,}\n{subtitle}", y=1.05)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        _save_pdf(fig, out_dir, f"{dataset}__{stem}")


PT_EDGES = np.array([0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 50.0, 110.0, np.inf])


def pt_bin_table(res: dict, ds: str) -> list[str]:
    """iter-3sigma RMSE per pT bin, SSM vs reference, all five parameters.

    The readout for data-sufficiency / low-pT runs: compare two models bin by bin
    (the uniform muon set has ~1.76 M training tracks per GeV, nothing below 1 GeV).
    """
    pt = res["pt"]
    idx = np.clip(np.digitize(pt, PT_EDGES) - 1, 0, len(PT_EDGES) - 2)
    lines = [f"{ds} — iter-3sigma RMSE per pT bin, SSM / {res.get('ref_name', 'CKF')}",
             f"{'pT [GeV]':>12s} {'N':>9s} " + " ".join(f"{p + ' ' + DISPLAY_UNIT[p]:>22s}" for p in PARAMS)]
    for b in range(len(PT_EDGES) - 1):
        m = idx == b
        n = int(m.sum())
        if n < 100:
            continue
        cells = []
        for p in PARAMS:
            s_ = _iter_rms(res[f"ssm_{p}"][m]) * DISPLAY_SCALE[p]
            r_ = _iter_rms(res[f"ckf_{p}"][m]) * DISPLAY_SCALE[p]
            cells.append(f"{_fmt(s_) + '/' + _fmt(r_):>22s}")
        lo, hi = PT_EDGES[b], PT_EDGES[b + 1]
        lines.append(f"{f'{lo:g}-{hi:g}':>12s} {n:>9,d} " + " ".join(cells))
    return lines


def summary_row(res: dict) -> dict:
    row = {"n_dm": res["count"], "n_total": res["n_total"]}
    for p in PARAMS:
        s = DISPLAY_SCALE[p]
        row[f"{p}_ssm_pre"] = _raw_rms(res[f"ssm_{p}"]) * s
        row[f"{p}_ssm_post"] = _iter_rms(res[f"ssm_{p}"]) * s
        row[f"{p}_ckf_pre"] = _raw_rms(res[f"ckf_{p}"]) * s
        row[f"{p}_ckf_post"] = _iter_rms(res[f"ckf_{p}"]) * s
        # What the clip actually removed. A large ratio with a tiny clipped
        # fraction means a few catastrophic outliers, not a wide core.
        n = len(res[f"ssm_{p}"])
        row[f"{p}_ssm_clipped_pct"] = 100.0 * (
            1.0 - iterative_rms_convergence(res[f"ssm_{p}"])["n_kept"] / n)
        row[f"{p}_ssm_tail_ratio"] = row[f"{p}_ssm_pre"] / max(row[f"{p}_ssm_post"], 1e-30)
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred-dir", required=True, help="directory of <dataset>.h5 files")
    ap.add_argument("--store-root", required=True, help="root holding <dataset>/test flat stores")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--subtitle", default="")
    a = ap.parse_args()

    apply_paper_style()
    pred_dir, root, out = Path(a.pred_dir), Path(a.store_root), Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    names = a.datasets or sorted(p.stem for p in pred_dir.glob("*.h5"))

    def store_of(ds: str) -> Path:
        """`OWN_<name>` refers to the legacy dataset a checkpoint was trained on."""
        if ds.startswith("OWN_"):
            return Path("/scratch/colliderml/arxiv_retraining") / ds[4:] / "test"
        return root / ds / "test"

    table = {}
    pt_lines: list[str] = []
    for ds in names:
        h5 = pred_dir / f"{ds}.h5"
        if not h5.exists():
            print(f"  [skip] {ds}: no {h5.name}")
            continue
        t0 = time.time()
        res = build_residuals(h5, store_of(ds))
        make_plots(res, out, ds, a.subtitle)
        table[ds] = summary_row(res)
        pt_lines += pt_bin_table(res, ds) + [""]
        print(f"  {ds:22s} N_dm={res['count']:>10,d} / {res['n_total']:>10,d}"
              f"   ({time.time() - t0:.1f}s)", flush=True)

    (out / "rms_summary.json").write_text(json.dumps(table, indent=1))
    lines = [f"{'dataset':22s} {'N_dm':>10s} " + " ".join(f"{p:>22s}" for p in PARAMS),
             f"{'':22s} {'':>10s} " + " ".join(
                 f"{'SSM/CKF ' + DISPLAY_UNIT[p]:>22s}" for p in PARAMS)]
    for ds, r in table.items():
        lines.append(f"{ds:22s} {r['n_dm']:>10,d} " + " ".join(
            f"{_fmt(r[p + '_ssm_post']) + '/' + _fmt(r[p + '_ckf_post']):>22s}" for p in PARAMS))
    lines += ["", "SSM tails: pre-clip / iter-3sigma ratio, and % of tracks the clip removed",
              f"{'dataset':22s} " + " ".join(f"{p:>18s}" for p in PARAMS)]
    for ds, r in table.items():
        lines.append(f"{ds:22s} " + " ".join(
            f"{_fmt(r[p + '_ssm_tail_ratio']) + 'x / ' + format(r[p + '_ssm_clipped_pct'], '.2f') + '%':>18s}"
            for p in PARAMS))
    txt = "iter-3sigma RMSE, SSM/CKF\n" + "\n".join(lines) + "\n"
    (out / "rms_summary.txt").write_text(txt)
    (out / "rms_by_pt.txt").write_text("\n".join(pt_lines))
    print("\n" + txt)
    print("\n".join(pt_lines))
    print(f"figures + summary -> {out}")


if __name__ == "__main__":
    main()
