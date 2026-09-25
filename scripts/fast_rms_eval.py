#!/usr/bin/env python3
"""Per-dataset resolution summary of the network predictions vs the reference fit.

For every ``<dataset>.h5`` in ``--pred-dir`` (written by ``scripts/predict.py``)
this reads the matching evaluation store ``<store-root>/<dataset>/test`` and compares the network (labelled by --label) with
the reference fit on the double-matched (DM) tracks: tracks matched by the
ACTS combinatorial KF (CKF) and fitted by the truth-seeded KF (truth-KF).
The truth-KF is the reference whenever the store carries its side-car
``truth_kf_reco.npy``; otherwise the CKF is.

Outputs in ``--out-dir``:
  * ``rms_summary.{txt,json}`` -- unbinned RMSE per parameter, pre-clip and
    iterative-3-sigma-clipped, plus the fraction of tracks the clip removed;
  * ``rms_by_pt.txt`` -- the clipped RMSE per pT bin;
  * ``<dataset>__rms_vs_eta_summary{,_logy,_preclip,_postclip}.pdf`` -- RMSE vs
    eta, 2x3 grid (five parameters + the eta distribution), network in C0, the
    reference in C3, solid = iter-3-sigma, dashed = pre-clip.

Only tracks with |truth eta| <= ``--eta-max`` enter (default 2, the paper's
fiducial region).  No bootstrap: a single pass, fast on multi-million-track sets.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from track_regression.eval_utils import (  # noqa: E402
    DISPLAY_SCALE, DISPLAY_UNIT, PARAMS, apply_paper_style, fill_eta_stephist,
    iterative_rms_convergence, make_grid,
)


def _save_pdf(fig, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{stem}.pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

def _parts(store_dir: Path) -> list[Path]:
    man = json.loads((store_dir / "manifest.json").read_text())
    return [store_dir / p["name"] for p in man["parts"]]


def load_truth_kf(store_dir: Path) -> np.ndarray | None:
    """Truth-seeded KF fits (N, 5), or None if the store has no side-cars."""
    files = [d / "truth_kf_reco.npy" for d in _parts(store_dir)]
    if not all(f.exists() for f in files):
        return None
    return np.concatenate([np.load(f, mmap_mode="r") for f in files], axis=0)


def load_flat_acts(store_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """(CKF fits (N, 5), CKF double-match mask (N,)) in on-disk track order."""
    parts = _parts(store_dir)
    reco = np.concatenate([np.load(d / "acts_reco.npy", mmap_mode="r") for d in parts], axis=0)
    dm = np.concatenate([np.load(d / "acts_dm.npy", mmap_mode="r") for d in parts], axis=0)
    return reco, dm


def _wrap(x):
    return np.mod(x + np.pi, 2 * np.pi) - np.pi


def build_residuals(h5_path: Path, store_dir: Path, eta_max: float) -> dict:
    """Network and reference residuals over the DM subset inside |eta| <= eta_max."""
    with h5py.File(h5_path, "r") as f:
        preds = {p: f["preds"][p][:] for p in PARAMS}
        targets = {p: f["targets"][p][:] for p in PARAMS}

    acts, dm_mask = load_flat_acts(store_dir)
    tkf = load_truth_kf(store_dir)
    n = len(targets["d0"])
    if n != len(acts):
        raise ValueError(f"{h5_path.name}: {n:,} predictions vs {len(acts):,} tracks in {store_dir}")

    has_ckf = np.asarray(dm_mask, bool) & np.isfinite(acts[:, 0])
    if tkf is not None:
        tkf = np.asarray(tkf)
        dm = has_ckf & np.isfinite(tkf[:, 0])
        ref, ref_name = tkf, "truth-KF"
    else:
        dm = has_ckf
        ref, ref_name = acts, "CKF"
    out = {"count": int(dm.sum()), "n_total": n, "ref_name": ref_name}
    for i, p in enumerate(PARAMS):
        sres = preds[p] - targets[p]
        rres = ref[:, i] - targets[p]
        if p == "phi":
            sres, rres = _wrap(sres), _wrap(rres)
        out[f"ssm_{p}"] = sres[dm]
        out[f"ref_{p}"] = rres[dm]
    th = targets["theta"][dm]
    out["eta"] = -np.log(np.tan(np.clip(th, 1e-8, np.pi - 1e-8) / 2.0))
    out["pt"] = np.sin(th) / np.maximum(np.abs(targets["qop"][dm]), 1e-12)
    if eta_max < 3.0:                          # the samples extend to |eta| = 3
        keep = np.abs(out["eta"]) <= eta_max
        for k, v in out.items():
            if isinstance(v, np.ndarray):
                out[k] = v[keep]
        out["count"] = int(keep.sum())
    return out


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

def _raw_rms(x):
    return float(np.sqrt(np.mean(np.asarray(x, np.float64) ** 2)))


def _iter_rms(x):
    return float(iterative_rms_convergence(x)["rms"])


def _profile(eta, res, fn, eta_edges, min_n=30):
    """RMSE per eta bin (bins with fewer than min_n tracks are left empty)."""
    centers = 0.5 * (eta_edges[:-1] + eta_edges[1:])
    out = np.full(len(centers), np.nan)
    idx = np.clip(np.digitize(eta, eta_edges) - 1, 0, len(centers) - 1)
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


# ---------------------------------------------------------------------------
# plotting
# ---------------------------------------------------------------------------

def _draw(ax, eta, ssm, ref, p, eta_edges, *, mode):
    scale, unit = DISPLAY_SCALE[p], DISPLAY_UNIT[p]
    show_post = mode in ("both", "postclip")
    show_pre = mode in ("both", "preclip")
    pre_style = "--" if mode == "both" else "-"
    pre_lw = 1.0 if mode == "both" else 1.8
    pre_alpha = 0.7 if mode == "both" else 1.0

    for arr, colour in ((ssm, "C0"), (ref, "C3")):
        if show_post:
            c, y = _profile(eta, arr, _iter_rms, eta_edges)
            ax.plot(c, y * scale, "-", color=colour, lw=1.8,
                    label=f"μ = {_fmt(iterative_rms_convergence(arr)['rms'] * scale)}")
        if show_pre:
            c, y = _profile(eta, arr, _raw_rms, eta_edges)
            ax.plot(c, y * scale, pre_style, color=colour, lw=pre_lw, alpha=pre_alpha,
                    label=f"μ = {_fmt(_raw_rms(arr) * scale)}")
    ax.set_xlabel(r"truth $\eta$")
    ax.set_ylabel(f"RMSE({p}) [{unit}]")
    ax.set_xlim(-3, 3)
    ax.set_ylim(bottom=0)
    ax.set_title(p)


NET = "minGRU"   # network label in tables and legends (--label)


def _legend_handles(mode, ref_name):
    if mode == "both":
        h = [Line2D([0], [0], color="C0", lw=1.8),
             Line2D([0], [0], color="C0", lw=1.0, ls="--", alpha=0.7),
             Line2D([0], [0], color="C3", lw=1.8),
             Line2D([0], [0], color="C3", lw=1.0, ls="--", alpha=0.7)]
        l = [f"{NET} (iter-3σ)", f"{NET} (pre-clip)",
             f"{ref_name} (iter-3σ)", f"{ref_name} (pre-clip)"]
        return h, l
    label = "iter-3σ" if mode == "postclip" else "pre-clip"
    h = [Line2D([0], [0], color="C0", lw=1.8), Line2D([0], [0], color="C3", lw=1.8)]
    return h, [f"{NET} ({label})", f"{ref_name} ({label})"]


MODES = [
    ("both", "rms_vs_eta_summary", "RMSE vs η — pre-clip + iter-3σ", False),
    # The pre-clip curve can sit far above the clipped one, which flattens the
    # clipped curves on a linear axis; the same figure on a log y stays readable.
    ("both", "rms_vs_eta_summary_logy", "RMSE vs η — pre-clip + iter-3σ (log y)", True),
    ("preclip", "rms_vs_eta_summary_preclip", "RMSE vs η — pre-clip only (tail-inclusive)", False),
    ("postclip", "rms_vs_eta_summary_postclip", "RMSE vs η — iter-3σ-clipped core only", False),
]


def make_plots(res: dict, out_dir: Path, dataset: str, subtitle: str, eta_max: float) -> None:
    eta = res["eta"]
    # 0.2-wide eta bins over the fiducial range
    eta_edges = np.linspace(-eta_max, eta_max, int(round(10 * eta_max)) + 1)
    ref_name = res["ref_name"]
    for mode, stem, title, logy in MODES:
        fig, axes = make_grid()
        for i, p in enumerate(PARAMS):
            _draw(axes[i], eta, res[f"ssm_{p}"], res[f"ref_{p}"], p, eta_edges,
                  mode=mode)
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
        h, l = _legend_handles(mode, ref_name)
        fig.legend(h, l, loc="upper center", ncol=len(l), fontsize=9, frameon=False,
                   bbox_to_anchor=(0.5, 0.985))
        fig.suptitle(f"{dataset} — {title} — reference: {ref_name} — "
                     f"N={res['count']:,}\n{subtitle}", y=1.05)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        _save_pdf(fig, out_dir, f"{dataset}__{stem}")


PT_EDGES = np.array([0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 50.0, 110.0, np.inf])


def pt_bin_table(res: dict, ds: str) -> list[str]:
    """iter-3-sigma RMSE per pT bin, network / reference, all five parameters."""
    pt = res["pt"]
    idx = np.clip(np.digitize(pt, PT_EDGES) - 1, 0, len(PT_EDGES) - 2)
    lines = [f"{ds} — iter-3sigma RMSE per pT bin, {NET} / {res['ref_name']}",
             f"{'pT [GeV]':>12s} {'N':>9s} " + " ".join(f"{p + ' ' + DISPLAY_UNIT[p]:>22s}" for p in PARAMS)]
    for b in range(len(PT_EDGES) - 1):
        m = idx == b
        n = int(m.sum())
        if n < 100:
            continue
        cells = []
        for p in PARAMS:
            s_ = _iter_rms(res[f"ssm_{p}"][m]) * DISPLAY_SCALE[p]
            r_ = _iter_rms(res[f"ref_{p}"][m]) * DISPLAY_SCALE[p]
            cells.append(f"{_fmt(s_) + '/' + _fmt(r_):>22s}")
        lo, hi = PT_EDGES[b], PT_EDGES[b + 1]
        lines.append(f"{f'{lo:g}-{hi:g}':>12s} {n:>9,d} " + " ".join(cells))
    return lines


def summary_row(res: dict) -> dict:
    row = {"n_dm": res["count"], "n_total": res["n_total"], "ref_name": res["ref_name"]}
    for p in PARAMS:
        s = DISPLAY_SCALE[p]
        row[f"{p}_ssm_pre"] = _raw_rms(res[f"ssm_{p}"]) * s
        row[f"{p}_ssm_post"] = _iter_rms(res[f"ssm_{p}"]) * s
        row[f"{p}_ref_pre"] = _raw_rms(res[f"ref_{p}"]) * s
        row[f"{p}_ref_post"] = _iter_rms(res[f"ref_{p}"]) * s
        # What the clip removed: a large pre/post ratio with a tiny clipped
        # fraction means a few far outliers rather than a wide core.
        n = len(res[f"ssm_{p}"])
        row[f"{p}_ssm_clipped_pct"] = 100.0 * (
            1.0 - iterative_rms_convergence(res[f"ssm_{p}"])["n_kept"] / n)
        row[f"{p}_ssm_tail_ratio"] = row[f"{p}_ssm_pre"] / max(row[f"{p}_ssm_post"], 1e-30)
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred-dir", required=True, help="directory of <dataset>.h5 files")
    ap.add_argument("--store-root", required=True, help="root holding <dataset>/test evaluation stores")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--datasets", nargs="*", default=None, help="default: every .h5 in --pred-dir")
    ap.add_argument("--subtitle", default="", help="second title line of the figures")
    ap.add_argument("--eta-max", type=float, default=2.0, help="fiducial |truth eta| cut (default 2)")
    ap.add_argument("--label", default="minGRU", help="name of the network in tables and legends")
    a = ap.parse_args()
    global NET
    NET = a.label

    apply_paper_style()
    pred_dir, root, out = Path(a.pred_dir), Path(a.store_root), Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    names = a.datasets or sorted(p.stem for p in pred_dir.glob("*.h5"))

    table = {}
    pt_lines: list[str] = []
    for ds in names:
        h5 = pred_dir / f"{ds}.h5"
        if not h5.exists():
            print(f"  [skip] {ds}: no {h5.name}")
            continue
        t0 = time.time()
        res = build_residuals(h5, root / ds / "test", a.eta_max)
        make_plots(res, out, ds, a.subtitle, a.eta_max)
        table[ds] = summary_row(res)
        pt_lines += pt_bin_table(res, ds) + [""]
        print(f"  {ds:22s} N_dm={res['count']:>10,d} / {res['n_total']:>10,d}"
              f"   ({time.time() - t0:.1f}s)", flush=True)

    (out / "rms_summary.json").write_text(json.dumps(table, indent=1))
    lines = [f"{'dataset':22s} {'N_dm':>10s} {'reference':>9s} " + " ".join(f"{p:>22s}" for p in PARAMS),
             f"{'':22s} {'':>10s} {'':>9s} " + " ".join(
                 f"{NET + '/ref ' + DISPLAY_UNIT[p]:>22s}" for p in PARAMS)]
    for ds, r in table.items():
        lines.append(f"{ds:22s} {r['n_dm']:>10,d} {r['ref_name']:>9s} " + " ".join(
            f"{_fmt(r[p + '_ssm_post']) + '/' + _fmt(r[p + '_ref_post']):>22s}" for p in PARAMS))
    lines += ["", f"{NET} tails: pre-clip / iter-3sigma ratio, and % of tracks the clip removed",
              f"{'dataset':22s} " + " ".join(f"{p:>18s}" for p in PARAMS)]
    for ds, r in table.items():
        lines.append(f"{ds:22s} " + " ".join(
            f"{_fmt(r[p + '_ssm_tail_ratio']) + 'x / ' + format(r[p + '_ssm_clipped_pct'], '.2f') + '%':>18s}"
            for p in PARAMS))
    txt = f"iter-3sigma RMSE, {NET} / reference, |eta| <= {a.eta_max:g}\n" + "\n".join(lines) + "\n"
    (out / "rms_summary.txt").write_text(txt)
    (out / "rms_by_pt.txt").write_text("\n".join(pt_lines))
    print("\n" + txt)
    print("\n".join(pt_lines))
    print(f"figures + summary -> {out}")


if __name__ == "__main__":
    main()
