#!/usr/bin/env python3
"""RMSE-vs-eta figures for the REFERENCE fits alone (truth-KF and CKF) on an
ICLR eval farm -- the network-free companion of ``fast_rms_eval.py``, in the
same 2x3 design (5 parameters + eta step-histogram; solid = iter-3sigma,
dashed = pre-clip).  Reads targets / acts_reco / acts_dm / truth_kf_reco
straight from the flat store, so it needs no checkpoint.

    python scripts/kf_baseline_plots.py --store-root /scratch/colliderml/ICLR_eval_geom \
        --out-dir eval_plots/baselines_KF_rebuilt_geom [--datasets ...]
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import fast_rms_eval as fre  # noqa: E402
from track_regression.eval_utils import PARAMS, iterative_rms_convergence  # noqa: E402
from track_regression.paper_plots import apply_paper_style  # noqa: E402
from track_regression.paper_plots.plots._panels import fill_eta_stephist, make_grid  # noqa: E402
from track_regression.paper_plots.stats import DISPLAY_SCALE, DISPLAY_UNIT  # noqa: E402


def load_targets(store_dir: Path) -> np.ndarray:
    man = json.loads((store_dir / "manifest.json").read_text())
    return np.concatenate([np.load(store_dir / p["name"] / "targets.npy", mmap_mode="r")
                           for p in man["parts"]], axis=0)


def build(store_dir: Path) -> dict:
    tg = np.asarray(load_targets(store_dir), np.float64)
    acts, dm = fre.load_flat_acts(store_dir)
    acts = np.asarray(acts, np.float64)
    tkf = fre.load_truth_kf(store_dir)
    sel = np.asarray(dm, bool) & np.isfinite(acts[:, 0])
    series = [("CKF", "C2", acts)]
    if tkf is not None:
        tkf = np.asarray(tkf, np.float64)
        sel &= np.isfinite(tkf[:, 0])
        series.insert(0, ("truth-KF", "C3", tkf))
    out = {"count": int(sel.sum()), "n_total": len(tg), "series": [], "n_dm_any": int(np.asarray(dm, bool).sum())}
    for name, colour, arr in series:
        res = {}
        for i, p in enumerate(PARAMS):
            r = arr[sel, i] - tg[sel, i]
            res[p] = fre._wrap(r) if p == "phi" else r
        out["series"].append((name, colour, res))
    th = tg[sel, 3]
    out["eta"] = -np.log(np.tan(np.clip(th, 1e-8, np.pi - 1e-8) / 2.0))
    return out


def draw(ax, eta, series, p, mode):
    scale, unit = DISPLAY_SCALE[p], DISPLAY_UNIT[p]
    show_post = mode in ("both", "postclip"); show_pre = mode in ("both", "preclip")
    pre_style = "--" if mode == "both" else "-"
    for name, colour, res in series:
        arr = res[p]
        if show_post:
            c, y = fre._profile(eta, arr, fre._iter_rms)
            cut = iterative_rms_convergence(arr)
            ax.plot(c, y * scale, "-", color=colour, lw=1.8,
                    label=f"{name} iter-3σ  μ = {fre._fmt(cut['rms'] * scale)} {unit} (N={fre._fmt_n(cut['n_kept'])})")
        if show_pre:
            c, y = fre._profile(eta, arr, fre._raw_rms)
            ax.plot(c, y * scale, pre_style, color=colour, lw=1.0 if mode == "both" else 1.8,
                    alpha=0.7 if mode == "both" else 1.0,
                    label=f"{name} pre-clip  μ = {fre._fmt(fre._raw_rms(arr) * scale)} {unit}")
    ax.set_xlabel(r"truth $\eta$"); ax.set_ylabel(f"RMSE({p}) [{unit}]")
    ax.set_xlim(-3, 3); ax.set_ylim(bottom=0); ax.set_title(p)


def make_plots(res: dict, out_dir: Path, dataset: str, subtitle: str) -> None:
    eta = res["eta"]
    for mode, stem, title, logy in fre.MODES:
        fig, axes = make_grid()
        for i, p in enumerate(PARAMS):
            draw(axes[i], eta, res["series"], p, mode)
            axes[i].legend(loc="upper left", bbox_to_anchor=(0.02, 0.98), fontsize=7.0,
                           handlelength=1.3, handletextpad=0.4, framealpha=0.85,
                           borderpad=0.25, labelspacing=0.2)
            if logy:
                axes[i].set_yscale("log"); lo, hi = axes[i].get_ylim(); axes[i].set_ylim(top=hi * 3.0)
            else:
                lo, hi = axes[i].get_ylim(); axes[i].set_ylim(bottom=lo, top=hi * 1.20)
        fill_eta_stephist(axes[5], eta)
        h, l = [], []
        for name, colour, _ in res["series"]:
            if mode in ("both", "postclip"):
                h.append(Line2D([0], [0], color=colour, lw=1.8)); l.append(f"{name} (iter-3σ)")
            if mode in ("both", "preclip"):
                h.append(Line2D([0], [0], color=colour, lw=1.0 if mode == "both" else 1.8,
                                ls="--" if mode == "both" else "-", alpha=0.7))
                l.append(f"{name} (pre-clip)")
        fig.legend(h, l, loc="upper center", ncol=len(l), fontsize=9, frameon=False,
                   bbox_to_anchor=(0.5, 0.985))
        names = " and ".join(n for n, _, _ in res["series"])
        fig.suptitle(f"{dataset} — reference fits only ({names}) — {title} — "
                     f"N={res['count']:,} double-matched of {res['n_total']:,}\n{subtitle}", y=1.05)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fre._save_pdf(fig, out_dir, f"{dataset}__kf_baselines_{stem}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--store-root", required=True, help="eval farm root holding <dataset>/test")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--subtitle", default="")
    a = ap.parse_args()
    apply_paper_style()
    root, out = Path(a.store_root), Path(a.out_dir)
    names = a.datasets or sorted(p.name for p in root.iterdir() if (p / "test" / "manifest.json").exists())
    for ds in names:
        res = build(root / ds / "test")
        print(f"{ds}: {res['count']:,} tracks, references: {[n for n, _, _ in res['series']]}", flush=True)
        make_plots(res, out, ds, a.subtitle or f"store: {root}")


if __name__ == "__main__":
    main()
