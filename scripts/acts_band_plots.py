#!/usr/bin/env python3
"""Legacy-design resolution plots from the OFFICIAL pipeline's own fit profiles,
with shaded uncertainty bands from the ACTS writers' per-bin fit-sigma errors
(NOT bootstrap).

Reads ``resolution_data.pkl`` (written by acts_integration.py) and renders:

- ``<ds>_acts__res_vs_eta_bands.pdf``  — every dataset: sigma vs eta, campaign
  colors (SSM C0, ACTS KF C3), band = sigma +- delta_sigma (writer errors),
  legend = the UNBINNED (eta+pT-integrated) Gaussian sigma +- its error.
- ``<ds>_acts__res_vs_pt_bands.pdf``   — --with-pt only (the uniform sample):
  same design plus a ratio-to-ACTS-KF sub-panel per parameter with the
  propagated error band.

Usage: acts_band_plots.py <official_out_dir> <dataset_label> [--with-pt]
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec  # noqa: E402

PARAMS = ["d0", "z0", "phi", "theta", "qop", "qopt_rel"]
LABELS = {"d0": ("d0", "µm", 1e3), "z0": ("z0", "µm", 1e3), "phi": ("phi", "mrad", 1e3),
          "theta": ("theta", "mrad", 1e3), "qop": ("qop", "1/GeV", 1.0),
          "qopt_rel": ("rel. q/pT", "%", 1.0)}
COLORS = {"SSM": "C0", "ACTS KF": "C3"}
BAND_ALPHA = 0.25  # legacy rms_vs_eta band style


def _fmt(v):
    return f"{v:.3g}" if (abs(v) >= 0.01 or v == 0) else f"{v:.2e}"


def _clean(curve, scale):
    c = np.asarray(curve["centers"], float)
    v = np.asarray(curve["values"], float) * scale
    e = np.asarray(curve["errors"], float) * scale
    ok = np.isfinite(v) & (v > 0)
    return c[ok], v[ok], e[ok]


def _n_matched(out_dir):
    p = out_dir / "matched_residuals.npz"
    if not p.exists():
        return None
    z = np.load(p)
    both = np.isfinite(z["ssm"][:, 0]) & np.isfinite(z["kf"][:, 0])
    return int(both.sum()), len(z["truth"])


def draw_pages(bundle: dict, out_dir: Path, dataset: str, with_pt: bool):
    data, integ = bundle["data"], bundle.get("integrated", {})
    nboth = _n_matched(out_dir)
    ncap = (f"; $N_{{\\mathrm{{both}}}}={nboth[0]:,}$ of {nboth[1]:,} matched"
            if nboth else "")
    variants = [("eta", "res_vs_eta_bands", "truth $\\eta$")]
    if with_pt:
        variants.append(("pT", "res_vs_pt_bands", "$p_{\\mathrm{T}}$ [GeV]"))
    for var, stem, xlabel in variants:
        fig = plt.figure(figsize=(15, 9.6))
        gs = GridSpec(2, 3, figure=fig, hspace=0.34, wspace=0.26)
        for i, p in enumerate(PARAMS):
            name, unit, scale = LABELS[p]
            sub = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[i],
                                          height_ratios=[3, 1], hspace=0.06)
            ax = fig.add_subplot(sub[0]); axr = fig.add_subplot(sub[1], sharex=ax)
            curves = {}
            for label, colour in COLORS.items():
                if label not in data:
                    continue
                c, v, e = _clean(data[label][p][var], scale)
                curves[label] = (c, v, e)
                ig = integ.get(label, {}).get(p)
                leg = (f"{label}: $\\sigma = {_fmt(ig['sigma'] * scale)} \\pm "
                       f"{_fmt(ig['sigmaErr'] * scale)}$ {unit}" if ig else label)
                ax.plot(c, v, "-", color=colour, lw=1.8, label=leg)
                ax.fill_between(c, v - e, v + e, color=colour, alpha=BAND_ALPHA, lw=0)
            ax.set_ylabel(f"$\\sigma$({name}) [{unit}]")
            ax.set_title(name)
            # robust top: 1.15x the 95th pct of both curves, so a single
            # degenerate KF-fit bin does not blow up the panel
            allv = np.concatenate([v for (_, v, _) in curves.values()]) if curves else np.array([1.0])
            ax.set_ylim(0, 1.15 * float(np.percentile(allv, 95)))
            ax.legend(loc="upper left", fontsize=7.2, framealpha=0.85,
                      handlelength=1.3, borderpad=0.25, labelspacing=0.2)
            if "SSM" in curves and "ACTS KF" in curves:
                cs, vs, es = curves["SSM"]; ck, vk, ek = curves["ACTS KF"]
                common, is_, ik_ = np.intersect1d(cs, ck, return_indices=True)
                r = vs[is_] / vk[ik_]
                re = r * np.sqrt((es[is_] / vs[is_]) ** 2 + (ek[ik_] / vk[ik_]) ** 2)
                axr.axhline(1.0, color="0.4", lw=0.8, ls=":")
                axr.plot(common, r, "-", color="C0", lw=1.5)
                axr.fill_between(common, r - re, r + re, color="C0", alpha=BAND_ALPHA, lw=0)
                axr.set_ylabel("SSM / KF", fontsize=8)
            axr.set_xlabel(xlabel)
            plt.setp(ax.get_xticklabels(), visible=False)
        fig.suptitle(f"{dataset} --- Gaussian-fit resolution vs {xlabel} "
                     f"(ACTS pipeline; bands = ACTS fit-$\\sigma$ uncertainties{ncap})",
                     y=0.995)
        fig.savefig(out_dir / f"{dataset}_acts__{stem}.pdf", bbox_inches="tight")
        plt.close(fig)
        print(f"[bands] {out_dir / f'{dataset}_acts__{stem}.pdf'}", flush=True)


def main():
    out_dir = Path(sys.argv[1]); dataset = sys.argv[2]
    with_pt = "--with-pt" in sys.argv[3:]
    bundle = pickle.load(open(out_dir / "resolution_data.pkl", "rb"))
    draw_pages(bundle, out_dir, dataset, with_pt)


if __name__ == "__main__":
    main()
