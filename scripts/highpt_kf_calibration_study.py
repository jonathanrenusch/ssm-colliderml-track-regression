#!/usr/bin/env python3
"""Why does the SSM still gain on the truth-KF above ~90 GeV inside |eta|<=2?
(INTERNAL analysis, 2026-09-06 -- stakeholder follow-up.)

Reads the full-acceptance matched_residuals.npz of the uniform and 100 GeV
samples (deploy predictions vs the shipped truth-KF) and produces:

  1. ratio_vs_pt.pdf      -- per-parameter SSM/tKF resolution ratio in fine pT
                             bins (60-110 GeV) inside |eta|<=2, three
                             estimators per bin: core (68% quantile width),
                             iter-3sigma RMS, and q99.9 (far tail).  Separates
                             "core miscalibration" from "tail population".
  2. ratio_vs_eta_highpt.pdf -- the same ratio vs |eta| for pT>90 (and the
                             20-50 GeV control): is the residual gain
                             edge-of-acceptance leakage?
  3. residuals_bin_<lo>-<hi>.pdf -- residual histograms (log-y, +-12 iter-3sigma)
                             in the problem bins 80-90 / 90-100 / 100-110 GeV
                             and for the 100 GeV sample, with core/iter-3sigma/
                             q99.9 per curve.
  4. summary.txt          -- the pT_max and eta_max scans: what must be
                             clipped for every per-bin ratio to stay >= 0.97.

Usage: highpt_kf_calibration_study.py <uniform_npz> <100gev_npz> <out_dir>
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from track_regression.eval_utils import iterative_rms_convergence  # noqa: E402

P = ["d0", "z0", "phi", "theta", "qop"]
UNIT = {"d0": "µm", "z0": "µm", "phi": "mrad", "theta": "mrad", "qop": "1/GeV"}
SC = {"d0": 1e3, "z0": 1e3, "phi": 1e3, "theta": 1e3, "qop": 1.0}


def wrap(a):
    return np.remainder(a + np.pi, 2 * np.pi) - np.pi


def load(npz_path):
    z = np.load(npz_path)
    truth, ssm, kf = z["truth"], z["ssm"], z["kf"]
    both = np.isfinite(ssm[:, 0]) & np.isfinite(kf[:, 0])
    truth, ssm, kf = truth[both], ssm[both], kf[both]
    th = truth[:, 3]
    eta = -np.log(np.tan(np.clip(th, 1e-8, np.pi - 1e-8) / 2.0))
    pt = np.sin(th) / np.maximum(np.abs(truth[:, 4]), 1e-12)
    res = {}
    for i, p in enumerate(P):
        s, k = ssm[:, i] - truth[:, i], kf[:, i] - truth[:, i]
        if p == "phi":
            s, k = wrap(s), wrap(k)
        res[("SSM", p)], res[("tKF", p)] = s, k
    return res, eta, pt, truth


def core68(x):
    q = np.quantile(x, [0.16, 0.84])
    return 0.5 * (q[1] - q[0])


def q999(x):
    return np.quantile(np.abs(x - np.median(x)), 0.999)


def it3(x):
    return iterative_rms_convergence(x)["rms"]


ESTIMATORS = [("core (68% width)", core68), ("iter-3σ RMS", it3), ("q99.9 |res|", q999)]


def ratio_vs_pt(res, eta, pt, out):
    m2 = np.abs(eta) <= 2.0
    edges = np.array([60, 70, 80, 85, 90, 95, 100, 105, 110], float)
    cx = 0.5 * (edges[:-1] + edges[1:])
    fig, axes = plt.subplots(1, 5, figsize=(18, 3.6), sharex=True)
    lines = ["\nratio SSM/tKF in fine pT bins (|eta|<=2); columns: core / iter3s / q99.9"]
    for i, p in enumerate(P):
        ax = axes[i]
        for (lab, fn), style in zip(ESTIMATORS, ["-o", "-s", "-^"]):
            r = []
            for lo, hi in zip(edges[:-1], edges[1:]):
                m = m2 & (pt >= lo) & (pt < hi)
                r.append(fn(res[("SSM", p)][m]) / fn(res[("tKF", p)][m]) if m.sum() > 500 else np.nan)
            ax.plot(cx, r, style, ms=3.5, lw=1.3, label=lab)
        ax.axhline(1.0, color="0.4", ls=":", lw=0.9)
        ax.set_title(p); ax.set_xlabel(r"$p_\mathrm{T}$ [GeV]")
        if i == 0:
            ax.set_ylabel("SSM / truth-KF"); ax.legend(fontsize=7)
        row = [f"{it3(res[('SSM', p)][m2 & (pt >= lo) & (pt < hi)]) / it3(res[('tKF', p)][m2 & (pt >= lo) & (pt < hi)]):.3f}"
               for lo, hi in zip(edges[:-1], edges[1:])]
        lines.append(f"  {p:6s} iter3s: " + "  ".join(f"{lo:.0f}-{hi:.0f}:{v}" for lo, hi, v in zip(edges[:-1], edges[1:], row)))
    fig.suptitle("uniform muons, |η|≤2 — SSM/truth-KF resolution ratio vs pT (core vs clipped vs far tail)")
    fig.tight_layout(); fig.savefig(out / "ratio_vs_pt.pdf", bbox_inches="tight"); plt.close(fig)
    return lines


def ratio_vs_eta(res, eta, pt, out, sel_lists):
    fig, axes = plt.subplots(1, 5, figsize=(18, 3.6), sharex=True)
    ae = np.abs(eta)
    edges = np.linspace(0, 2.0, 9)
    cx = 0.5 * (edges[:-1] + edges[1:])
    lines = ["\nratio (iter3s) vs |eta| per pT selection:"]
    for i, p in enumerate(P):
        ax = axes[i]
        for lab, msel, col in sel_lists:
            r = []
            for lo, hi in zip(edges[:-1], edges[1:]):
                m = msel & (ae >= lo) & (ae < hi)
                r.append(it3(res[("SSM", p)][m]) / it3(res[("tKF", p)][m]) if m.sum() > 500 else np.nan)
            ax.plot(cx, r, "-o", ms=3.5, lw=1.3, color=col, label=lab)
            if i == 0 or p == "phi":
                lines.append(f"  {p:6s} {lab:12s}: " + "  ".join(f"{c:.2f}:{v:.3f}" for c, v in zip(cx, r) if np.isfinite(v)))
        ax.axhline(1.0, color="0.4", ls=":", lw=0.9)
        ax.set_title(p); ax.set_xlabel(r"$|\eta|$")
        if i == 0:
            ax.set_ylabel("SSM / truth-KF (iter-3σ)"); ax.legend(fontsize=7)
    fig.suptitle("uniform muons — SSM/truth-KF iter-3σ ratio vs |η|: is the high-pT gain edge-of-acceptance leakage?")
    fig.tight_layout(); fig.savefig(out / "ratio_vs_eta_highpt.pdf", bbox_inches="tight"); plt.close(fig)
    return lines


def residual_pages(res, eta, pt, out, tag, bins):
    m2 = np.abs(eta) <= 2.0
    for lo, hi in bins:
        m = m2 & (pt >= lo) & (pt < hi)
        if m.sum() < 1000:
            continue
        fig, axes = plt.subplots(2, 3, figsize=(15, 8.2))
        axes = axes.ravel()
        for i, p in enumerate(P):
            ax = axes[i]
            sc = SC[p]
            ref3 = it3(res[("tKF", p)][m])
            lox, hix = -12 * ref3, 12 * ref3
            for tag2, col in (("SSM", "C0"), ("tKF", "C3")):
                arr = res[(tag2, p)][m]
                ax.hist(np.clip(arr, lox, hix) * sc, bins=240, range=(lox * sc, hix * sc),
                        histtype="step", lw=1.3, color=col,
                        label=(f"{tag2}: core {core68(arr)*sc:.3g}, iter3σ {it3(arr)*sc:.3g}, "
                               f"q99.9 {q999(arr)*sc:.3g} {UNIT[p]}"))
            ax.set_yscale("log"); ax.set_xlabel(f"residual({p}) [{UNIT[p]}]"); ax.set_ylabel("tracks/bin")
            ax.set_title(f"{p} (axis ±12 tKF-iter3σ, overflow in edge bins)")
            ax.legend(fontsize=6.2, loc="upper right")
        ae = np.abs(eta[m])
        bad = np.abs(res[("tKF", "phi")][m]) > 3 * it3(res[("tKF", "phi")][m])
        axes[5].hist(ae, bins=40, range=(0, 2), histtype="step", color="0.4", density=True, label="all")
        if bad.sum() > 20:
            axes[5].hist(ae[bad], bins=40, range=(0, 2), histtype="step", color="C3", density=True,
                         label=f"tKF |φ res|>3σ (N={bad.sum():,})")
        axes[5].set_xlabel(r"$|\eta|$"); axes[5].set_ylabel("density"); axes[5].legend(fontsize=7)
        axes[5].set_title("where the KF φ outliers live")
        fig.suptitle(f"{tag}, pT {lo:.0f}–{hi:.0f} GeV, |η|≤2 — residuals, log-y (N={m.sum():,})")
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(out / f"residuals_{tag}_pt{lo:.0f}-{hi:.0f}.pdf", bbox_inches="tight")
        plt.close(fig)


def scans(res, eta, pt):
    m2 = np.abs(eta) <= 2.0
    lines = ["\npT_max scan: worst per-parameter iter3s ratio in any 10-GeV bin ending <= pmax (|eta|<=2):"]
    for pmax in (80, 90, 100, 110):
        worst = 1.0; worst_lab = ""
        for lo in range(10, pmax, 10):
            hi = min(lo + 10, pmax)
            m = m2 & (pt >= lo) & (pt < hi)
            if m.sum() < 2000:
                continue
            for p in P:
                r = it3(res[("SSM", p)][m]) / it3(res[("tKF", p)][m])
                if r < worst:
                    worst, worst_lab = r, f"{p} @ {lo}-{hi}"
        lines.append(f"  pmax={pmax:3d}: worst bin ratio {worst:.3f}  ({worst_lab})")
    lines.append("\neta_max scan at pT>90 (integrated iter3s ratio, worst parameter):")
    for emax in (1.0, 1.5, 1.8, 2.0):
        m = (np.abs(eta) <= emax) & (pt > 90)
        worst = min(it3(res[("SSM", p)][m]) / it3(res[("tKF", p)][m]) for p in P)
        lines.append(f"  |eta|<={emax}: worst-parameter ratio {worst:.3f}  (N={m.sum():,})")
    return lines


def main():
    uni_npz, gev100_npz, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    out.mkdir(parents=True, exist_ok=True)
    res, eta, pt, _ = load(uni_npz)
    lines = [f"high-pT KF-calibration study, uniform sample: {len(eta):,} DM tracks (full acceptance)"]
    lines += ratio_vs_pt(res, eta, pt, out)
    m2 = np.abs(eta) <= 2.0
    lines += ratio_vs_eta(res, eta, pt, out, [
        ("pT 20–50", m2 & (pt >= 20) & (pt < 50), "0.5"),
        ("pT 80–90", m2 & (pt >= 80) & (pt < 90), "C2"),
        ("pT > 90", m2 & (pt > 90), "C1"),
    ])
    residual_pages(res, eta, pt, out, "uniform", [(80, 90), (90, 100), (100, 110)])
    lines += scans(res, eta, pt)
    res1, eta1, pt1, _ = load(gev100_npz)
    residual_pages(res1, eta1, pt1, out, "100GeV", [(95, 105)])
    m1 = np.abs(eta1) <= 2.0
    lines.append("\n100 GeV sample, |eta|<=2 iter3s ratios: " +
                 "  ".join(f"{p}:{it3(res1[('SSM', p)][m1]) / it3(res1[('tKF', p)][m1]):.3f}" for p in P))
    (out / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nplots -> {out}")


if __name__ == "__main__":
    main()
