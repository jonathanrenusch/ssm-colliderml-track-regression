#!/usr/bin/env python3
"""Input-sensitivity ("Hessian sensitivity") page for the paper (reviewer
request 2026-09-11).

For each predicted perigee parameter p and each of the 15 per-hit input
features f we measure how strongly the prediction responds to a STANDARDISED
change in that feature (gradient x feature std, so heterogeneous units -- mm,
rad, integer ids -- are comparable; a bare 1/[feature] gradient just reflects
units, not learned importance):

  first order  (Jacobian)  S1[p,f] = RMS_{track,hit} | d pred_p / d x_{hit,f} | * std(x_f)
  second order (Hessian)   S2[p,f] = RMS_{track,hit} | d^2 pred_p / d x_{hit,f}^2 | * std(x_f)^2

S2 (the input Hessian diagonal) is a curvature / non-linearity measure,
estimated with a Hutchinson probe (Rademacher v; the diagonal is E[v * (H v)]).
It needs double backward, which the fused/compiled kernels do not provide, so
S2 is computed on the eager pure-torch scan (v3) and skipped with a note if
even that path refuses second-order autograd.

Both matrices are shown as (5 output x 15 feature) heatmaps in the paper's
Blues palette, each row normalised to its own maximum so feature *ranking*
per parameter is the readable quantity (absolute scales differ per head and
per order).  Physically the network should lean on the three seed-residual
features and the geometry, not re-derive the fit from raw (x,y,z).

Usage: input_sensitivity.py <config.yaml> <ckpt> <eval_store_test_dir> <out_dir> [n_tracks]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import torch  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from bench_infer_flat import build_model  # noqa: E402
import bench_infer_flat as B  # noqa: E402
from track_regression.mamba_short import apply_variant  # noqa: E402
from track_regression.data import ColliderMLRegrDataModule  # noqa: E402

PARAMS = ["d0", "z0", "phi", "theta", "qop"]
LABELS = ["$d_0$", "$z_0$", r"$\varphi$", r"$\theta$", "$q/p$"]
FEATURES = ["x", "y", "z", "r", r"$\varphi_{\rm hit}$", r"$\theta_{\rm hit}$", "s",
            "vol", "layer", "surf", "det", r"$\eta_{\rm hit}$",
            r"$\Delta u$", r"$\Delta v$", r"$s_{\rm helix}$"]


def _batch(store_dir, n_tracks):
    dm = ColliderMLRegrDataModule(preprocessed_dir=str(Path(store_dir).parent),
                                  batch_size=n_tracks, num_workers=0,
                                  pin_memory=False, packed_batches=True,
                                  load_acts=False, seed_residual_features=True)
    dm.setup("test")
    for inputs, targets in dm.test_dataloader():
        return inputs, targets


def _point(model, inputs):
    out = model(inputs)
    return model.loss_module.predict(out["pred"])   # dict param -> (T,) delta pred


def main():
    cfg, ckpt, store_dir, out_dir = sys.argv[1:5]
    n_tracks = int(sys.argv[5]) if len(sys.argv) > 5 else 3000
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)

    B.SEED_RESIDUALS = True
    model = build_model(Path(cfg), Path(ckpt), "cuda:0").eval()

    inputs, _ = _batch(store_dir, n_tracks)
    inputs = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in inputs.items()}
    hf0 = inputs["hit_features"]                       # (1, H, 15)
    nfeat = hf0.shape[-1]
    # per-feature std over all hits: converts the bare 1/[feature] gradient into
    # a response per standard-deviation change, comparable across units.  Guard
    # zero-variance columns (a constant id in one sample) with 1.0.
    fstd = hf0[0].std(dim=0)
    fstd = torch.where(fstd > 1e-12, fstd, torch.ones_like(fstd)).cpu().numpy()

    # ---- first order (any differentiable kernel; use the training kernel v3c)
    apply_variant(model, "v3c")
    S1 = np.zeros((len(PARAMS), nfeat))
    hf = hf0.detach().clone().requires_grad_(True)
    inp = dict(inputs); inp["hit_features"] = hf
    preds = _point(model, inp)
    for i, p in enumerate(PARAMS):
        g, = torch.autograd.grad(preds[p].sum(), hf,
                                 retain_graph=(i < len(PARAMS) - 1), create_graph=False)
        S1[i] = g.detach().abs().pow(2).mean(dim=(0, 1)).sqrt().cpu().numpy() * fstd

    # ---- second order (input Hessian diagonal, Hutchinson on the eager path)
    S2 = None
    note2 = ""
    try:
        apply_variant(model, "v3")                    # eager, double-backward-capable
        n_probe = 8
        acc = np.zeros((len(PARAMS), nfeat))
        for i, p in enumerate(PARAMS):
            diag = torch.zeros_like(hf0[0])            # (H, 15)
            for _ in range(n_probe):
                hfp = hf0.detach().clone().requires_grad_(True)
                inp = dict(inputs); inp["hit_features"] = hfp
                pr = _point(model, inp)[p].sum()
                g, = torch.autograd.grad(pr, hfp, create_graph=True)
                v = (torch.randint(0, 2, hfp.shape, device=hfp.device).float() * 2 - 1)
                hv, = torch.autograd.grad((g * v).sum(), hfp, retain_graph=False)
                diag += (v[0] * hv[0]).detach()
            diag /= n_probe
            acc[i] = diag.abs().pow(2).mean(dim=0).sqrt().cpu().numpy() * (fstd ** 2)
        S2 = acc
    except Exception as e:  # noqa: BLE001
        note2 = f"(second-order skipped: {type(e).__name__})"
        print(f"[sens] Hessian diagonal unavailable {note2}", flush=True)

    # ---- render
    def _panel(ax, S, title):
        Sn = S / (S.max(axis=1, keepdims=True) + 1e-30)
        im = ax.imshow(Sn, vmin=0.0, vmax=1.0, cmap="Blues", aspect="auto")
        ax.set_xticks(range(nfeat), FEATURES[:nfeat], rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(PARAMS)), LABELS, fontsize=10)
        ax.axvline(11.5, color="0.4", lw=1.0, ls="--")   # absolute | seed-residual split
        ax.set_title(title, fontsize=11)
        return im

    ncol = 1 if S2 is None else 2
    fig, axes = plt.subplots(1, ncol, figsize=(7.0 * ncol, 3.6), squeeze=False)
    im = _panel(axes[0][0], S1, "first order  $|\\partial\\,\\mathrm{pred}/\\partial x|$ (RMS over hits)")
    if S2 is not None:
        _panel(axes[0][1], S2, "second order  $|\\partial^2\\,\\mathrm{pred}/\\partial x^2|$ (Hutchinson)")
    cb = fig.colorbar(im, ax=axes[0].tolist(), shrink=0.85, pad=0.02)
    cb.set_label("sensitivity, row-normalised")
    fig.suptitle(f"Input sensitivity of the deployment model "
                 f"($N={n_tracks:,}$ tracks; features left of the dashed line are "
                 f"absolute, right are the seed residuals) {note2}", y=1.04, fontsize=10)
    out_f = out / "input_sensitivity.pdf"
    fig.savefig(out_f, bbox_inches="tight")
    print(f"[sens] {out_f}", flush=True)
    np.savez(out / "input_sensitivity.npz", S1=S1, S2=(S2 if S2 is not None else np.zeros(0)),
             params=PARAMS, features=[f for f in FEATURES[:nfeat]])


if __name__ == "__main__":
    main()
