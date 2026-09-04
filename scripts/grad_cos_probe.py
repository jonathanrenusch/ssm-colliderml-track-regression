#!/usr/bin/env python3
"""Trunk-gradient cosine-similarity probe (paper interpretability figure).

Reimplementation of the campaign-1 probe (logs_Neurips .../grad_cos/summary.txt):
for each perigee parameter, backprop ONLY that parameter's loss term and collect
the gradient on the shared trunk (everything except the output head); report the
mean pairwise cosine over N minibatches, and render the 5x5 heatmap.

Two different uncertainties come out of this and must not be confused:
  * ``std``  -- the scatter of the SINGLE-BATCH cosine across minibatches.  This
    is the per-batch gradient-noise floor; it is a property of the batch size,
    NOT of how many batches we average, so it does not shrink with more batches.
  * ``sem = std / sqrt(N)`` -- the uncertainty on the reported MEAN cosine.  This
    is the error bar that belongs on the heatmap, and it is what more batches buy.
The full per-batch stack is saved so either can be recomputed (or bootstrapped)
without re-running the probe.

Averaging SINGLE-BATCH cosines is also a biased estimate of the alignment of the
true (full-sample) gradients: each minibatch gradient carries independent noise,
which attenuates the cosine toward zero, and it attenuates the noisiest parameter
(q/p) hardest.  So the probe also reports the ``pooled`` cosine -- the cosine of
the gradients accumulated over ALL batches, i.e. one gradient per parameter over
the whole sample -- with a leave-one-block-out jackknife error.

MEASURED (R2L-FT, 450 x 2048 ttbar tracks, 2026-09-03): ``pooled`` is CONFOUNDED
and ``mean`` is the number to quote.  The pooled matrix looks far stronger
(geometry block +0.44..+0.95), but ``systematic_frac`` is only 0.12-0.19 -- the
full-sample gradient is barely above the noise floor -- and projecting out the
direction common to all five tasks collapses it: (d0,z0) +0.67 -> -0.12,
(z0,theta) +0.44 -> -0.29, (phi,qop) +0.09 -> -0.55.  So the pooled block
structure is almost entirely ONE shared direction (this model was fine-tuned on
mix3 and probed on held-out ttbar, and the objective is a plain sum, so the
residual gradients must roughly cancel), not geometric coupling.  ``mean``
measures the correlation of the per-track gradient FLUCTUATIONS, which is the
multi-task-interference quantity the paper cites, and it is not affected by a
common offset.  Only (d0,phi) survives in all three views (+0.797 / +0.952 /
+0.734).  Keep ``pooled`` as a diagnostic, quote ``mean`` +- ``sem``.

Usage: grad_cos_probe.py <config.yaml> <ckpt> <data_dir> <out_dir> [n_batches] [bs] [workers]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import torch  # noqa: E402

from bench_infer_flat import build_model  # noqa: E402
import bench_infer_flat as B  # noqa: E402

PARAMS = ["d0", "z0", "phi", "theta", "qop"]
LABELS = ["$d_0$", "$z_0$", r"$\varphi$", r"$\theta$", "$q/p$"]


def main():
    cfg, ckpt, data_dir, out_dir = sys.argv[1:5]
    n_batches = int(sys.argv[5]) if len(sys.argv) > 5 else 20
    bs = int(sys.argv[6]) if len(sys.argv) > 6 else 2048
    workers = int(sys.argv[7]) if len(sys.argv) > 7 else 8
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)

    variant = sys.argv[8] if len(sys.argv) > 8 else ""
    B.SEED_RESIDUALS = True
    model = build_model(Path(cfg), Path(ckpt), "cuda:0")
    if variant:
        # d_conv=1 checkpoints: the stock Mamba2 block cannot even run
        # (causal_conv1d requires width 2-4) and the native short block has no
        # packed TRAIN path -- swap to the training kernel (v3c, exact
        # autograd), exactly what the training itself ran.
        from track_regression.mamba_short import apply_variant
        apply_variant(model, variant)
    # train() only to keep the encoder's DDP tie (`+ 0.0 * x_hits.sum()`, an
    # exact-zero gradient) in the graph; dropout is 0.0 and the eval-only
    # auto-seed branch cannot fire here (the collate already supplies the 15
    # seed-residual features), so numerics are unchanged.
    model = model.train()

    # trunk = everything except the output head(s)
    trunk_names = [n for n, p in model.named_parameters()
                   if not n.startswith("output_head")]
    trunk = [dict(model.named_parameters())[n] for n in trunk_names]
    n_trunk = sum(p.numel() for p in trunk)
    print(f"[gradcos] trunk params: {n_trunk:,} ({len(trunk)} tensors; output_head excluded)", flush=True)

    from track_regression.data import ColliderMLRegrDataModule
    dm = ColliderMLRegrDataModule(preprocessed_dir=data_dir, batch_size=bs, num_workers=workers,
                                  pin_memory=False, packed_batches=True, load_acts=False,
                                  seed_residual_features=True)
    dm.setup("test")
    loader = dm.test_dataloader()

    # Block-wise raw (unnormalised) gradient accumulators for the pooled
    # estimator: 5 params x n_trunk floats per block.  Blocks are what the
    # jackknife resamples, so the pooled number gets an error bar too.
    n_blocks = 10
    blk = torch.zeros(n_blocks, len(PARAMS), n_trunk, device="cuda")
    # mean single-batch gradient norm per parameter -- with the pooled norm this
    # says whether the full-sample gradient is systematic- or noise-dominated:
    # ||sum_b g_b|| / (N * mean||g_b||) ~ 1 means one fixed direction every
    # batch, ~1/sqrt(N) means the batches are pure noise about zero.
    bnorm = torch.zeros(len(PARAMS), device="cuda")

    loss_mod = model.loss_module
    per_batch = []
    for inputs, targets in loader:
        inputs = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in inputs.items()}
        targets = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in targets.items()}
        # ONE forward, five backwards through the retained graph: the five
        # per-parameter losses read disjoint slices of `pred`, so each
        # autograd.grad gives exactly that parameter's trunk gradient -- and all
        # five now come from a single, identical forward pass.
        outp = model(inputs)
        losses = loss_mod(outp["pred"], targets, valid_mask=targets.get("track_valid"))
        grads = []
        for i, p in enumerate(PARAMS):
            gs = torch.autograd.grad(losses[p], trunk, retain_graph=(i < len(PARAMS) - 1),
                                     allow_unused=True)
            g = torch.cat([(q.flatten() if q is not None else torch.zeros(t.numel(), device="cuda"))
                           for q, t in zip(gs, trunk)])
            blk[len(per_batch) % n_blocks, i] += g
            bnorm[i] += g.norm()
            grads.append(g / (g.norm() + 1e-12))
        G = torch.stack(grads)
        per_batch.append((G @ G.T).cpu().numpy())
        nb = len(per_batch)
        if nb % 20 == 0 or nb == n_batches:
            print(f"[gradcos] batch {nb}/{n_batches}", flush=True)
        if nb >= n_batches:
            break

    C = np.stack(per_batch)                       # (N, 5, 5) single-batch cosines
    nb = C.shape[0]
    mean = C.mean(axis=0)
    std = C.std(axis=0, ddof=1) if nb > 1 else np.zeros_like(mean)   # per-batch noise floor
    sem = std / np.sqrt(nb)                       # uncertainty on `mean`

    # Pooled estimator: cosine of the gradients summed over the whole sample.
    def _cos(S):                                  # S: (5, n_trunk) raw gradients
        U = S / (S.norm(dim=1, keepdim=True) + 1e-12)
        return (U @ U.T).cpu().numpy()
    tot = blk.sum(0)
    pooled = _cos(tot)
    # Is the pooled alignment just a direction ALL five tasks share?  The model
    # was fine-tuned on mix3, so on this held-out ttbar sample every task can
    # carry the same domain-shift offset, which inflates every pairwise cosine.
    # Project that common direction out and re-measure.
    u = tot / (tot.norm(dim=1, keepdim=True) + 1e-12)
    common = u.sum(0)
    common = common / (common.norm() + 1e-12)
    pooled_nocommon = _cos(tot - (tot @ common)[:, None] * common[None, :])
    pooled_norm = tot.norm(dim=1).cpu().numpy()
    mean_batch_norm = (bnorm / nb).cpu().numpy()
    systematic_frac = pooled_norm / (nb * mean_batch_norm)
    # leave-one-block-out jackknife: var = (B-1)/B * sum (x_i - x_bar)^2
    jk = np.stack([_cos(blk.sum(0) - blk[b]) for b in range(n_blocks)])
    pooled_err = np.sqrt((n_blocks - 1) / n_blocks * ((jk - jk.mean(0)) ** 2).sum(0))

    np.savez(out / "grad_cosines.npz", mean=mean, std=std, sem=sem, per_batch=C,
             pooled=pooled, pooled_err=pooled_err, pooled_nocommon=pooled_nocommon,
             pooled_norm=pooled_norm, mean_batch_norm=mean_batch_norm,
             systematic_frac=systematic_frac, n_batches=nb, batch_size=bs,
             n_blocks=n_blocks, params=PARAMS)

    def _block(f, M, fmt):
        f.write("        " + "".join(f"{p:>9s}" for p in PARAMS) + "\n")
        for i, p in enumerate(PARAMS):
            f.write(f"{p:>7s} " + "".join(f"  {M[i, j]:{fmt}}" for j in range(5)) + "\n")

    with open(out / "summary.txt", "w") as f:
        f.write(f"Config: {cfg}\nCheckpoint: {ckpt}\nData: {data_dir}\n"
                f"N batches: {nb}  BS={bs}  ({nb * bs:,} tracks)\nTrunk params: {n_trunk:,}\n\n"
                "Mean cosine matrix:\n")
        _block(f, mean, "+.3f")
        f.write("\nStd across batches (per-batch noise floor; independent of N):\n")
        _block(f, std, ".3f")
        f.write(f"\nStandard error on the mean (std/sqrt({nb})):\n")
        _block(f, sem, ".4f")
        f.write("\nMean / sem (significance of each entry):\n")
        _block(f, np.divide(mean, sem, out=np.zeros_like(mean), where=sem > 0), "+.1f")
        f.write(f"\nPOOLED cosine -- gradients accumulated over all {nb * bs:,} tracks\n"
                "(no minibatch-noise attenuation; this is the alignment of the true gradients):\n")
        _block(f, pooled, "+.3f")
        f.write(f"\nPooled jackknife error ({n_blocks} blocks):\n")
        _block(f, pooled_err, ".4f")
        f.write("\nPooled cosine with the direction common to all five tasks projected out\n"
                "(guards against a shared domain-shift offset inflating every entry):\n")
        _block(f, pooled_nocommon, "+.3f")
        f.write(f"\nGradient character per parameter (1.0 = one fixed direction every batch,\n"
                f"{1/np.sqrt(nb):.3f} = pure noise about zero at N={nb}):\n")
        f.write("        " + "".join(f"{p:>9s}" for p in PARAMS) + "\n")
        f.write("  syst. " + "".join(f"  {v:+.3f}" for v in systematic_frac) + "\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    im = ax.imshow(mean, vmin=-0.2, vmax=0.2, cmap="RdBu_r")
    disp = np.where(np.eye(5, dtype=bool), 1.0, mean)
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f"{disp[i, j]:+.2f}", ha="center", va="center",
                    fontsize=9, color="black")
    ax.set_xticks(range(5), LABELS); ax.set_yticks(range(5), LABELS)
    ax.set_title("Trunk-gradient cosine similarity")
    fig.colorbar(im, ax=ax, shrink=0.85, label="mean cosine")
    fig.tight_layout()
    fig.savefig(out / "cos_heatmap.pdf", bbox_inches="tight")
    print(f"[gradcos] wrote {out}/cos_heatmap.pdf", flush=True)


if __name__ == "__main__":
    main()
