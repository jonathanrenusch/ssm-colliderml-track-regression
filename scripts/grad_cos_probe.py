#!/usr/bin/env python3
"""Trunk-gradient cosine-similarity probe (paper interpretability figure).

Reimplementation of the campaign-1 probe (logs_Neurips .../grad_cos/summary.txt):
for each perigee parameter, backprop ONLY that parameter's loss term and collect
the gradient on the shared trunk (everything except the output head); report the
mean pairwise cosine over N minibatches, and render the 5x5 heatmap.

Usage: grad_cos_probe.py <config.yaml> <ckpt> <data_dir> <out_dir> [n_batches] [bs]
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
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)

    B.SEED_RESIDUALS = True
    model = build_model(Path(cfg), Path(ckpt), "cuda:0")
    model = model.train()  # gradients; dropout-free model so numerics unchanged

    # trunk = everything except the output head(s)
    head_prefixes = ("output_head", "pool_head")
    trunk_names = [n for n, p in model.named_parameters()
                   if not n.startswith("output_head")]
    trunk = [dict(model.named_parameters())[n] for n in trunk_names]
    n_trunk = sum(p.numel() for p in trunk)
    print(f"[gradcos] trunk params: {n_trunk:,} ({len(trunk)} tensors; output_head excluded)", flush=True)

    from track_regression.data import ColliderMLRegrDataModule
    dm = ColliderMLRegrDataModule(preprocessed_dir=data_dir, batch_size=bs, num_workers=0,
                                  pin_memory=False, packed_batches=True, load_acts=False,
                                  seed_residual_features=True)
    dm.setup("test")
    loader = dm.test_dataloader()

    loss_mod = model.loss_module
    cos_sum = np.zeros((5, 5)); cos_sq = np.zeros((5, 5)); nb = 0
    for inputs, targets in loader:
        inputs = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in inputs.items()}
        targets = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in targets.items()}
        grads = []
        for p in PARAMS:
            model.zero_grad(set_to_none=True)
            outp = model(inputs)
            losses = loss_mod(outp["pred"], targets)  # dict of per-parameter losses
            loss_val = losses[p]
            loss_val.backward()  # cosine of NORMALISED grads -> head weights irrelevant
            g = torch.cat([q.grad.flatten() if q.grad is not None else torch.zeros(q.numel(), device="cuda")
                           for q in trunk])
            grads.append(g / (g.norm() + 1e-12))
        G = torch.stack(grads)
        C = (G @ G.T).cpu().numpy()
        cos_sum += C; cos_sq += C ** 2; nb += 1
        print(f"[gradcos] batch {nb}/{n_batches}", flush=True)
        if nb >= n_batches:
            break
    mean = cos_sum / nb
    std = np.sqrt(np.maximum(cos_sq / nb - mean ** 2, 0))
    np.savez(out / "grad_cosines.npz", mean=mean, std=std, params=PARAMS)

    with open(out / "summary.txt", "w") as f:
        f.write(f"Config: {cfg}\nCheckpoint: {ckpt}\nData: {data_dir}\n"
                f"N batches: {nb}  BS={bs}\nTrunk params: {n_trunk:,}\n\nMean cosine matrix:\n")
        f.write("        " + "".join(f"{p:>9s}" for p in PARAMS) + "\n")
        for i, p in enumerate(PARAMS):
            f.write(f"{p:>7s} " + "".join(f"  {mean[i, j]:+.3f}" for j in range(5)) + "\n")
        f.write("\nStd across batches:\n")
        for i, p in enumerate(PARAMS):
            f.write(f"{p:>7s} " + "".join(f"  {std[i, j]:.3f}" for j in range(5)) + "\n")

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
