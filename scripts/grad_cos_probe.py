#!/usr/bin/env python3
"""Trunk-gradient cosine-similarity probe (appendix figure).

For each perigee parameter, backpropagate only that parameter's loss and take
the gradient on the shared trunk (every parameter except the output head);
report the cosine between the five gradients, averaged over minibatches.
``sem = std / sqrt(N)`` is the uncertainty on the mean (``std`` is the scatter
of single-batch cosines, which does not shrink with N).

    python scripts/grad_cos_probe.py --config <cfg> --ckpt <ckpt> --data-dir <store> \
        --out-dir figures/grad_cos [--n-batches 450] [--batch-size 2048]

Writes ``grad_cosines.npz`` (mean, std, sem, per_batch) and ``summary.txt``;
plot with ``scripts/plot_grad_cos.py``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from track_regression.data import ColliderMLRegrDataModule
from track_regression.inference import load_model

PARAMS = ["d0", "z0", "phi", "theta", "qop"]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data-dir", required=True, help="store root with a test/ split")
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--n-batches", type=int, default=450)
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.set_float32_matmul_precision("highest")

    # training-path forward in fp32 (the data loader supplies the seed and its residual features)
    model = load_model(args.config, args.ckpt, "cuda", "float32").train()
    trunk = [p for n, p in model.named_parameters() if not n.startswith("output_head")]
    print(f"[gradcos] trunk parameters: {sum(p.numel() for p in trunk):,}", flush=True)
    dm = ColliderMLRegrDataModule(preprocessed_dir=args.data_dir, batch_size=args.batch_size,
                                  num_workers=args.workers, pin_memory=False, seed_residual_features=True)
    dm.setup("test")

    per_batch = []
    for inputs, targets in dm.test_dataloader():
        inputs = {k: v.cuda() for k, v in inputs.items()}
        targets = {k: v.cuda() for k, v in targets.items()}
        losses = model.loss_module(model(inputs)["pred"], targets)
        grads = []
        for i, p in enumerate(PARAMS):      # one forward, five backwards through the retained graph
            gs = torch.autograd.grad(losses[p], trunk, retain_graph=i < len(PARAMS) - 1, allow_unused=True)
            g = torch.cat([(q if q is not None else torch.zeros_like(t)).flatten() for q, t in zip(gs, trunk)])
            grads.append(g / (g.norm() + 1e-12))
        G = torch.stack(grads)
        per_batch.append((G @ G.T).cpu().numpy())
        if len(per_batch) % 50 == 0:
            print(f"[gradcos] batch {len(per_batch)}/{args.n_batches}", flush=True)
        if len(per_batch) >= args.n_batches:
            break

    C = np.stack(per_batch)
    nb = C.shape[0]
    mean, std = C.mean(axis=0), (C.std(axis=0, ddof=1) if nb > 1 else np.zeros(C.shape[1:]))
    sem = std / np.sqrt(nb)
    np.savez(args.out_dir / "grad_cosines.npz", mean=mean, std=std, sem=sem, per_batch=C,
             n_batches=nb, batch_size=args.batch_size, params=PARAMS)
    with open(args.out_dir / "summary.txt", "w") as f:
        f.write(f"{nb} batches x {args.batch_size} tracks\n\nmean cosine +- sem\n")
        f.write("        " + "".join(f"{p:>16s}" for p in PARAMS) + "\n")
        for i, p in enumerate(PARAMS):
            f.write(f"{p:>7s} " + "".join(f"  {mean[i, j]:+.3f} +- {sem[i, j]:.3f}" for j in range(5)) + "\n")
    print((args.out_dir / "summary.txt").read_text(), flush=True)


if __name__ == "__main__":
    main()
