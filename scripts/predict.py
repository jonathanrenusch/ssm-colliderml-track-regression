#!/usr/bin/env python3
"""Run a trained model over the test split of one store and write its predictions.

    python scripts/predict.py --config configs/minGRU_stage2_finetune.yaml \
        --ckpt runs/minGRU_stage2_finetune/version_0/checkpoints/last.ckpt \
        --data-dir data/eval/single_muon_10GeV --out results/minGRU/preds/single_muon_10GeV.h5

Defaults are the paper's inference settings: fused kernels, TF32 GEMMs, the
encoder under float16 autocast, the analytic seed computed on the GPU in
float64 inside the forward pass.  Output (HDF5): ``preds/<param>`` and
``targets/<param>`` (N,) and ``quantiles/<param>`` (N, 7) in the order of the
store's test split.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import torch

from track_regression.data import ColliderMLRegrDataModule
from track_regression.inference import load_model

PARAMS = ("d0", "z0", "phi", "theta", "qop")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data-dir", required=True, help="store root with a test/ split")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--batch-size", type=int, default=10000)
    ap.add_argument("--encoder-dtype", default="float16", choices=["float32", "float16", "bfloat16"])
    ap.add_argument("--matmul-precision", default="high", choices=["high", "highest"])
    args = ap.parse_args()

    torch.set_float32_matmul_precision(args.matmul_precision)
    model = load_model(args.config, args.ckpt, "cuda", args.encoder_dtype)
    dm = ColliderMLRegrDataModule(preprocessed_dir=args.data_dir, batch_size=args.batch_size,
                                  num_workers=0, pin_memory=False)
    dm.setup("test")
    out = {g: {p: [] for p in PARAMS} for g in ("preds", "targets", "quantiles")}
    with torch.inference_mode():
        for inputs, targets in dm.test_dataloader():
            inputs = {k: v.cuda(non_blocking=True) for k, v in inputs.items()}
            o = model(inputs)
            anchors = {f"seed_{p}": o["seed"][:, i] for i, p in enumerate(PARAMS)}
            preds = model.loss_module.predict_physical(o["pred"], anchors)
            quant = model.loss_module.predict_quantiles(o["pred"])
            for p in PARAMS:
                out["preds"][p].append(preds[p].float().cpu().numpy())
                out["targets"][p].append(targets[p].numpy())
                out["quantiles"][p].append(quant[p].float().cpu().numpy())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.out, "w") as f:
        for g, d in out.items():
            for p, v in d.items():
                ds = f.create_dataset(f"{g}/{p}", data=np.concatenate(v), compression="gzip", compression_opts=1)
                if g == "quantiles":
                    ds.attrs["levels"] = model.loss_module.losses[p].quantiles.cpu().numpy()
    print(f"[predict] {len(np.concatenate(out['preds']['d0'])):,} tracks -> {args.out}")


if __name__ == "__main__":
    main()
