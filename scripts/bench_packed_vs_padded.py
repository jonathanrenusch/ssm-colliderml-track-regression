"""SSM-CLS packed-vs-padded micro-benchmark.

Stand-alone forward+backward benchmark of the publication SSM-CLS encoder under
the packed and padded data paths. Uses synthetic data with the production
track-length distribution (mean ~12.5, max=20, min=6) so we measure the
encoder/collate cost without hitting the I/O subsystem.

Run from the publication repo root:

    cd /shared/tracking/ssm-colliderml-track-regression
    CUDA_VISIBLE_DEVICES=MIG-57d25c0d-b961-55c1-96bc-b532b1f86aea \\
        pixi run python scripts/bench_packed_vs_padded.py \\
            --batch-size 2048 --num-layers 10 --warmup 50 --iters 500

Reports for each path: collate ms, fwd ms, bwd ms, total step ms, real-hit
tokens/sec, padded-tokens/sec, peak memory MiB. Also writes a CSV row per path
to stdout for easy aggregation across runs.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as stats
import sys
import time
from contextlib import nullcontext

import numpy as np
import torch
import yaml

# Make the in-repo package importable regardless of where this is launched from.
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.normpath(os.path.join(HERE, "..", "src"))
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from track_regression.data import collate_tracks, collate_tracks_packed  # noqa: E402
from track_regression.mamba_cls import BidirectionalMambaCLSEncoder  # noqa: E402
from track_regression.losses import TrackParameterLoss  # noqa: E402
from track_regression.model import TrackParameterRegressor  # noqa: E402


def synth_batch(batch_size: int, rng: np.random.Generator, feat_dim: int = 12) -> list[dict]:
    """Produce a list of (variable-length) synthetic tracks matching the
    production length distribution: mean ~12.5, std ~4, clipped to [6, 20].
    """
    lengths = rng.normal(12.5, 4.0, size=batch_size)
    lengths = np.clip(np.round(lengths).astype(np.int32), 6, 20)
    out: list[dict] = []
    for L in lengths:
        feats = rng.standard_normal((int(L), feat_dim)).astype(np.float32)
        # column 6 is `s` (sort key surrogate); collate doesn't sort but the
        # encoder packed path ignores it. We fill in a monotone increasing
        # series so the padded path's argsort is stable.
        feats[:, 6] = np.arange(int(L), dtype=np.float32)
        hit_time = np.arange(int(L), dtype=np.float32)
        out.append({
            "hit_features": feats,
            "hit_s": feats[:, 6].copy(),
            "hit_time": hit_time,
            "targets": rng.standard_normal(5).astype(np.float32),
            "length": int(L),
        })
    return out


def build_model(num_layers: int, dim: int, d_state: int) -> TrackParameterRegressor:
    encoder = BidirectionalMambaCLSEncoder(
        num_layers=num_layers, dim=dim, d_state=d_state, d_conv=4, expand=2,
        headdim=32, ngroups=1, chunk_size=16, norm="RMSNorm",
        dropout=0.0, residual_depth_init=True,
    )
    loss = TrackParameterLoss(
        parameter_order=["d0", "z0", "phi", "theta", "qop"],
        config={
            "d0": {"type": "quantile", "weight": 0.1, "norm_min": -2.5, "norm_max": 2.5,
                   "quantiles": [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]},
            "z0": {"type": "quantile", "weight": 1.0, "norm_min": -200.0, "norm_max": 200.0,
                   "quantiles": [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]},
            "phi": {"type": "circular", "weight": 0.1, "beta": 0.01},
            "theta": {"type": "quantile_eta", "weight": 1.0, "norm_min": -3.0, "norm_max": 3.0,
                      "quantiles": [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]},
            "qop": {"type": "quantile", "weight": 1.0, "norm_min": -2.0, "norm_max": 2.0,
                    "quantiles": [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]},
        },
    )
    return TrackParameterRegressor(
        input_dim=12, dim=dim, encoder=encoder, loss_module=loss, pool="ssm_cls",
        state_head_output_dim=128, state_head_hidden_layers=[192],
        state_head_dropout=0.0, state_head_activation="SiLU",
        output_head_hidden_layers=[256], output_head_dropout=0.0,
        output_head_activation="SiLU", input_net_hidden_layers=[192],
        input_net_dropout=0.0, input_net_activation="SiLU",
        sort_field="s",
        fourier_scales=[-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5], fourier_base=2,
        norm_min=[-1031.0, -1031.0, -3026.0, 31.0, -3.1416, 0.027, 31.0, 16.0, 2.0, 1.0, 0.0, -4.3],
        norm_max=[1031.0, 1031.0, 3026.0, 1032.0, 3.1416, 3.114, 3185.0, 30.0, 16.0, 3360.0, 8.0, 4.3],
        encoder_autocast_dtype="float32", output_head_init_scale=0.01,
    )


def time_one_path(
    *, model: TrackParameterRegressor, packed: bool, batch_size: int,
    warmup: int, iters: int, seed: int = 0,
) -> dict:
    rng = np.random.default_rng(seed)
    device = next(model.parameters()).device
    collate = collate_tracks_packed if packed else collate_tracks

    # ---- Warmup ----
    for _ in range(warmup):
        batch = synth_batch(batch_size, rng)
        inputs, targets = collate(batch)
        inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}
        targets = {k: v.to(device, non_blocking=True) for k, v in targets.items()}
        out = model(inputs)
        loss = model.compute_loss(out, targets, valid_mask=targets["track_valid"])["total"]
        loss.backward()
        for p in model.parameters():
            p.grad = None
    torch.cuda.synchronize()

    collate_ms: list[float] = []
    fwd_ms: list[float] = []
    bwd_ms: list[float] = []
    step_ms: list[float] = []
    real_hits: list[int] = []
    padded_hits: list[int] = []

    torch.cuda.reset_peak_memory_stats(device)

    for _ in range(iters):
        batch = synth_batch(batch_size, rng)

        t0 = time.perf_counter()
        inputs, targets = collate(batch)
        if packed:
            real = int(inputs["hit_features"].shape[1])
            padded = real
        else:
            real = int(sum(b["length"] for b in batch))
            padded = int(inputs["hit_features"].shape[0] * inputs["hit_features"].shape[1])
        inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}
        targets = {k: v.to(device, non_blocking=True) for k, v in targets.items()}
        torch.cuda.synchronize()
        t1 = time.perf_counter()

        out = model(inputs)
        torch.cuda.synchronize()
        t2 = time.perf_counter()

        loss = model.compute_loss(out, targets, valid_mask=targets["track_valid"])["total"]
        loss.backward()
        torch.cuda.synchronize()
        t3 = time.perf_counter()

        for p in model.parameters():
            p.grad = None

        collate_ms.append((t1 - t0) * 1e3)
        fwd_ms.append((t2 - t1) * 1e3)
        bwd_ms.append((t3 - t2) * 1e3)
        step_ms.append((t3 - t0) * 1e3)
        real_hits.append(real)
        padded_hits.append(padded)

    peak = torch.cuda.max_memory_allocated(device) / 2**20

    def _quants(xs):
        return {"med": stats.median(xs),
                "p10": float(np.percentile(xs, 10)),
                "p90": float(np.percentile(xs, 90))}

    return {
        "path": "packed" if packed else "padded",
        "batch_size": batch_size,
        "iters": iters,
        "collate_ms": _quants(collate_ms),
        "fwd_ms": _quants(fwd_ms),
        "bwd_ms": _quants(bwd_ms),
        "step_ms": _quants(step_ms),
        "real_hits_per_batch_med": int(stats.median(real_hits)),
        "padded_hits_per_batch_med": int(stats.median(padded_hits)),
        "samples_per_sec_med": batch_size / (stats.median(step_ms) / 1e3),
        "real_tokens_per_sec_med": stats.median(real_hits) / (stats.median(step_ms) / 1e3),
        "peak_alloc_MiB": peak,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--num-layers", type=int, default=10)
    ap.add_argument("--dim", type=int, default=192)
    ap.add_argument("--d-state", type=int, default=32)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--iters", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-json", type=str, default="")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True

    if not torch.cuda.is_available():
        raise RuntimeError("Need a CUDA device for this benchmark.")
    device = torch.device("cuda")
    print(f"# device: {torch.cuda.get_device_name(device)}  capability={torch.cuda.get_device_capability(device)}")

    results = []
    for packed in (False, True):
        # Fresh model per path so optimizer-state / autograd graph doesn't bias.
        model = build_model(args.num_layers, args.dim, args.d_state).to(device)
        # No optimizer — we're measuring the encoder + heads only. Backward
        # without optimizer.step() is the standard fwd+bwd compute cost.
        r = time_one_path(model=model, packed=packed,
                          batch_size=args.batch_size, warmup=args.warmup,
                          iters=args.iters, seed=args.seed)
        results.append(r)
        del model
        torch.cuda.empty_cache()

    # Summary table
    print("=" * 110)
    print(f"  {'path':<8} {'step_ms (med, p10/p90)':<26} {'fwd_ms':<14} {'bwd_ms':<14} {'collate_ms':<12} {'samp/s':<10} {'real_tok/s':<12} {'peak_MiB':<10}")
    print("-" * 110)
    for r in results:
        print(
            f"  {r['path']:<8} "
            f"{r['step_ms']['med']:6.1f}  ({r['step_ms']['p10']:5.1f}/{r['step_ms']['p90']:5.1f})  "
            f"{r['fwd_ms']['med']:6.1f}        "
            f"{r['bwd_ms']['med']:6.1f}        "
            f"{r['collate_ms']['med']:6.2f}    "
            f"{r['samples_per_sec_med']:8.0f}  "
            f"{r['real_tokens_per_sec_med']:10.0f}  "
            f"{r['peak_alloc_MiB']:8.0f}"
        )
    if len(results) == 2:
        ratio = results[0]["step_ms"]["med"] / results[1]["step_ms"]["med"]
        delta_pct = (1.0 - results[1]["step_ms"]["med"] / results[0]["step_ms"]["med"]) * 100
        print(f"\n  packed speedup vs padded: {ratio:.3f}×  ({delta_pct:+.1f}% step time)")
        pad_waste = 1.0 - results[1]["real_hits_per_batch_med"] / results[0]["padded_hits_per_batch_med"]
        print(f"  padded path wastes {pad_waste*100:.1f}% of tokens on padding "
              f"(real={results[1]['real_hits_per_batch_med']} / padded={results[0]['padded_hits_per_batch_med']})")

    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump({"args": vars(args), "results": results}, f, indent=2)
        print(f"\n  wrote {args.out_json}")


if __name__ == "__main__":
    main()
