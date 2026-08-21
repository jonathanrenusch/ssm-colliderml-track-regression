"""Physics-drift gate: clipped-RMS drift of a variant vs the V0 reference.

Runs the model's predict path (``model(inputs)`` →
``loss_module.predict_physical`` — mirrors ``TrackRegressionWrapper.predict_step``)
over a fixed ~131k-track subset (one shard's worth of the test split), computes
per-parameter residuals via ``track_regression.eval_utils.compute_residuals``
and the ATLAS-style clipped RMS via ``iterative_rms_convergence`` for
d0/z0/phi/theta/qop (+ eta, informational only). Against ``--ref`` it prints a
fixed-width drift table and gates at |drift| <= 1.0 % on all five parameters.

Exit codes: 0 = PASS (or no --ref given), 1 = FAIL, 2 = error,
3 = variant kernel module not available yet.

If ``--subset`` does not exist it is built first: datamodule test split
(data-path override applied), ``num_workers=0``, first ``--subset-tracks``
tracks in loader order (deterministic), saved as a list of collated CPU
batches via ``torch.save``.

Usage::

    pixi run -e default python scripts/perf/physics_drift.py \\
        --config src/track_regression/config/experimental/scaling/finetune_ssm_cls_4L_muon.yaml \\
        --ckpt logs/.../epoch=049-val_total=0.00125.ckpt \\
        --variant v0 --out docs/perf/results/night1/physics/v0.npz

    # later, gating a variant:
    ... --variant v3 --ref docs/perf/results/night1/physics/v0.npz \\
        --out docs/perf/results/night1/physics/v3.npz
"""
from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path

os.environ.setdefault("TRITON_CACHE_DIR", "/tmp/triton_cache")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import common  # noqa: E402

common.ensure_src_on_path()

import numpy as np  # noqa: E402
import torch  # noqa: E402

from bench_variant import apply_variant_or_exit, build_datamodule, build_model  # noqa: E402

PARAMS = ["d0", "z0", "phi", "theta", "qop"]
DRIFT_LIMIT_PCT = 1.0


# ---------------------------------------------------------------------------
# Subset
# ---------------------------------------------------------------------------

def build_subset(cfg: dict, subset_path: Path, subset_tracks: int, batch_size: int) -> None:
    """Capture the first ~one-shard's-worth of test tracks as CPU batches."""
    print(f"[physics] building subset ({subset_tracks} tracks) -> {subset_path}", flush=True)
    dm = build_datamodule(cfg, batch_size=batch_size, num_workers=0)
    loader = dm.test_dataloader()
    batches: list = []
    n = 0
    for inputs, targets in loader:
        batches.append((
            {k: v.cpu() for k, v in inputs.items()},
            {k: v.cpu() for k, v in targets.items()},
        ))
        n += int(targets["d0"].shape[0])
        print(f"[physics] subset {n}/{subset_tracks} tracks", flush=True)
        if n >= subset_tracks:
            break
    if n < subset_tracks:
        print(f"[physics] WARNING: test split exhausted at {n} tracks", flush=True)
    subset_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(batches, subset_path)
    print(f"[physics] subset saved: {len(batches)} batches, {n} tracks", flush=True)


# ---------------------------------------------------------------------------
# Predict + residuals
# ---------------------------------------------------------------------------

def predict_subset(model: torch.nn.Module, batches: list) -> tuple[dict, dict]:
    """Mirror TrackRegressionWrapper.predict_step (model.py:816) over the subset."""
    preds_acc: dict[str, list[np.ndarray]] = {p: [] for p in PARAMS}
    targ_acc: dict[str, list[np.ndarray]] = {p: [] for p in PARAMS}
    with torch.inference_mode():
        for i, (inputs, targets) in enumerate(batches):
            inputs = {k: v.cuda(non_blocking=True) for k, v in inputs.items()}
            targets = {k: v.cuda(non_blocking=True) for k, v in targets.items()}
            outputs = model(inputs)
            preds = model.loss_module.predict_physical(outputs["pred"], targets)
            for p in PARAMS:
                preds_acc[p].append(preds[p].float().cpu().numpy().reshape(-1))
                targ_acc[p].append(targets[p].float().cpu().numpy().reshape(-1))
            if (i + 1) % 10 == 0:
                print(f"[physics] predict batch {i + 1}/{len(batches)}", flush=True)
    torch.cuda.synchronize()
    return (
        {p: np.concatenate(v) for p, v in preds_acc.items()},
        {p: np.concatenate(v) for p, v in targ_acc.items()},
    )


def clipped_rms(preds: dict, targets: dict) -> dict[str, float]:
    """Per-param iterative clipped RMS (the headline physics metric) + eta info."""
    from track_regression.eval_utils import compute_residuals, iterative_rms_convergence

    residuals = compute_residuals({"preds": preds, "targets": targets})
    rms = {p: iterative_rms_convergence(residuals[p])["rms"] for p in PARAMS}
    # Informational: eta residual derived from theta (not part of the gate).
    def eta(theta: np.ndarray) -> np.ndarray:
        return -np.log(np.tan(np.clip(theta, 1e-8, np.pi - 1e-8) / 2.0))

    rms["eta"] = iterative_rms_convergence(eta(preds["theta"]) - eta(targets["theta"]))["rms"]
    return rms


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--ckpt", required=True, type=Path)
    ap.add_argument("--variant", required=True)
    ap.add_argument("--subset", type=Path,
                    default=common.REPO_ROOT / "docs/perf/results/physics_subset.pt")
    ap.add_argument("--subset-tracks", type=int, default=131_072,
                    help="~one shard's worth of tracks")
    ap.add_argument("--batch-size", type=int, default=16384,
                    help="tracks per batch when building/predicting the subset")
    ap.add_argument("--ref", type=Path, default=None,
                    help="reference npz (v0) — enables the drift gate")
    ap.add_argument("--out", type=Path, required=True, help="output .npz (preds+targets)")
    ap.add_argument("--gate-csv", type=Path,
                    default=common.REPO_ROOT / "docs/perf/results/physics_gate.csv")
    ap.add_argument("--job-id", default="physics")
    ap.add_argument("--build-subset-only", action="store_true",
                    help="build/refresh the subset file and exit (no model run)")
    args = ap.parse_args()

    try:
        precision_flags = common.pin_precision_flags()
        env = common.env_fingerprint()
        print(f"[physics] env: {env}", flush=True)
        print(f"[physics] precision: {precision_flags}", flush=True)

        cfg = common.load_config(args.config)

        if not args.subset.exists():
            build_subset(cfg, args.subset, args.subset_tracks, args.batch_size)
        elif args.build_subset_only:
            print(f"[physics] subset already exists: {args.subset}", flush=True)
        if args.build_subset_only:
            sys.exit(0)

        model = build_model(cfg, args.ckpt)
        model = apply_variant_or_exit(model, args.variant)  # may sys.exit(3)

        batches = torch.load(args.subset, map_location="cpu", weights_only=False)
        print(f"[physics] subset loaded: {len(batches)} batches", flush=True)
        preds, targets = predict_subset(model, batches)
        n_tracks = len(preds["d0"])

        args.out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.out,
            **{f"preds_{p}": preds[p] for p in PARAMS},
            **{f"targets_{p}": targets[p] for p in PARAMS},
        )
        print(f"[physics] wrote {args.out} ({n_tracks} tracks)", flush=True)

        rms = clipped_rms(preds, targets)

        drift: dict[str, float] = {}
        passed: bool | None = None
        if args.ref is not None:
            ref = np.load(args.ref)
            ref_preds = {p: ref[f"preds_{p}"] for p in PARAMS}
            ref_targets = {p: ref[f"targets_{p}"] for p in PARAMS}
            rms_ref = clipped_rms(ref_preds, ref_targets)
            print()
            print(f"{'param':<8}{'rms_ref':>14}{'rms_var':>14}{'drift_%':>10}  gate")
            print("-" * 52)
            passed = True
            for p in PARAMS + ["eta"]:
                d = 100.0 * (rms[p] - rms_ref[p]) / rms_ref[p]
                drift[p] = d
                gated = p in PARAMS
                ok = abs(d) <= DRIFT_LIMIT_PCT
                if gated and not ok:
                    passed = False
                mark = ("PASS" if ok else "FAIL") if gated else "info"
                print(f"{p:<8}{rms_ref[p]:>14.6g}{rms[p]:>14.6g}{d:>+10.3f}  {mark}")
            print("-" * 52)
            print(f"[physics] GATE: {'PASS' if passed else 'FAIL'} "
                  f"(|drift| <= {DRIFT_LIMIT_PCT}% on {PARAMS})", flush=True)
        else:
            for p in PARAMS + ["eta"]:
                print(f"[physics] rms[{p}] = {rms[p]:.6g}")

        row = {
            "ts": common.utc_ts(),
            "job_id": args.job_id,
            "variant": args.variant,
            "n_tracks": n_tracks,
            **{f"rms_{p}": rms[p] for p in PARAMS + ["eta"]},
            **{f"drift_pct_{p}": drift.get(p) for p in PARAMS + ["eta"]},
            "ref": str(args.ref) if args.ref else "",
            "out": str(args.out),
            "gate": {True: "PASS", False: "FAIL", None: ""}[passed],
            "git_sha": env["git_sha"],
        }
        common.append_csv(args.gate_csv, row)

        sys.exit(0 if passed in (True, None) else 1)
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        sys.exit(2)


if __name__ == "__main__":
    main()
