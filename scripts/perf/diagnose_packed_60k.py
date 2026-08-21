"""Diagnose the ~60K packed-token ceiling of the stock Mamba2 kernel.

Binary-searches the token ceiling of the stock model (packed) between
``--min-tokens`` and ``--max-tokens``: coarse ascending scan until the first
failure, then bisection of the last-good/first-fail boundary. Every probe runs
in a **fresh subprocess** (this script re-invoked with ``--probe-tokens``) so
an illegal-address failure cannot poison the CUDA context of later probes.

On failure it captures the full traceback verbatim, the exact exception text,
batch/token counts, and classifies the failure (OOM / launch error /
illegal-address / other). With ``--sanitize`` the first failing size is
re-run under ``compute-sanitizer`` and its output embedded in the report.

Output: a markdown report (default
``docs/perf/results/night1/packed60k_diagnosis.md``) with the stack trace
verbatim.

Usage::

    pixi run -e default python scripts/perf/diagnose_packed_60k.py \\
        --config src/track_regression/config/experimental/scaling/finetune_ssm_cls_4L_muon.yaml \\
        --ckpt logs/.../epoch=049-val_total=0.00125.ckpt --sanitize
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("TRITON_CACHE_DIR", "/tmp/triton_cache")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import common  # noqa: E402

DEFAULT_BIN_DIR = "/usr/local/cuda-13.1/bin"
PROBE_OK_MARK = "PROBE_OK"


def classify_failure(text: str) -> str:
    low = text.lower()
    if "out of memory" in low or "outofmemoryerror" in low:
        return "OOM"
    if "illegal memory access" in low or "misaligned address" in low:
        return "illegal-address"
    if ("invalid configuration argument" in low or "invalid argument" in low
            or "too many blocks" in low or "grid" in low or "launch" in low):
        return "launch-error"
    return "other"


# ---------------------------------------------------------------------------
# Probe (child-process mode)
# ---------------------------------------------------------------------------

def run_probe(args) -> None:
    """Build model + synthetic packed batch at ~--probe-tokens, forward a few iters."""
    common.ensure_src_on_path()
    import torch

    from bench_variant import (
        batch_stats,
        build_datamodule,
        build_model,
        resize_packed_batch,
        to_cuda,
    )

    common.pin_precision_flags()
    cfg = common.load_config(args.config)
    model = build_model(cfg, args.ckpt)

    dm = build_datamodule(cfg, batch_size=args.batch_size, num_workers=0)
    inputs, _ = next(iter(dm.test_dataloader()))
    if "cu_seqlens" not in inputs:
        print("PROBE_ERROR: captured batch is not packed (no cu_seqlens)", flush=True)
        sys.exit(2)
    base = batch_stats(inputs)
    len_mean = base["batch_tokens"] / base["batch_tracks"]
    target_tracks = max(1, round(args.probe_tokens / len_mean))
    inputs = resize_packed_batch(inputs, target_tracks)
    stats = batch_stats(inputs)
    print(f"PROBE_STATS tracks={stats['batch_tracks']} tokens={stats['batch_tokens']} "
          f"len_mean={stats['len_mean']}", flush=True)

    gpu_inputs = to_cuda(inputs)
    with torch.inference_mode():
        model(gpu_inputs)  # warmup / triton compile — outside the timing
        torch.cuda.synchronize()
    t0 = time.monotonic()
    with torch.inference_mode():
        for _ in range(args.iters):
            model(gpu_inputs)
        torch.cuda.synchronize()
    dt_ms = (time.monotonic() - t0) * 1e3 / args.iters
    print(f"{PROBE_OK_MARK} tokens={stats['batch_tokens']} tracks={stats['batch_tracks']} "
          f"t_iter_ms={dt_ms:.2f}", flush=True)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def probe_subprocess(args, tokens: int, sanitize: bool = False) -> dict:
    cmd = [
        sys.executable, str(Path(__file__).resolve()),
        "--config", str(args.config), "--ckpt", str(args.ckpt),
        "--batch-size", str(args.batch_size), "--iters", str(args.iters),
        "--probe-tokens", str(tokens),
    ]
    if sanitize:
        san = Path(args.bin_dir) / "compute-sanitizer"
        cmd = [str(san), "--tool", "memcheck"] + cmd
    print(f"[diag] probe {tokens} tokens{' (compute-sanitizer)' if sanitize else ''}",
          flush=True)
    t0 = time.monotonic()
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=args.probe_timeout_s * (4 if sanitize else 1),
            cwd=common.REPO_ROOT,
        )
        out = proc.stdout + "\n" + proc.stderr
        ok = proc.returncode == 0 and PROBE_OK_MARK in proc.stdout
        rc = proc.returncode
    except subprocess.TimeoutExpired as e:
        out = f"(probe timed out after {e.timeout}s)\n" \
              f"{(e.stdout or '')}\n{(e.stderr or '')}"
        ok, rc = False, -1
    stats_line = next((ln for ln in out.splitlines() if ln.startswith("PROBE_STATS")), "")
    t_line = next((ln for ln in out.splitlines() if ln.startswith(PROBE_OK_MARK)), "")
    result = {
        "tokens_requested": tokens,
        "ok": ok,
        "rc": rc,
        "wall_s": round(time.monotonic() - t0, 1),
        "stats": stats_line,
        "ok_line": t_line,
        "output": out,
        "classification": None if ok else classify_failure(out),
    }
    print(f"[diag]   -> {'OK' if ok else 'FAIL (' + str(result['classification']) + ')'}",
          flush=True)
    return result


def extract_traceback(output: str) -> str:
    lines = output.splitlines()
    starts = [i for i, ln in enumerate(lines) if ln.startswith("Traceback (")]
    if not starts:
        return output[-6000:]
    return "\n".join(lines[starts[0]:])


def write_report(args, curve: list[dict], last_good: int | None,
                 first_fail_result: dict | None, sanitizer_result: dict | None) -> None:
    env = common.env_fingerprint()
    lines = [
        "# Packed ~60K-token ceiling diagnosis",
        "",
        f"- generated: {common.utc_ts()}",
        f"- config: `{args.config}`",
        f"- ckpt: `{args.ckpt}`",
        f"- env: `{env}`",
        f"- probe range: {args.min_tokens}..{args.max_tokens} tokens, "
        f"step {args.step_tokens}, {args.iters} iters/probe",
        "",
        "## Probe curve",
        "",
        "| tokens (req) | status | classification | detail |",
        "|---:|---|---|---|",
    ]
    for r in curve:
        detail = r["ok_line"] or r["stats"] or ""
        lines.append(
            f"| {r['tokens_requested']} | {'ok' if r['ok'] else 'FAIL'} "
            f"| {r['classification'] or ''} | {detail} |"
        )
    lines += [
        "",
        f"**Last good:** {last_good if last_good is not None else 'none'} tokens (requested)  ",
        f"**First fail:** "
        f"{first_fail_result['tokens_requested'] if first_fail_result else 'none within range'} tokens",
        "",
    ]
    if first_fail_result:
        lines += [
            f"## First failure ({first_fail_result['tokens_requested']} tokens, "
            f"classified: {first_fail_result['classification']}, rc={first_fail_result['rc']})",
            "",
            f"{first_fail_result['stats']}",
            "",
            "### Traceback (verbatim)",
            "",
            "```",
            extract_traceback(first_fail_result["output"]),
            "```",
            "",
        ]
    if sanitizer_result:
        lines += [
            "## compute-sanitizer (memcheck) on the failing size",
            "",
            "```",
            sanitizer_result["output"][-20000:],
            "```",
            "",
        ]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines))
    print(f"[diag] report written: {args.report}", flush=True)


def orchestrate(args) -> None:
    curve: list[dict] = []
    last_good: int | None = None
    first_fail: dict | None = None

    tokens = args.min_tokens
    while tokens <= args.max_tokens:
        r = probe_subprocess(args, tokens)
        curve.append(r)
        if r["ok"]:
            last_good = tokens
            tokens += args.step_tokens
        else:
            first_fail = r
            break

    # Bisect the boundary down to ~512 tokens.
    if first_fail is not None and last_good is not None:
        lo, hi = last_good, first_fail["tokens_requested"]
        while hi - lo > 512:
            mid = (lo + hi) // 2
            r = probe_subprocess(args, mid)
            curve.append(r)
            if r["ok"]:
                lo, last_good = mid, mid
            else:
                hi, first_fail = mid, r
        print(f"[diag] boundary: last_good={last_good} first_fail={hi} tokens", flush=True)
    elif first_fail is None:
        print(f"[diag] no failure up to {args.max_tokens} tokens", flush=True)

    sanitizer_result = None
    if args.sanitize and first_fail is not None:
        sanitizer_result = probe_subprocess(
            args, first_fail["tokens_requested"], sanitize=True,
        )

    write_report(args, curve, last_good, first_fail, sanitizer_result)
    sys.exit(0)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--ckpt", required=True, type=Path)
    ap.add_argument("--min-tokens", type=int, default=30_000)
    ap.add_argument("--max-tokens", type=int, default=120_000)
    ap.add_argument("--step-tokens", type=int, default=10_000)
    ap.add_argument("--batch-size", type=int, default=2048,
                    help="tracks in the initially captured (then replicated) batch")
    ap.add_argument("--iters", type=int, default=5, help="forwards per probe")
    ap.add_argument("--probe-timeout-s", type=int, default=900)
    ap.add_argument("--sanitize", action="store_true",
                    help="re-run the first failing size under compute-sanitizer")
    ap.add_argument("--bin-dir", default=DEFAULT_BIN_DIR,
                    help="dir containing compute-sanitizer")
    ap.add_argument("--report", type=Path,
                    default=common.REPO_ROOT / "docs/perf/results/night1/packed60k_diagnosis.md")
    ap.add_argument("--probe-tokens", type=int, default=0,
                    help="(internal) run a single probe at this token count")
    args = ap.parse_args()

    if args.probe_tokens > 0:
        try:
            run_probe(args)
        except Exception:
            traceback.print_exc()
            sys.exit(1)
    else:
        orchestrate(args)


if __name__ == "__main__":
    main()
