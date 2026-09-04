#!/usr/bin/env python3
"""Cross-device throughput figure for the paper (no-conv 2L, deployment path).

Parses the collaborator's RTX 5000 Ada bench logs and this repo's H100 sweep
(identical tool: scripts/bench_infer_flat.py, TF32 + kernel switches + GPU
auto-seed, same ttbar_bench store) into one two-panel figure:
  (a) inference throughput vs batch size, with the CPU Kalman-filter reference,
  (b) peak GPU power vs batch size (from the nvidia-smi 1 Hz poll CSVs).
Also writes a plateau-values .txt for the paper table.

Usage: plot_throughput_cross_device.py <ada_log_dir> <h100_log_dir> <out_dir>
"""
from __future__ import annotations

import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

MODEL = "R2Lnoconv_2L"
CPU_REF = 30_000            # ACTS KF fit, 64-core Threadripper 3990X (colleague)
LOG_RE = re.compile(rf"bench_model_{MODEL}_(?P<batch>\d+)\.log$")
CSV_RE = re.compile(rf"gpu\d+_model_{MODEL}_(?P<batch>\d+)_metrics\.csv$")
GPU_RE = re.compile(r"GPU\s*:\s*(?P<gpu>.+)")
THROUGHPUT_RE = re.compile(r"throughput\s*:\s*([\d,]+)\s*tracks/s")
VRAM_RE = re.compile(r"peak VRAM\s*:\s*([\d.]+)\s*GiB")


def parse(log_dir: Path):
    thr, vram, power = {}, {}, {}
    gpu_name = None
    for p in sorted(log_dir.glob("*.log")):
        m = LOG_RE.search(p.name)
        if not m:
            continue
        text = p.read_text()
        g, t = GPU_RE.search(text), THROUGHPUT_RE.search(text)
        if not (g and t):
            print(f"  [skip incomplete] {p.name}")
            continue
        gpu_name = g.group("gpu").strip()
        bs = int(m.group("batch"))
        thr[bs] = int(t.group(1).replace(",", ""))
        v = VRAM_RE.search(text)
        if v:
            vram[bs] = float(v.group(1))
    for p in sorted(log_dir.glob("gpu*_metrics.csv")):
        m = CSV_RE.search(p.name)
        if not m:
            continue
        vals = []
        with p.open(newline="") as f:
            for row in csv.reader(f):
                if len(row) == 2:
                    try:
                        vals.append(float(row[1]))
                    except ValueError:
                        pass
        if vals:
            power[int(m.group("batch"))] = max(vals)
    return gpu_name, thr, vram, power


def main():
    ada_dir, h100_dir, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    out.mkdir(parents=True, exist_ok=True)
    devices = []
    for d, color, short in ((h100_dir, "C0", "H100 NVL"), (ada_dir, "C1", "RTX 5000 Ada")):
        name, thr, vram, power = parse(d)
        if not thr:
            raise SystemExit(f"no complete {MODEL} logs in {d}")
        devices.append((short, name, color, thr, vram, power))

    fig, ax_t = plt.subplots(figsize=(5.4, 3.5))
    for short, name, color, thr, vram, power in devices:
        bs = sorted(thr)
        ax_t.plot(bs, [thr[b] / 1e6 for b in bs], "o-", ms=3.5, lw=1.6,
                  color=color, label=f"{short}: {max(thr.values())/1e6:.2f} M tracks/s peak")
    ax_t.axhline(CPU_REF / 1e6, color="0.35", lw=1.2, ls="--")
    ax_t.text(300, CPU_REF / 1e6 * 1.35, "ACTS KF fit, 64-core CPU (30 k tracks/s)",
              fontsize=7.5, color="0.35")
    ax_t.set_xscale("log", base=2); ax_t.set_yscale("log")
    ax_t.set_xlabel("tracks per batch"); ax_t.set_ylabel("throughput [$10^6$ tracks/s]")
    ax_t.legend(fontsize=8, loc="lower right")
    ax_t.grid(True, which="both", ls=":", alpha=0.35)
    fig.tight_layout()
    stem = out / "throughput_noconv_h100_vs_ada"
    fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)

    lines = [f"{MODEL} deployment path (TF32 + kernel switches, GPU seed fp64), ttbar_bench store"]
    for short, name, color, thr, vram, power in devices:
        b_best = max(thr, key=thr.get)
        lines.append(f"{short} ({name}): peak {thr[b_best]:,} tracks/s at bs {b_best:,}"
                     f"; VRAM {vram.get(b_best, float('nan')):.1f} GiB"
                     f"; peak power {max(power.values()) if power else float('nan'):.0f} W")
    (out / "throughput_summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"[plot] {stem}.pdf")


if __name__ == "__main__":
    main()
