#!/usr/bin/env python3
"""Inference-throughput figure: GPU tracks/s vs batch size, and the CPU
Kalman-filter reference vs thread count.

GPU curves are parsed from benchmark logs named ``bench_<tag>_<batchsize>.log``
(one per batch size, as written by the batch-size sweep), which contain the lines
    throughput            : 5,205,205 tracks/s ...
    peak VRAM             : 6.03 GiB
The GPU curves read against the batch-size axis on top; the CPU thread scan of
the ACTS Kalman filter (measured constants below) reads against the thread axis
at the bottom, with its ideal linear scaling dotted.

Writes <out_dir>/throughput_mingru_devices.{pdf,png} and throughput_summary.txt.

Usage: plot_throughput.py <h100_log_dir> <ada_log_dir|none> <out_dir> [--tag minGRU]
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# ACTS Kalman filter, AMD Threadripper 3990X: measured tracks/s per thread count
CPU_LABEL = "ACTS Kalman filter, AMD Threadripper 3990X"
CPU_THREADS = (1, 2, 4, 8, 16, 32, 64)
CPU_TRACKS_PER_S = (7_500, 14_400, 27_600, 53_500, 93_700, 141_300, 172_700)
THR_RE = re.compile(r"throughput\s*:\s*([\d,]+)\s*tracks/s")
VRAM_RE = re.compile(r"peak VRAM\s*:\s*([\d.]+)\s*GiB")


def parse(log_dir: Path, tag: str):
    """Throughput and peak VRAM per batch size from ``bench_<tag>_<bs>.log``."""
    thr, vram = {}, {}
    for p in log_dir.glob(f"bench_{tag}_*.log"):
        m = re.fullmatch(rf"bench_{re.escape(tag)}_(\d+)\.log", p.name)
        if not m:
            continue
        text = p.read_text()
        t = THR_RE.search(text)
        if not t:
            continue
        bs = int(m.group(1))
        thr[bs] = int(t.group(1).replace(",", ""))
        v = VRAM_RE.search(text)
        if v:
            vram[bs] = float(v.group(1))
    return thr, vram


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("h100_log_dir", type=Path)
    ap.add_argument("ada_log_dir", help="RTX 5000 Ada log directory, or 'none'")
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--tag", default="minGRU", help="model tag in the log names (default minGRU)")
    a = ap.parse_args()
    out = a.out_dir
    out.mkdir(parents=True, exist_ok=True)

    fig, ax_t = plt.subplots(figsize=(5.0, 3.3))
    lines = [f"inference throughput, model tag '{a.tag}'"]
    thr, vram = parse(a.h100_log_dir, a.tag)
    if not thr:
        raise SystemExit(f"no bench_{a.tag}_<bs>.log with a throughput line in {a.h100_log_dir}")
    bs = sorted(thr)
    ax_t.plot(bs, [thr[b] / 1e6 for b in bs], lw=1.6, color="C0", ls="-", marker="o",
              mfc="C0", ms=3.5,
              label=f"minGRU, H100 NVL: {max(thr.values())/1e6:.2f} M tracks/s")
    b = max(thr, key=thr.get)
    lines.append(f"H100 NVL      peak {thr[b]:,} tracks/s at bs {b:,}"
                 f"  ({vram.get(b, float('nan')):.1f} GiB)")

    ada = {} if a.ada_log_dir == "none" else parse(Path(a.ada_log_dir), a.tag)[0]
    if ada:
        bs = sorted(ada)
        ax_t.plot(bs, [ada[b] / 1e6 for b in bs], color="C0", ls="--", marker="s",
                  mfc="white", lw=1.6, ms=3.2,
                  label=f"minGRU, RTX 5000 Ada: {max(ada.values())/1e6:.2f} M tracks/s")
        b = max(ada, key=ada.get)
        lines.append(f"RTX 5000 Ada  peak {ada[b]:,} tracks/s at bs {b:,}")

    ax_t.set_xscale("log", base=2); ax_t.set_yscale("log")
    ax_t.set_xlabel("tracks per batch"); ax_t.set_ylabel("throughput [$10^6$ tracks/s]")
    ax_t.grid(True, which="both", ls=":", alpha=0.35)

    ax_c = ax_t.twiny()
    ax_c.set_xscale("log", base=2)
    ax_c.set_xlim(2 ** -0.55, 2 ** 6.55)
    ax_c.set_xticks(CPU_THREADS)
    ax_c.set_xticklabels([str(t) for t in CPU_THREADS])
    ax_c.set_xlabel("CPU threads")
    # Batch-size axis on top (next to the GPU curves), thread axis at the
    # bottom (next to the CPU curve).  twiny() resets the tick sides, so this
    # has to come after it.
    ax_t.xaxis.set_label_position("top")
    ax_t.tick_params(axis="x", which="both", top=True, labeltop=True,
                     bottom=False, labelbottom=False)
    ax_c.xaxis.set_label_position("bottom")
    ax_c.tick_params(axis="x", which="both", bottom=True, labelbottom=True,
                     top=False, labeltop=False)
    ideal = [CPU_TRACKS_PER_S[0] * t for t in CPU_THREADS]
    ax_c.plot(CPU_THREADS, [v / 1e6 for v in ideal], color="C3", ls=":", lw=1.2,
              alpha=0.7, label="ACTS Kalman filter, ideal linear scaling")
    ax_c.plot(CPU_THREADS, [v / 1e6 for v in CPU_TRACKS_PER_S], color="C3",
              ls="-", lw=1.6, marker="^", ms=3.5,
              label=f"ACTS Kalman filter, CPU: {CPU_TRACKS_PER_S[-1]/1e3:.0f} k tracks/s\n(AMD Threadripper 3990X)")

    # tight_layout first: the legend is anchored in axes coordinates below the panel
    fig.tight_layout()
    handles = ax_t.get_lines() + ax_c.get_lines()[::-1]
    ax_c.legend(handles, [h.get_label() for h in handles], fontsize=7.6,
                loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=2,
                framealpha=0.9, borderaxespad=0.0, columnspacing=1.2,
                handlelength=1.8)
    stem = out / "throughput_mingru_devices"
    fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(f"{stem}.png", dpi=110, bbox_inches="tight")
    plt.close(fig)

    lines.append(f"CPU ({CPU_LABEL}) thread scan [tracks/s]: "
                 + ", ".join(f"{t}t {v:,}" for t, v in zip(CPU_THREADS, CPU_TRACKS_PER_S)))
    (out / "throughput_summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"[plot] {stem}.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
