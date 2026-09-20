#!/usr/bin/env python3
"""Cross-device throughput figure for the ICLR_v2 (minGRU main line) draft.

Same design as scripts/plot_throughput_cross_device.py -- the GPU curves read
against a batch-size axis on top, the ACTS Kalman-filter CPU scan against a
thread axis at the bottom -- with the H100 series replaced by the minGRU
backbone at fp16 inference.

The RTX 5000 Ada series is still the Mamba-2 model: the collaborator has not
re-run the benchmark on the minGRU yet, so that curve is labelled as such and
must not be read as a minGRU number.

Usage: plot_throughput_v2.py <h100_v2_log_dir> <ada_log_dir> <out_dir>
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

CPU_THREADS = (1, 2, 4, 8, 16, 32, 64)
CPU_TRACKS_PER_S = (7_500, 14_400, 27_600, 53_500, 93_700, 141_300, 172_700)
THR_RE = re.compile(r"throughput\s*:\s*([\d,]+)\s*tracks/s")
VRAM_RE = re.compile(r"peak VRAM\s*:\s*([\d.]+)\s*GiB")

# (tag, legend label, style) -- H100 series, drawn in this order.
H100 = [
    ("minGRU_h192_fp16", "minGRU $h{=}192$, fp16",
     dict(color="C0", ls="-", marker="o", mfc="C0")),
    ("minGRU_FT_h194_fp16", "minGRU $h{=}194$, fp16",
     dict(color="C0", ls="-.", marker="D", mfc="white", ms=3.0)),
]


def parse(log_dir: Path, tag: str):
    thr, vram = {}, {}
    for p in log_dir.glob(f"bench_model_{tag}_*.log"):
        m = re.search(rf"bench_model_{re.escape(tag)}_(\d+)\.log$", p.name)
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
    h100_dir, ada_dir, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    out.mkdir(parents=True, exist_ok=True)

    fig, ax_t = plt.subplots(figsize=(5.0, 3.3))
    lines = ["ICLR_v2 throughput, deployment path (TF32 linears + kernel switches, "
             "on-GPU fp64 seed), ttbar_new_pt1 eval store"]
    series = []
    for tag, label, style in H100:
        thr, vram = parse(h100_dir, tag)
        if not thr:
            print(f"  [skip] no logs for {tag} in {h100_dir}")
            continue
        series.append((tag, label, thr, vram))
        bs = sorted(thr)
        style = {"ms": 3.5, **style}
        ax_t.plot(bs, [thr[b] / 1e6 for b in bs], lw=1.6, **style,
                  label=f"{label}, H100 NVL: {max(thr.values())/1e6:.2f} M tracks/s")
        b = max(thr, key=thr.get)
        lines.append(f"H100 NVL  {tag:24s} peak {thr[b]:,} tracks/s at bs {b:,}"
                     f"  ({vram.get(b, float('nan')):.1f} GiB)")

    # The Ada curve is the Mamba-2 model (collaborator's bench_logs_v2): kept
    # for the device comparison, explicitly labelled, to be re-measured.
    ada, _ = parse(ada_dir, "R2Lnoconv_2L")
    if ada:
        bs = sorted(ada)
        ax_t.plot(bs, [ada[b] / 1e6 for b in bs], color="0.45", ls="--", marker="s",
                  mfc="white", lw=1.4, ms=3.2,
                  label=f"Mamba-2, RTX 5000 Ada: {max(ada.values())/1e3:.0f} k tracks/s")
        lines.append(f"RTX 5000 Ada (Mamba-2, collaborator) peak {max(ada.values()):,} tracks/s"
                     " -- NOT yet re-measured on the minGRU")

    ax_t.set_xscale("log", base=2); ax_t.set_yscale("log")
    ax_t.set_xlabel("tracks per batch"); ax_t.set_ylabel("throughput [$10^6$ tracks/s]")
    ax_t.grid(True, which="both", ls=":", alpha=0.35)

    ax_c = ax_t.twiny()
    ax_c.set_xscale("log", base=2)
    ax_c.set_xlim(2 ** -0.55, 2 ** 6.55)
    ax_c.set_xticks(CPU_THREADS)
    ax_c.set_xticklabels([str(t) for t in CPU_THREADS])
    ax_c.set_xlabel("CPU threads")
    # Each axis sits next to its own curves; twiny() has to happen first.
    ax_t.xaxis.set_label_position("top")
    ax_t.tick_params(axis="x", which="both", top=True, labeltop=True,
                     bottom=False, labelbottom=False)
    ax_c.xaxis.set_label_position("bottom")
    ax_c.tick_params(axis="x", which="both", bottom=True, labelbottom=True,
                     top=False, labeltop=False)
    ideal = [CPU_TRACKS_PER_S[0] * t for t in CPU_THREADS]
    ax_c.plot(CPU_THREADS, [v / 1e6 for v in ideal], color="C3", ls=":", lw=1.2,
              alpha=0.7, label="ACTS KF fit, ideal linear scaling")
    ax_c.plot(CPU_THREADS, [v / 1e6 for v in CPU_TRACKS_PER_S], color="C3",
              ls="-", lw=1.6, marker="^", ms=3.5,
              label=f"ACTS KF fit, CPU: {CPU_TRACKS_PER_S[-1]/1e3:.0f} k tracks/s")

    fig.tight_layout()
    handles = ax_t.get_lines() + ax_c.get_lines()[::-1]
    ax_c.legend(handles, [h.get_label() for h in handles], fontsize=7.6,
                loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=2,
                framealpha=0.9, borderaxespad=0.0, columnspacing=1.2,
                handlelength=1.8)
    stem = out / "throughput_mingru_h100_vs_ada"
    fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(f"{stem}.png", dpi=110, bbox_inches="tight")
    plt.close(fig)

    lines.append("CPU (ACTS KF fit, Threadripper 3990X) thread scan [tracks/s]: "
                 + ", ".join(f"{t}t {v:,}" for t, v in zip(CPU_THREADS, CPU_TRACKS_PER_S)))
    (out / "throughput_summary_v2.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"[plot] {stem}.pdf")
    return 0


if __name__ == "__main__":
    sys.exit(main())
