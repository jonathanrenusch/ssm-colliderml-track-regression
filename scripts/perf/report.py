"""Nightly KPI report: results.jsonl + physics_gate.csv → OPTIMIZATION_LOG.md + PNGs.

Reads ``<results-dir>/results.jsonl`` and ``docs/perf/results/physics_gate.csv``,
builds a per-variant KPI table (best tracks/s, headline t2k, max working batch,
VRAM at best point, mean power, physics gate) and appends it to
``docs/perf/OPTIMIZATION_LOG.md`` under the "Night N" section (skipped when an
identical table is already present, so it can be re-run). Also renders
``t2k_summary.png`` and ``throughput_vs_batch.png`` from the sweep curves into
``docs/perf/plots/nightN/``.

Usage::

    pixi run -e default python scripts/perf/report.py \\
        --results-dir docs/perf/results/night1 --night 1
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import common  # noqa: E402

T2K_TARGET_MS = 0.5


def load_results(results_dir: Path) -> pd.DataFrame:
    path = results_dir / "results.jsonl"
    rows = []
    if path.exists():
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    return pd.DataFrame(rows)


def load_gates(gate_csv: Path) -> dict[str, str]:
    """variant -> latest gate verdict."""
    if not gate_csv.exists():
        return {}
    df = pd.read_csv(gate_csv)
    gates: dict[str, str] = {}
    for _, r in df.iterrows():
        verdict = str(r.get("gate", "") or "")
        if verdict:
            gates[str(r["variant"])] = verdict
    return gates


def kpi_table(df: pd.DataFrame, gates: dict[str, str]) -> str:
    lines = [
        "| variant | best tracks/s | t2k [ms] | best batch | max batch | "
        "VRAM [GiB] | power mean [W] | physics gate |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    ok = df[(df.get("status") == "ok") & df["tracks_per_s"].notna()]
    for variant in sorted(ok["variant"].dropna().unique()):
        sub = ok[ok["variant"] == variant]
        best = sub.loc[sub["tracks_per_s"].idxmax()]
        max_batch = int(sub["batch_tracks"].max())
        power = best.get("power_w_mean")
        power_s = f"{power:.0f}" if pd.notna(power) else "n/a"
        vram = best.get("vram_gib_torch_peak")
        vram_s = f"{vram:.2f}" if pd.notna(vram) else "n/a"
        lines.append(
            f"| {variant} | {best['tracks_per_s']:,.0f} | {best['t2k_ms']:.4f} "
            f"| {int(best['batch_tracks'])} | {max_batch} | {vram_s} | {power_s} "
            f"| {gates.get(variant, '—')} |"
        )
    return "\n".join(lines)


def append_to_log(log_path: Path, night: int, table_md: str) -> None:
    block = (
        f"\n### KPI table (auto, {common.utc_ts()})\n\n"
        f"Target: t2k <= {T2K_TARGET_MS} ms (>= 4 M tracks/s), 4L config.\n\n"
        f"{table_md}\n"
    )
    text = log_path.read_text() if log_path.exists() else ""
    if table_md in text:
        print("[report] identical KPI table already in log — skipping append", flush=True)
        return
    heading = f"## Night {night}"
    idx = text.find(heading)
    if idx < 0:
        text += f"\n{heading} — (auto-created)\n{block}"
    else:
        nxt = text.find("\n## ", idx + len(heading))
        insert_at = len(text) if nxt < 0 else nxt
        text = text[:insert_at] + block + text[insert_at:]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(text)
    print(f"[report] KPI table appended to {log_path}", flush=True)


def render_plots(df: pd.DataFrame, plots_dir: Path) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)
    ok = df[(df.get("status") == "ok") & df["tracks_per_s"].notna()]

    # throughput vs batch, sweep curves per variant
    sweep = ok[ok["mode"] == "sweep"]
    fig, ax = plt.subplots(figsize=(8, 5))
    for variant in sorted(sweep["variant"].dropna().unique()):
        sub = sweep[sweep["variant"] == variant].sort_values("batch_tracks")
        ax.plot(sub["batch_tracks"], sub["tracks_per_s"], "o-", label=variant)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("batch size [tracks]")
    ax.set_ylabel("tracks / s")
    ax.set_title("Sweep: throughput vs batch")
    if len(sweep):
        ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / "throughput_vs_batch.png", dpi=120)
    plt.close(fig)

    # t2k summary bar
    fig, ax = plt.subplots(figsize=(7, 4.5))
    variants, t2ks = [], []
    for variant in sorted(ok["variant"].dropna().unique()):
        sub = ok[ok["variant"] == variant]
        variants.append(variant)
        t2ks.append(float(sub["t2k_ms"].min()))
    ax.bar(variants, t2ks)
    ax.axhline(T2K_TARGET_MS, color="tab:red", ls="--",
               label=f"target {T2K_TARGET_MS} ms")
    for x, y in zip(variants, t2ks):
        ax.text(x, y, f"{y:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("t2k [ms] (2000 tracks)")
    ax.set_title("Headline t2k per variant (best point)")
    if variants:
        ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "t2k_summary.png", dpi=120)
    plt.close(fig)
    print(f"[report] plots -> {plots_dir}/t2k_summary.png, throughput_vs_batch.png",
          flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--results-dir", required=True, type=Path)
    ap.add_argument("--night", required=True, type=int)
    ap.add_argument("--log", type=Path,
                    default=common.REPO_ROOT / "docs/perf/OPTIMIZATION_LOG.md")
    ap.add_argument("--gate-csv", type=Path,
                    default=common.REPO_ROOT / "docs/perf/results/physics_gate.csv")
    ap.add_argument("--plots-dir", type=Path, default=None,
                    help="default: docs/perf/plots/night<N>")
    args = ap.parse_args()

    df = load_results(args.results_dir)
    if df.empty:
        print(f"[report] no results in {args.results_dir}/results.jsonl", file=sys.stderr)
        sys.exit(1)
    gates = load_gates(args.gate_csv)

    table_md = kpi_table(df, gates)
    print(table_md, flush=True)
    append_to_log(args.log, args.night, table_md)
    plots_dir = args.plots_dir or common.REPO_ROOT / f"docs/perf/plots/night{args.night}"
    render_plots(df, plots_dir)


if __name__ == "__main__":
    main()
