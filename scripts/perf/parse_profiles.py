"""Parse nsys / ncu profiles into per-kernel tables and PNGs.

nsys: exports the ``.nsys-rep`` to sqlite (``nsys export --type sqlite``),
reads ``CUPTI_ACTIVITY_KIND_KERNEL`` (restricted to the NVTX range named
'timed' when present — degrades gracefully to the whole capture), and emits:
a per-kernel table (short name, total ms, %, count, mean µs), the GPU-busy
vs wall-clock fraction, a kernel-waterfall PNG and a launch-gap histogram.

ncu: imports the ``.ncu-rep`` via ``ncu --import X --csv --page details`` and
tabulates the top kernels by duration with SM%, DRAM%, achieved occupancy and
tensor-pipe% where present.

Outputs land in ``--outdir``: ``<label>_nsys_kernels.{csv,txt}``,
``<label>_ncu_kernels.{csv,txt}``, ``kernel_waterfall_<label>.png``,
``launch_gaps_<label>.png``.

Usage::

    pixi run -e default python scripts/perf/parse_profiles.py \\
        --nsys-rep docs/perf/profiles/night1/v0_staged.nsys-rep \\
        --label v0_staged --outdir docs/perf/plots/night1
"""
from __future__ import annotations

import argparse
import re
import sqlite3
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

DEFAULT_BIN_DIR = "/usr/local/cuda-13.1/bin"


def short_name(name: str, maxlen: int = 64) -> str:
    """Strip template/param clutter from a demangled kernel name."""
    s = name.split("(")[0]
    s = re.sub(r"<.*", "", s)
    s = s.split()[-1] if s.split() else s
    return s[-maxlen:]


# ---------------------------------------------------------------------------
# nsys
# ---------------------------------------------------------------------------

def export_sqlite(nsys_rep: Path, bin_dir: str) -> Path:
    sqlite_path = nsys_rep.with_suffix(".sqlite")
    if sqlite_path.exists() and sqlite_path.stat().st_mtime >= nsys_rep.stat().st_mtime:
        print(f"[profiles] reusing {sqlite_path}", flush=True)
        return sqlite_path
    nsys = Path(bin_dir) / "nsys"
    cmd = [str(nsys), "export", "--type", "sqlite", "--force-overwrite", "true",
           "--output", str(sqlite_path), str(nsys_rep)]
    print(f"[profiles] $ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)
    return sqlite_path


def _nvtx_timed_range(con: sqlite3.Connection) -> tuple[int, int] | None:
    """(start, end) of the NVTX range whose text contains 'timed', else None."""
    try:
        rows = con.execute(
            "SELECT start, end, text, textId FROM NVTX_EVENTS WHERE end IS NOT NULL"
        ).fetchall()
    except sqlite3.OperationalError:
        return None
    strings: dict[int, str] = {}
    try:
        strings = dict(con.execute("SELECT id, value FROM StringIds").fetchall())
    except sqlite3.OperationalError:
        pass
    hits = []
    for start, end, text, text_id in rows:
        label = text or strings.get(text_id, "") or ""
        if "timed" in label.lower():
            hits.append((start, end))
    if not hits:
        return None
    return min(h[0] for h in hits), max(h[1] for h in hits)


def parse_nsys(nsys_rep: Path, label: str, outdir: Path, bin_dir: str) -> None:
    sqlite_path = export_sqlite(nsys_rep, bin_dir)
    con = sqlite3.connect(sqlite_path)

    strings = {}
    try:
        strings = dict(con.execute("SELECT id, value FROM StringIds").fetchall())
    except sqlite3.OperationalError:
        pass

    try:
        rows = con.execute(
            "SELECT start, end, demangledName, shortName "
            "FROM CUPTI_ACTIVITY_KIND_KERNEL"
        ).fetchall()
    except sqlite3.OperationalError as e:
        print(f"[profiles] ERROR: no kernel table in {sqlite_path}: {e}", file=sys.stderr)
        return
    if not rows:
        print("[profiles] no kernel records", file=sys.stderr)
        return

    window = _nvtx_timed_range(con)
    con.close()
    if window:
        n_before = len(rows)
        rows = [r for r in rows if r[0] >= window[0] and r[1] <= window[1]]
        print(f"[profiles] NVTX 'timed' range found: {len(rows)}/{n_before} kernels inside",
              flush=True)
    else:
        print("[profiles] no NVTX 'timed' range — using the whole capture", flush=True)

    recs = []
    for start, end, dem, short in rows:
        name = strings.get(dem) or strings.get(short) or str(dem)
        recs.append((short_name(str(name)), start, end, (end - start) / 1e3))  # ns → µs
    df = pd.DataFrame(recs, columns=["kernel", "start", "end", "dur_us"])

    total_us = df["dur_us"].sum()
    table = (
        df.groupby("kernel")
        .agg(total_ms=("dur_us", lambda x: x.sum() / 1e3),
             count=("dur_us", "size"),
             mean_us=("dur_us", "mean"))
        .sort_values("total_ms", ascending=False)
    )
    table["pct"] = 100.0 * table["total_ms"] * 1e3 / total_us

    # GPU-busy vs wall: union of kernel intervals over the window.
    iv = df[["start", "end"]].sort_values("start").to_numpy()
    busy, cur_s, cur_e = 0, iv[0][0], iv[0][1]
    for s, e in iv[1:]:
        if s > cur_e:
            busy += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    busy += cur_e - cur_s
    wall = df["end"].max() - df["start"].min()
    busy_pct = 100.0 * busy / wall if wall else float("nan")

    # Launch gaps between consecutive kernels (merged-interval holes).
    gaps_us = []
    iv2 = df[["start", "end"]].sort_values("start").to_numpy()
    prev_end = iv2[0][1]
    for s, e in iv2[1:]:
        if s > prev_end:
            gaps_us.append((s - prev_end) / 1e3)
        prev_end = max(prev_end, e)

    outdir.mkdir(parents=True, exist_ok=True)
    table.to_csv(outdir / f"{label}_nsys_kernels.csv")
    header = (
        f"nsys kernel summary — {label}\n"
        f"source: {nsys_rep}\n"
        f"kernels: {len(df)}   GPU busy: {busy / 1e6:.3f} ms / wall {wall / 1e6:.3f} ms "
        f"= {busy_pct:.1f}%\n"
        f"gaps: n={len(gaps_us)} total={sum(gaps_us) / 1e3:.3f} ms\n\n"
    )
    txt = header + table.to_string(float_format=lambda x: f"{x:.3f}")
    (outdir / f"{label}_nsys_kernels.txt").write_text(txt + "\n")
    print(txt, flush=True)

    # Waterfall: top-15 kernels by total time.
    top = table.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, max(3, 0.4 * len(top))))
    ax.barh(top.index, top["total_ms"])
    ax.set_xlabel("total time [ms]")
    ax.set_title(f"Kernel time — {label} (GPU busy {busy_pct:.1f}%)")
    fig.tight_layout()
    fig.savefig(outdir / f"kernel_waterfall_{label}.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    if gaps_us:
        ax.hist(gaps_us, bins=min(100, max(10, len(gaps_us) // 20)))
        ax.set_yscale("log")
    ax.set_xlabel("inter-kernel gap [µs]")
    ax.set_ylabel("count")
    ax.set_title(f"Launch gaps — {label} "
                 f"(n={len(gaps_us)}, total {sum(gaps_us) / 1e3:.2f} ms)")
    fig.tight_layout()
    fig.savefig(outdir / f"launch_gaps_{label}.png", dpi=120)
    plt.close(fig)
    print(f"[profiles] wrote {outdir}/kernel_waterfall_{label}.png, launch_gaps_{label}.png",
          flush=True)


# ---------------------------------------------------------------------------
# ncu
# ---------------------------------------------------------------------------

_NCU_METRIC_MAP = {
    "duration": ["Duration"],
    "sm_pct": ["Compute (SM) Throughput", "SM [%]", "SM Busy"],
    "dram_pct": ["Memory Throughput", "DRAM Throughput"],
    "occupancy_pct": ["Achieved Occupancy"],
    "tensor_pct": ["Tensor (All)", "Pipe Tensor", "Tensor Pipes Busy",
                   "Pipe Tensor Cycles Active"],
}


def parse_ncu(ncu_rep: Path, label: str, outdir: Path, bin_dir: str) -> None:
    ncu = Path(bin_dir) / "ncu"
    if not ncu.exists():
        ncu = Path("ncu")  # fall back to PATH (also installed in the env)
    cmd = [str(ncu), "--import", str(ncu_rep), "--csv", "--page", "details"]
    print(f"[profiles] $ {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"[profiles] ncu import failed: {proc.stderr[-2000:]}", file=sys.stderr)
        return
    from io import StringIO

    # ncu prepends ==PROF== noise lines; keep from the CSV header on.
    lines = proc.stdout.splitlines()
    start = next((i for i, ln in enumerate(lines) if ln.startswith('"ID"')), 0)
    df = pd.read_csv(StringIO("\n".join(lines[start:])))
    if df.empty:
        print("[profiles] ncu CSV empty", file=sys.stderr)
        return

    name_col = "Kernel Name" if "Kernel Name" in df.columns else "Name"
    rows: dict[tuple, dict] = {}
    for _, r in df.iterrows():
        key = (r.get("ID"), short_name(str(r.get(name_col, ""))))
        rec = rows.setdefault(key, {"id": r.get("ID"), "kernel": key[1]})
        metric = str(r.get("Metric Name", ""))
        try:
            value = float(str(r.get("Metric Value", "")).replace(",", ""))
        except ValueError:
            continue
        unit = str(r.get("Metric Unit", ""))
        for col, names in _NCU_METRIC_MAP.items():
            if any(metric == n or metric.startswith(n) for n in names):
                if col == "duration":
                    scale = {"second": 1e6, "msecond": 1e3, "usecond": 1.0,
                             "nsecond": 1e-3}.get(unit, 1.0)
                    rec["duration_us"] = value * scale
                else:
                    rec[col] = value

    out = pd.DataFrame(rows.values())
    if "duration_us" in out.columns:
        out = out.sort_values("duration_us", ascending=False)
    outdir.mkdir(parents=True, exist_ok=True)
    out.to_csv(outdir / f"{label}_ncu_kernels.csv", index=False)
    txt = (f"ncu kernel details — {label}\nsource: {ncu_rep}\n\n"
           + out.head(30).to_string(index=False,
                                    float_format=lambda x: f"{x:.2f}"))
    (outdir / f"{label}_ncu_kernels.txt").write_text(txt + "\n")
    print(txt, flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--nsys-rep", type=Path, default=None)
    ap.add_argument("--ncu-rep", type=Path, default=None)
    ap.add_argument("--label", required=True)
    ap.add_argument("--outdir", type=Path, default=Path("docs/perf/plots/night1"))
    ap.add_argument("--bin-dir", default=DEFAULT_BIN_DIR,
                    help="dir with nsys/ncu binaries")
    args = ap.parse_args()

    if not args.nsys_rep and not args.ncu_rep:
        ap.error("give at least one of --nsys-rep / --ncu-rep")
    if args.nsys_rep:
        parse_nsys(args.nsys_rep, args.label, args.outdir, args.bin_dir)
    if args.ncu_rep:
        parse_ncu(args.ncu_rep, args.label, args.outdir, args.bin_dir)


if __name__ == "__main__":
    main()
