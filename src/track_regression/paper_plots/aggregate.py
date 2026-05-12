"""Cross-run summary tables (LaTeX + CSV) — auto-runs at end of every cli.py call.

Walks ``<output_root>/*/stats.json`` + their ``metadata.yaml``, builds:
  * all_runs.{tex,csv}            — every run, every param, IQR + iter-3σ RMS
  * ablation_<axis>.{tex,csv}     — runs tagged with ``axis`` in metadata.yaml

Empty tables are still emitted so a downstream LaTeX include never breaks.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import yaml

from hepattn.experiments.colliderml_regr.eval_utils import PARAMS

from . import PAPER_PLOTS_ROOT
from .stats import DISPLAY_SCALE, DISPLAY_UNIT


KNOWN_AXES = ["pooling", "transformer", "finetune", "scaling", "d0_head", "loss_design"]


def _load_runs(root: Path) -> list[dict]:
    runs = []
    for run_dir in sorted(root.iterdir()):
        if not run_dir.is_dir() or run_dir.name.startswith("_"):
            continue
        sj = run_dir / "stats.json"
        mj = run_dir / "metadata.yaml"
        if not sj.exists() or not mj.exists():
            continue
        with open(sj) as f:
            stats = json.load(f)
        with open(mj) as f:
            meta = yaml.safe_load(f) or {}
        runs.append({"name": run_dir.name, "stats": stats, "meta": meta})
    return runs


def _csv_rows(runs: list[dict]) -> list[list]:
    """Per-run flat rows, columns: nicename, count,
       d0_iqr_ssm ± σ, d0_iqr_ckf ± σ, d0_iqr_ratio ± σ,
       d0_rms_ssm ± σ, d0_rms_ckf ± σ, d0_rms_ratio ± σ, ... × 5 params."""
    header = ["nicename", "count"]
    for p in PARAMS:
        u = DISPLAY_UNIT[p]
        for metric in ("iqr", "rms"):
            for who in ("ssm", "ckf", "ratio"):
                header.append(f"{p}_{metric}_{who}_mean ({u if who != 'ratio' else '–'})")
                header.append(f"{p}_{metric}_{who}_2sigma")

    rows = [header]
    for r in runs:
        s = r["stats"]
        row = [r["name"], s.get("count", 0)]
        for p in PARAMS:
            scale = DISPLAY_SCALE[p]
            for metric, key in (("iqr", "iqr_robust_sigma"), ("rms", "iter3sigma_rms")):
                d = s[p][key]
                for who in ("ssm", "ckf"):
                    mean, sig = d[who]
                    row.append(mean * scale)
                    row.append(2 * sig * scale)  # 2σ (95% CI)
                mean, sig = d["ratio"]
                row.append(mean)
                row.append(2 * sig)
        rows.append(row)
    return rows


def _write_csv(path: Path, rows: list[list]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerows(rows)


def _write_latex(path: Path, runs: list[dict], caption: str) -> None:
    """Compact LaTeX table: per-run, per-param IQR / RMS / ratio (post-clip)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not runs:
        path.write_text(f"% empty: {caption}\n")
        return

    cols = "l " + " ".join(["c c c"] * len(PARAMS))  # iqr / rms / ratio per param
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        rf"\caption{{{caption}}}",
        rf"\begin{{tabular}}{{{cols}}}",
        r"\toprule",
        "Run & " + " & ".join(
            rf"\multicolumn{{3}}{{c}}{{{p} [{DISPLAY_UNIT[p]}]}}" for p in PARAMS
        ) + r" \\",
    ]
    cmid = " ".join(rf"\cmidrule(lr){{{2 + 3*i}-{4 + 3*i}}}" for i in range(len(PARAMS)))
    lines.append(cmid)
    lines.append(" & " + " & ".join(["IQR", r"RMS$_{3\sigma}$", "ratio"] * len(PARAMS)) + r" \\")
    lines.append(r"\midrule")

    for r in runs:
        name_safe = r["name"].replace("_", r"\_")
        cells = [name_safe]
        for p in PARAMS:
            scale = DISPLAY_SCALE[p]
            iqr = r["stats"][p]["iqr_robust_sigma"]
            rms = r["stats"][p]["iter3sigma_rms"]
            cells.append(f"{iqr['ssm'][0] * scale:.3g}")
            cells.append(f"{rms['ssm'][0] * scale:.3g}")
            cells.append(f"{rms['ratio'][0]:.2f}")
        lines.append(" & ".join(cells) + r" \\")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    path.write_text("\n".join(lines) + "\n")


def run(output_root: Path = PAPER_PLOTS_ROOT) -> None:
    output_root = Path(output_root)
    summary_dir = output_root / "_summary"
    summary_dir.mkdir(exist_ok=True)
    runs = _load_runs(output_root)

    # all runs
    rows = _csv_rows(runs)
    _write_csv(summary_dir / "all_runs.csv", rows)
    _write_latex(summary_dir / "all_runs.tex", runs,
                 "All runs — IQR/1.349, iter-3σ RMS, SSM/CKF post-clip ratio.")

    # per axis
    for axis in KNOWN_AXES:
        axis_runs = [r for r in runs if axis in (r["meta"].get("ablation_axes") or [])]
        _write_csv(summary_dir / f"ablation_{axis}.csv", _csv_rows(axis_runs))
        _write_latex(summary_dir / f"ablation_{axis}.tex", axis_runs,
                     f"Ablation: {axis} — IQR/1.349, iter-3σ RMS, SSM/CKF post-clip ratio.")

    # log
    print(f"[aggregate] wrote {len(runs)} runs into {summary_dir}")
    for axis in KNOWN_AXES:
        n = sum(1 for r in runs if axis in (r["meta"].get("ablation_axes") or []))
        print(f"  axis={axis:<14s} n={n}")


if __name__ == "__main__":
    run()
