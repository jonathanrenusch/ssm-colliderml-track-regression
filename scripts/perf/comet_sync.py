"""Best-effort Comet sync of nightly perf results. NEVER fails the caller.

Tails ``<results-dir>/results.jsonl`` from a byte-offset cursor
(``<results-dir>/.comet_cursor``) and logs new rows as Comet metrics named
``{variant}/tracks_per_s``, ``{variant}/t2k_ms``, etc. Tries an online
``comet_ml.Experiment`` first; on ANY exception falls back to
``OfflineExperiment`` (same offline-zip pattern/dir as
``track_regression._lib.comet_logger``: ``logs/comet_offline``). One
experiment per night: the experiment key is persisted in
``<results-dir>/.comet_experiment_key`` and reattached via
``ExistingExperiment`` on later invocations when online.

``--finalize`` additionally uploads the night's plots directory as assets.

All failures are caught and reported on stderr; exit code is always 0 unless
the arguments themselves are invalid.

Usage::

    pixi run -e default python scripts/perf/comet_sync.py \\
        --results-dir docs/perf/results/night1 --night 1 --project ssm-track-perf
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import common  # noqa: E402

METRIC_KEYS = [
    "tracks_per_s", "tokens_per_s", "t2k_ms", "t_iter_ms_mean", "t_iter_ms_p50",
    "batch_tracks", "batch_tokens", "vram_gib_torch_peak",
    "power_w_mean", "power_w_max", "sm_util_mean", "clocks_sm_mean",
]
OFFLINE_DIR = common.REPO_ROOT / "logs" / "comet_offline"


def _err(msg: str) -> None:
    print(f"[comet_sync] {msg}", file=sys.stderr, flush=True)


def get_experiment(project: str, night: int, key_file: Path):
    """Online Experiment (reattached if possible), else OfflineExperiment."""
    import comet_ml

    name = f"night{night}-perf"
    prev_key = key_file.read_text().strip() if key_file.exists() else ""
    try:
        if prev_key:
            exp = comet_ml.ExistingExperiment(
                previous_experiment=prev_key, log_env_details=False,
            )
        else:
            exp = comet_ml.Experiment(
                project_name=project, log_env_details=False,
                auto_param_logging=False, parse_args=False,
            )
            exp.set_name(name)
            key_file.write_text(exp.get_key())
        return exp, "online"
    except Exception as e:
        _err(f"online experiment failed ({type(e).__name__}: {e}); going offline")
        OFFLINE_DIR.mkdir(parents=True, exist_ok=True)
        exp = comet_ml.OfflineExperiment(
            project_name=project, offline_directory=str(OFFLINE_DIR),
            log_env_details=False, auto_param_logging=False, parse_args=False,
        )
        exp.set_name(name)
        return exp, "offline"


def read_new_rows(results_jsonl: Path, cursor_file: Path) -> tuple[list[dict], int]:
    offset = 0
    if cursor_file.exists():
        try:
            offset = int(cursor_file.read_text().strip() or 0)
        except ValueError:
            offset = 0
    rows: list[dict] = []
    with open(results_jsonl) as f:
        f.seek(offset)
        for line in f:
            if not line.endswith("\n"):
                break  # partial write in flight — retry next invocation
            offset += len(line.encode())
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows, offset


def sync(args) -> None:
    results_jsonl = args.results_dir / "results.jsonl"
    cursor_file = args.results_dir / ".comet_cursor"
    key_file = args.results_dir / ".comet_experiment_key"

    rows: list[dict] = []
    new_offset = None
    if results_jsonl.exists():
        rows, new_offset = read_new_rows(results_jsonl, cursor_file)
    if not rows and not args.finalize:
        _err("nothing new to sync")
        return

    exp, mode = get_experiment(args.project, args.night, key_file)
    _err(f"experiment mode: {mode}, new rows: {len(rows)}")

    step0 = int(time.time())  # monotone-ish step base across invocations
    for i, row in enumerate(rows):
        variant = row.get("variant", "unk")
        mode_name = row.get("mode", "")
        metrics = {}
        for k in METRIC_KEYS:
            v = row.get(k)
            if isinstance(v, (int, float)) and v is not None:
                metrics[f"{variant}/{k}"] = v
                if mode_name:
                    metrics[f"{variant}/{mode_name}/{k}"] = v
        if metrics:
            exp.log_metrics(metrics, step=step0 + i)
        if row.get("status") not in (None, "ok"):
            exp.log_other(f"{variant}/last_error", str(row.get("error", ""))[:500])

    if args.finalize:
        plots_dir = common.REPO_ROOT / f"docs/perf/plots/night{args.night}"
        if plots_dir.is_dir():
            for p in sorted(plots_dir.glob("*")):
                if p.is_file() and p.suffix in (".png", ".csv", ".txt"):
                    try:
                        exp.log_asset(str(p), file_name=p.name)
                    except Exception as e:
                        _err(f"asset upload failed for {p.name}: {e}")
            _err(f"finalize: uploaded assets from {plots_dir}")
        gate_csv = common.REPO_ROOT / "docs/perf/results/physics_gate.csv"
        if gate_csv.exists():
            try:
                exp.log_asset(str(gate_csv), file_name=gate_csv.name)
            except Exception as e:
                _err(f"asset upload failed for physics_gate.csv: {e}")

    exp.end()
    if new_offset is not None:
        cursor_file.write_text(str(new_offset))
    _err(f"synced {len(rows)} rows (cursor -> {new_offset})")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--results-dir", required=True, type=Path)
    ap.add_argument("--night", required=True, type=int)
    ap.add_argument("--project", default="ssm-track-perf")
    ap.add_argument("--finalize", action="store_true",
                    help="also upload the night's plots dir as assets")
    args = ap.parse_args()

    try:
        sync(args)
    except Exception:
        _err("sync failed (non-fatal):")
        traceback.print_exc()
    sys.exit(0)


if __name__ == "__main__":
    main()
