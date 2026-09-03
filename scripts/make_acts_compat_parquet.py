#!/usr/bin/env python3
"""Shim the v2 (re-produced 2026-08) ColliderML parquet into the Release-1 layout
that pyacts' ``ColliderMLRelease1InputConverter`` expects, so the official
``acts_integration.py`` pipeline runs on the v2 data UNCHANGED.

What changed in the v2 production (and breaks the converter silently — every
particle is dropped as "without hits"):
- ``tracker_hits.particle_id``  became nested ``particle_ids: list<list<u64>>``
- ``tracker_hits.true_{x,y,z}`` moved to the separate ``tracker_simhits`` table
  (linked per hit by ``simhit_ids``, positional within the event)

This script writes, per run directory, a ``tracker_hits`` parquet with exactly
the columns/types of ``ColliderMLRelease1InputConverter.hitSchema()`` (plus
``event_id``): truth positions joined from ``tracker_simhits`` BY EVENT-ID VALUE
(never by row -- CLAUDE.md §0.4), positional simhit ids within the event, and
the first contributing particle per hit.  ``particles`` is symlinked unchanged
(its schema is a superset of the expected one, which the reader accepts).

Usage: make_acts_compat_parquet.py <src_run_dir> <out_run_dir> [...more pairs]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def _table_dirs(src: Path) -> tuple[Path, Path, Path]:
    """Locate (particles, tracker_hits, tracker_simhits) under either layout:
    ttbar runs: src/{particles,tracker_hits,tracker_simhits}; muon guns:
    src/parquet/{truth/{particles,tracker_simhits}, reco/tracker_hits}."""
    if (src / "particles").is_dir():
        return src / "particles", src / "tracker_hits", src / "tracker_simhits"
    for cand in (src / "parquet", src / "v1" / "parquet", src):
        if (cand / "truth" / "particles").is_dir():
            pq_root = cand
            break
    else:
        raise FileNotFoundError(f"no ColliderML table layout under {src}")
    return (pq_root / "truth" / "particles", pq_root / "reco" / "tracker_hits",
            pq_root / "truth" / "tracker_simhits")


def shim_run(src: Path, out: Path, max_events: int | None = None) -> None:
    p_dir, h_dir, s_dir = _table_dirs(src)
    (out / "tracker_hits").mkdir(parents=True, exist_ok=True)

    def _read(d, n=None):
        tabs, tot = [], 0
        for fp in sorted(d.glob("*.parquet")):
            t_ = pq.read_table(fp)
            tabs.append(t_); tot += t_.num_rows
            if n is not None and tot >= n:
                break
        t_ = pa.concat_tables(tabs)
        return t_.slice(0, n) if n is not None else t_

    if max_events is None:
        if not (out / "particles").exists():
            (out / "particles").symlink_to(p_dir.resolve())
    else:
        # sliced mode: the reader pairs collections row-wise, so particles must
        # be sliced to the same events as the hits
        (out / "particles").mkdir(parents=True, exist_ok=True)
        pq.write_table(_read(p_dir, max_events), out / "particles" / "part_000000.parquet")

    hits = _read(h_dir, max_events)
    sims = _read(s_dir)  # sim table stays whole; the join below is by event_id value
    sim_by_ev = {int(e): i for i, e in enumerate(sims["event_id"].to_pylist())}

    def cast_list(col, dtype):
        return [np.asarray(v, dtype=dtype) for v in col]

    out_cols: dict[str, list] = {k: [] for k in (
        "event_id", "x", "y", "z", "true_x", "true_y", "true_z", "time",
        "particle_id", "detector", "volume_id", "layer_id", "surface_id")}

    n_ev = hits.num_rows
    ev_ids = hits["event_id"].to_pylist()
    for i in range(n_ev):
        ev = int(ev_ids[i])
        j = sim_by_ev[ev]                      # join by event_id VALUE
        sh_ids = hits["simhit_ids"][i].values  # list<list<u64>> flattened per hit
        first_sim = np.array([int(v[0]) for v in sh_ids.to_pylist()], dtype=np.int64)
        p_ids = hits["particle_ids"][i].values
        first_p = np.array([int(v[0]) for v in p_ids.to_pylist()], dtype=np.uint64)
        tx = np.asarray(sims["true_x"][j].values, dtype=np.float32)[first_sim]
        ty = np.asarray(sims["true_y"][j].values, dtype=np.float32)[first_sim]
        tz = np.asarray(sims["true_z"][j].values, dtype=np.float32)[first_sim]
        # Release-1's hit `time` was the TRUTH time; v2's digitised time is 0 for
        # all strips, which scrambles TruthTrackFinder's per-track hit ordering
        # (it sorts by SimHit::time) and degrades the in-pipeline KF seeding/fit.
        # Default: substitute the sim true_time.  DIGI_TIME=1 keeps the raw column.
        t_time = np.asarray(sims["true_time"][j].values, dtype=np.float32)[first_sim]

        out_cols["event_id"].append(ev)
        for name, dtype in (("x", np.float32), ("y", np.float32), ("z", np.float32),
                            ("detector", np.uint8),
                            ("volume_id", np.uint8), ("layer_id", np.uint16),
                            ("surface_id", np.uint32)):
            out_cols[name].append(np.asarray(hits[name][i].values, dtype=dtype))
        out_cols["true_x"].append(tx); out_cols["true_y"].append(ty); out_cols["true_z"].append(tz)
        import os as _os
        if _os.environ.get("DIGI_TIME") == "1":
            out_cols["time"].append(np.asarray(hits["time"][i].values, dtype=np.float32))
        else:
            out_cols["time"].append(t_time)
        out_cols["particle_id"].append(first_p)

    schema = pa.schema([
        ("event_id", pa.int64()),
        ("x", pa.list_(pa.float32())), ("y", pa.list_(pa.float32())), ("z", pa.list_(pa.float32())),
        ("true_x", pa.list_(pa.float32())), ("true_y", pa.list_(pa.float32())), ("true_z", pa.list_(pa.float32())),
        ("time", pa.list_(pa.float32())),
        ("particle_id", pa.list_(pa.uint64())),
        ("detector", pa.list_(pa.uint8())), ("volume_id", pa.list_(pa.uint8())),
        ("layer_id", pa.list_(pa.uint16())), ("surface_id", pa.list_(pa.uint32())),
    ])
    table = pa.table({k: (out_cols[k] if k == "event_id"
                          else pa.array([v.tolist() for v in out_cols[k]],
                                        type=schema.field(k).type))
                      for k in out_cols}, schema=schema)
    pq.write_table(table, out / "tracker_hits" / "part_000000.parquet")
    print(f"[shim] {src} -> {out}: {n_ev} events, "
          f"{sum(len(v) for v in out_cols['x']):,} hits", flush=True)


if __name__ == "__main__":
    import os
    args = sys.argv[1:]
    assert args and len(args) % 2 == 0, __doc__
    max_ev = int(os.environ["MAX_EVENTS"]) if os.environ.get("MAX_EVENTS") else None
    for s, o in zip(args[::2], args[1::2]):
        shim_run(Path(s), Path(o), max_ev)
