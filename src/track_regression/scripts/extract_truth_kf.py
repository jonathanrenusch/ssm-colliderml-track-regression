#!/usr/bin/env python3
"""Attach truth-tracking-KF fits to a flat store, aligned to its track order.

`preprocess_flat.py` ingests the `tracks` table (CKF) into acts_reco. The
truth-tracking KF lives in a separate `truth_tracks` table, which only
single_muon_uniform ships, and is matched here on (event_id, particle_id) —
both of which the store records per track. Writes `truth_kf_reco.npy` next to
each part; fast_rms_eval then picks it up as the preferred reference.
"""
from __future__ import annotations
import argparse, glob, json, re
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq

P = ["d0", "z0", "phi", "theta", "qop"]


def _flat(col):
    ca = col.combine_chunks()
    return np.asarray(ca.values.to_numpy(zero_copy_only=False)), np.asarray(ca.offsets, np.int64)


def attach(store: Path, tt_glob: str, force: bool = False,
           drop_event_range: tuple[int, int] | None = None) -> None:
    files = sorted(glob.glob(tt_glob))
    rng = []
    for f in files:
        m = re.search(r"events(\d+)-(\d+)\.parquet", f)
        rng.append((int(m.group(1)), int(m.group(2)), f))
    man = json.loads((store / "manifest.json").read_text())

    # Duplicate keys must be found across the WHOLE store, not per part: the
    # write-time shuffle scatters the colliding events over different parts, so
    # a per-part scan sees each of a colliding pair as unique.
    all_keys = []
    for part in man["parts"]:
        d = store / part["name"]
        e = np.asarray(np.load(d / "track_event_ids.npy")).astype(np.int64)
        q = np.asarray(np.load(d / "track_particle_ids.npy")).astype(np.int64)
        all_keys.append((e << 32) | q)
    all_keys = np.concatenate(all_keys)
    uk, ucnt = np.unique(all_keys, return_counts=True)
    dup_keys = uk[ucnt > 1]
    lo_d, hi_d = drop_event_range if drop_event_range else (0, 0)
    if drop_event_range:
        print(f"  [provenance] dropping events [{lo_d}, {hi_d}): these ranges appear in more "
              f"than one run dir, so their event ids are reused by physically different events.")
    if dup_keys.size:
        n_amb = int(np.isin(all_keys, dup_keys).sum())
        print(f"  [ambiguous] {dup_keys.size:,} (event_id, particle_id) keys are shared by "
              f"more than one track, covering {n_amb:,}/{len(all_keys):,} tracks "
              f"({100*n_amb/len(all_keys):.2f}%) -- these are left unmatched. "
              f"See the module docstring: reused event_id ranges across run dirs.")

    for part in man["parts"]:
        d = store / part["name"]
        out = d / "truth_kf_reco.npy"
        if out.exists() and not force:
            print(f"  {part['name']}: cached"); continue
        ev = np.asarray(np.load(d / "track_event_ids.npy"))
        pid = np.asarray(np.load(d / "track_particle_ids.npy")).astype(np.uint64)
        lo, hi = int(ev.min()), int(ev.max())
        hit = [f for a, b, f in rng if a <= lo and hi <= b]
        if not hit:
            print(f"  {part['name']}: NO truth_tracks shard covers events {lo}-{hi}"); continue
        t = pq.read_table(hit[0], columns=["event_id"] + P + ["majority_particle_id"])
        maj, off = _flat(t.column("majority_particle_id"))
        ev_row = t.column("event_id").to_numpy(zero_copy_only=False).astype(np.int64)
        ev_tt = np.repeat(ev_row, np.diff(off))
        cols = {p: _flat(t.column(p))[0].astype(np.float64) for p in P}
        uniq, inv = np.unique(np.concatenate([pid, maj.astype(np.uint64)]), return_inverse=True)
        k_store = (ev.astype(np.int64) << 32) | inv[:len(pid)].astype(np.int64)
        k_tt = (ev_tt << 32) | inv[len(pid):].astype(np.int64)
        # a store key shared by >1 track cannot be resolved -- see the module
        # docstring on reused event_id ranges. Refuse rather than mispair.
        # NB: dup_keys lives in the RAW (event<<32)|pid space, not the dense-rank
        # space k_store uses, so rebuild the raw key here to test membership.
        k_raw = (ev.astype(np.int64) << 32) | pid.astype(np.int64)
        amb = np.isin(k_raw, dup_keys) if dup_keys.size else np.zeros(len(k_store), bool)
        if drop_event_range:
            amb |= (ev >= lo_d) & (ev < hi_d)

        order = np.argsort(k_tt, kind="stable"); ks = k_tt[order]
        pos = np.searchsorted(ks, k_store)
        ok = pos < len(ks); pc = np.where(ok, pos, 0); ok &= ks[pc] == k_store
        ok &= ~amb
        src = order[pc]
        reco = np.full((len(ev), 5), np.nan, np.float32)
        for i, p in enumerate(P):
            reco[ok, i] = cols[p][src[ok]]
        np.save(out, reco)
        print(f"  {part['name']}: matched {ok.sum():,}/{len(ok):,} ({100*ok.mean():.2f}%)"
              f"{f'  [{amb.sum():,} ambiguous keys dropped]' if amb.any() else ''}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", required=True, help="flat store split dir (holds manifest.json)")
    ap.add_argument("--truth-tracks-glob", required=True)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--drop-event-range", nargs=2, type=int, metavar=("LO", "HI"),
                    help="leave tracks with LO <= event_id < HI unmatched; use for event-id "
                         "ranges that appear in more than one runs/*/ directory")
    a = ap.parse_args()
    attach(Path(a.store), a.truth_tracks_glob, a.force,
           tuple(a.drop_event_range) if a.drop_event_range else None)
