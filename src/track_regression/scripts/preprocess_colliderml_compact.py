#!/usr/bin/env python3
# ruff: noqa: TID252, PLR0915, C901
"""Compact preprocessing of ColliderML Release-1 parquet shards for track regression.

This is a *compact* variant of ``preprocess_colliderml.py``.  It stores **only**
the hits belonging to selected tracks (not the full event), which typically
reduces per-shard size from ~1.3 GB to ~5–40 MB — a 30–250× reduction.

The output format is fully compatible with ``ColliderMLTrackDataset`` in
``data.py``; the CSR indices simply point into the compact hits array instead
of a full-event hits array.

Output structure (per shard)::

    <output_dir>/shard_XXXX/
        hits.npy                   — (compact_total_hits, N_HIT_FEATURES) float32
        selected_tracks/
            track_targets.npy       — (N_selected, 5) float32  [d0, z0, phi, theta, qop]
            track_hit_indices.npy   — (compact_total_hits,) int32  CSR values
            track_hit_offsets.npy   — (N_selected + 1,) int32  CSR offsets
            track_event_idx.npy     — (N_selected,) int32  event index within shard
            track_particle_ids.npy  — (N_selected,) int64  truth particle_id per track
            acts_reco.npy           — (N_selected, 5) float32  ACTS reco params
                                      (only when --augment-acts; NaN if no ACTS match)
            acts_dm_mask.npy        — (N_selected,) bool  double-matched mask
                                      (only when --augment-acts)

Usage::

    python preprocess_colliderml_compact.py \\
        --data-dir /scratch/colliderml/arxiv_retraining/raw \\
        --output-dir /scratch/colliderml/arxiv_retraining/p200_core_kf_matched_finetune \\
        --num-shards -1 --num-workers 8 --augment-acts

Quick test (2 shards)::

    python preprocess_colliderml_compact.py \\
        --data-dir /scratch/colliderml/arxiv_retraining/raw \\
        --output-dir /tmp/p200_compact_test \\
        --num-shards 2 --augment-acts
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from tqdm import tqdm

from track_regression.selection_utils import (
    load_selection_variant,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Hit features: x, y, z, r, phi_hit, theta_hit, s, volume_id, layer_id, surface_id, detector, eta_hit
# eta_hit is derived from theta_hit at preprocessing time (was historically computed
# at DataLoader time — moved here to reduce per-batch CPU work and improve GPU
# utilisation). Dataloader handles both 11- and 12-column legacy/new shards.
N_HIT_FEATURES = 12
HIT_FEATURE_NAMES = [
    "x", "y", "z", "r", "phi_hit", "theta_hit", "s",
    "volume_id", "layer_id", "surface_id", "detector", "eta_hit",
]

# Track target parameters
TARGET_NAMES = ["d0", "z0", "phi", "theta", "qop"]

# Detector type sets (ODD geometry)
PIXEL_DETECTORS = np.array([0, 1, 2, 3], dtype=np.int32)
STRIP_DETECTORS = np.array([4, 5, 6, 7, 8], dtype=np.int32)


# ---------------------------------------------------------------------------
# EOS-robust saving
# ---------------------------------------------------------------------------
#
# EOS under heavy parallel I/O occasionally returns ENOENT from open() even
# though the parent directory exists — a known transient consistency issue.
# `_safe_save` wraps np.save with a short retry loop that re-creates the parent
# directory and sleeps briefly between attempts.  On persistent failure, the
# original exception is re-raised so the worker still reports it.
#
# `_COMPLETE_MARKER` is the filename of an empty sentinel file written at the
# very end of process_shard().  Main loop uses its presence (not the older
# track_targets.npy check) to decide whether a shard is complete and skippable,
# so partially-written shards from a previous crash are automatically redone.

_COMPLETE_MARKER = "_complete"
_SAVE_RETRY_ATTEMPTS = 6
_SAVE_RETRY_BASE_SLEEP = 0.5  # seconds; exponential backoff


def _safe_save(path: Path, arr: np.ndarray) -> None:
    """np.save with retries for transient EOS ENOENT / OSError."""
    parent = path.parent
    last_exc: Exception | None = None
    for attempt in range(_SAVE_RETRY_ATTEMPTS):
        try:
            parent.mkdir(parents=True, exist_ok=True)
            np.save(path, arr)
            return
        except (FileNotFoundError, OSError) as e:
            last_exc = e
            time.sleep(_SAVE_RETRY_BASE_SLEEP * (2 ** attempt))
    # All retries exhausted — re-raise to surface the error in the worker.
    raise last_exc  # type: ignore[misc]

# Default selection file
_SELECTION_FILE = (
    Path(__file__).resolve().parent.parent
    / "selection_p200_datasets.yaml"
)


# ---------------------------------------------------------------------------
# Per-shard processing
# ---------------------------------------------------------------------------


def process_shard(
    particle_file: Path,
    hits_file: Path,
    output_dir: Path,
    selection: dict,
    tracks_file: Path | None = None,
) -> dict:
    """Process a single shard pair → write compact memmap files.

    Parameters
    ----------
    particle_file:
        Path to the particle parquet file for this shard.
    hits_file:
        Path to the hits parquet file for this shard.
    output_dir:
        Directory where the output numpy files will be written.
    selection:
        Dict of track selection cuts.
    tracks_file:
        Optional path to ACTS reconstructed tracks parquet file.

    Returns a summary dict with counts.
    """
    sel = selection

    # Read with column projection
    ptable = pq.read_table(
        particle_file,
        columns=[
            "event_id", "particle_id", "pdg_id", "charge",
            "px", "py", "pz", "perigee_d0", "perigee_z0",
            "primary", "vertex_primary",
        ],
    )
    htable = pq.read_table(
        hits_file,
        columns=[
            "event_id", "particle_id",
            "x", "y", "z",
            "volume_id", "layer_id", "surface_id", "detector",
            # Truth time (unsmeared, ns) — used as sort key in place of
            # ``s = sqrt(x²+y²+z²)``. Time is monotone in on-helix arc
            # length even for forward tracks, where ``s`` underestimates
            # arc length and produces a wrong sequence order.
            "time",
        ],
    )

    # Load ACTS reconstructed tracks if requested
    ttable = None
    ttable_eid_to_row: dict[int, int] = {}
    if tracks_file is not None:
        ttable = pq.read_table(
            tracks_file,
            columns=["event_id", "d0", "z0", "phi", "theta", "qop",
                     "majority_particle_id", "hit_ids"],
        )
        for row_idx in range(ttable.num_rows):
            ttable_eid_to_row[ttable.column("event_id")[row_idx].as_py()] = row_idx

    n_events = ptable.num_rows

    # Detector-specific hit requirement config
    require_pixel_strip = sel.get("require_pixel_strip", False)
    min_pixel_hits = sel.get("min_pixel_hits", 0)
    min_strip_hits = sel.get("min_strip_hits", 0)

    # Accumulators — compact format: only selected track hits
    all_compact_hits = []        # list of (L_i, N_HIT_FEATURES) arrays
    all_compact_hit_times = []   # list of (L_i,) float32 arrays — parallel to all_compact_hits
    all_track_targets = []       # list of (5,) arrays
    all_track_lengths = []       # list of ints
    all_track_event_idx = []     # list of ints
    all_track_acts_reco = []     # list of (5,) float32 arrays
    all_track_acts_dm = []       # list of bool
    all_track_particle_ids = []  # list of int
    all_track_meta = []          # list of (2,) arrays: [pt, vertex_primary]

    compact_hit_offset = 0

    for ev in range(n_events):
        # ---- Particles ------------------------------------------------
        pid = np.array(ptable.column("particle_id")[ev].as_py(), dtype=np.int64)
        pdg = np.array(ptable.column("pdg_id")[ev].as_py(), dtype=np.int32)
        charge = np.array(ptable.column("charge")[ev].as_py(), dtype=np.float32)
        px = np.array(ptable.column("px")[ev].as_py(), dtype=np.float64)
        py = np.array(ptable.column("py")[ev].as_py(), dtype=np.float64)
        pz = np.array(ptable.column("pz")[ev].as_py(), dtype=np.float64)
        d0 = np.array(ptable.column("perigee_d0")[ev].as_py(), dtype=np.float64)
        z0 = np.array(ptable.column("perigee_z0")[ev].as_py(), dtype=np.float64)
        is_primary = np.array(ptable.column("primary")[ev].as_py(), dtype=bool)
        vertex_primary = np.array(ptable.column("vertex_primary")[ev].as_py(), dtype=np.int32)

        # Derived kinematics
        pt = np.sqrt(px**2 + py**2)
        p = np.sqrt(px**2 + py**2 + pz**2)
        theta = np.arccos(np.clip(pz / (p + 1e-12), -1.0, 1.0))
        phi = np.arctan2(py, px)
        eta = -np.log(np.tan(theta / 2.0 + 1e-12))
        qop = np.where(p > 0, charge / p, 0.0)

        nparts = len(pid)

        # ---- Hits -----------------------------------------------------
        hx = np.array(htable.column("x")[ev].as_py(), dtype=np.float64)
        hy = np.array(htable.column("y")[ev].as_py(), dtype=np.float64)
        hz = np.array(htable.column("z")[ev].as_py(), dtype=np.float64)
        h_pid = np.array(htable.column("particle_id")[ev].as_py(), dtype=np.int64)
        h_vol = np.array(htable.column("volume_id")[ev].as_py(), dtype=np.int32)
        h_lay = np.array(htable.column("layer_id")[ev].as_py(), dtype=np.int32)
        h_surf = np.array(htable.column("surface_id")[ev].as_py(), dtype=np.int32)
        h_det = np.array(htable.column("detector")[ev].as_py(), dtype=np.int32)
        # Truth time per hit (ns, unsmeared). Used only as a sort key —
        # NOT included in the 12-feature input array (track regression
        # remains 3-D). Stored alongside hits.npy as hit_times.npy.
        h_time = np.array(htable.column("time")[ev].as_py(), dtype=np.float32)

        nhits = len(hx)

        # Derived hit features
        r = np.sqrt(hx**2 + hy**2)
        phi_hit = np.arctan2(hy, hx)
        theta_hit = np.arccos(np.clip(hz / (np.sqrt(hx**2 + hy**2 + hz**2) + 1e-12), -1.0, 1.0))
        s = np.sqrt(hx**2 + hy**2 + hz**2)  # distance from IP
        # eta_hit derived from theta_hit (same formula used previously in the DataLoader)
        eta_hit = -np.log(np.tan(np.clip(theta_hit, 1e-8, np.pi - 1e-8) / 2.0))
        eta_hit = np.clip(eta_hit, -10.0, 10.0)

        # Build full hit feature matrix for this event (used for gathering)
        hit_feats = np.zeros((nhits, N_HIT_FEATURES), dtype=np.float32)
        hit_feats[:, 0] = hx.astype(np.float32)
        hit_feats[:, 1] = hy.astype(np.float32)
        hit_feats[:, 2] = hz.astype(np.float32)
        hit_feats[:, 3] = r.astype(np.float32)
        hit_feats[:, 4] = phi_hit.astype(np.float32)
        hit_feats[:, 5] = theta_hit.astype(np.float32)
        hit_feats[:, 6] = s.astype(np.float32)
        hit_feats[:, 7] = h_vol.astype(np.float32)
        hit_feats[:, 8] = h_lay.astype(np.float32)
        hit_feats[:, 9] = h_surf.astype(np.float32)
        hit_feats[:, 10] = h_det.astype(np.float32)
        hit_feats[:, 11] = eta_hit.astype(np.float32)

        # ---- Track selection ------------------------------------------
        # Count hits per particle
        if nhits > 0:
            unique_pids, counts = np.unique(h_pid, return_counts=True)
            pid_to_nhits = dict(zip(unique_pids.tolist(), counts.tolist()))
        else:
            pid_to_nhits = {}

        nhits_per_particle = np.array(
            [pid_to_nhits.get(int(pp), 0) for pp in pid],
            dtype=np.int32,
        )

        mask = np.ones(nparts, dtype=bool)
        mask &= nhits_per_particle >= sel["min_hits"]
        if "max_hits" in sel:
            mask &= nhits_per_particle <= sel["max_hits"]
        if sel["primary"]:
            mask &= is_primary
        if sel["hard_scatter"]:
            mask &= vertex_primary == 1
        mask &= pt >= sel["pt_min"]
        mask &= (eta >= sel["eta_min"]) & (eta <= sel["eta_max"])
        mask &= charge != 0  # charged particles only
        # Filter out particles with NaN perigee parameters
        mask &= np.isfinite(d0) & np.isfinite(z0)
        # Perigee range cuts
        mask &= (d0 >= sel["d0_min"]) & (d0 <= sel["d0_max"])
        mask &= (z0 >= sel["z0_min"]) & (z0 <= sel["z0_max"])

        sel_indices = np.where(mask)[0]

        # Pre-compute ACTS reco lookup for this event
        pid_to_acts_track: dict = {}
        if ttable is not None:
            ev_event_id = int(ptable.column("event_id")[ev].as_py())
            trow = ttable_eid_to_row.get(ev_event_id)
            if trow is not None:
                ev_d0    = ttable.column("d0")[trow].as_py()
                ev_z0    = ttable.column("z0")[trow].as_py()
                ev_phi   = ttable.column("phi")[trow].as_py()
                ev_theta = ttable.column("theta")[trow].as_py()
                ev_qop   = ttable.column("qop")[trow].as_py()
                ev_maj   = ttable.column("majority_particle_id")[trow].as_py()
                ev_hids  = ttable.column("hit_ids")[trow].as_py()
                for t_idx, maj_pid in enumerate(ev_maj):
                    pid_to_acts_track[maj_pid] = (
                        np.array([ev_d0[t_idx], ev_z0[t_idx], ev_phi[t_idx],
                                   ev_theta[t_idx], ev_qop[t_idx]], dtype=np.float32),
                        set(ev_hids[t_idx]),  # reco hit id set
                    )

        use_acts_hits_only = sel.get("use_acts_hits_only", False)
        require_acts_dm = sel.get("require_acts_dm", False)

        for si in sel_indices:
            sel_pid = int(pid[si])
            # Find all truth hits belonging to this particle
            hit_mask = h_pid == sel_pid
            track_hit_local = np.where(hit_mask)[0]

            n_track_hits = len(track_hit_local)
            if n_track_hits < sel["min_hits"]:
                continue
            if "max_hits" in sel and n_track_hits > sel["max_hits"]:
                continue

            # Detector-specific hit requirement (on truth hits)
            if require_pixel_strip:
                track_det = h_det[track_hit_local]
                n_pixel = np.isin(track_det, PIXEL_DETECTORS).sum()
                n_strip = np.isin(track_det, STRIP_DETECTORS).sum()
                if n_pixel < min_pixel_hits or n_strip < min_strip_hits:
                    continue

            # ACTS double-matching — always computed on original truth hits
            acts_dm = False
            if ttable is not None:
                if sel_pid in pid_to_acts_track:
                    acts_reco_params, reco_hit_set = pid_to_acts_track[sel_pid]
                    truth_hit_set = set(track_hit_local.tolist())
                    n_majority = len(reco_hit_set & truth_hit_set)
                    n_reco = len(reco_hit_set)
                    n_truth = len(truth_hit_set)
                    purity = n_majority / n_reco if n_reco > 0 else 0.0
                    efficiency = n_majority / n_truth if n_truth > 0 else 0.0
                    acts_dm = purity > 0.75 and efficiency > 0.75

            # Skip non-double-matched tracks if required
            if require_acts_dm and not acts_dm:
                continue

            # After DM check: optionally replace truth hits with the exact set
            # of hits the CKF assigned to this track (pure CKF hits, *including*
            # any wrong/noise hits the CKF picked up — no truth intersection).
            if use_acts_hits_only:
                if sel_pid not in pid_to_acts_track:
                    continue  # no ACTS match → no KF hits → skip
                _, reco_hit_set_for_hits = pid_to_acts_track[sel_pid]
                # Filter to valid local indices (defensive — they should already
                # be local indices into the per-event hit table, same space as
                # the DM purity/efficiency calc above).
                track_hit_local = np.array(
                    sorted(i for i in reco_hit_set_for_hits if 0 <= i < nhits),
                    dtype=np.int64,
                )
                n_track_hits = len(track_hit_local)
                if n_track_hits < sel["min_hits"]:
                    continue
                if "max_hits" in sel and n_track_hits > sel["max_hits"]:
                    continue

            # Sort track hits by truth time (linear in on-helix arc length —
            # correct for forward tracks at large |z|, where the previous
            # ``s = sqrt(x²+y²+z²)`` heuristic underestimates arc length and
            # produced a wrong sequence order).
            track_time = h_time[track_hit_local]
            sort_order = np.argsort(track_time, kind="stable")
            track_hit_local = track_hit_local[sort_order]

            # Gather hit features for this track (compact storage)
            # Placed after all filtering to avoid appending hits for skipped tracks
            track_hit_feats = hit_feats[track_hit_local]  # (L, N_HIT_FEATURES)
            all_compact_hits.append(track_hit_feats)
            # Truth time per hit, in the same time-sorted order as the features.
            all_compact_hit_times.append(h_time[track_hit_local].astype(np.float32))

            all_track_particle_ids.append(sel_pid)
            if ttable is not None:
                if sel_pid in pid_to_acts_track:
                    all_track_acts_reco.append(acts_reco_params)
                    all_track_acts_dm.append(acts_dm)
                else:
                    all_track_acts_reco.append(np.full(5, np.nan, dtype=np.float32))
                    all_track_acts_dm.append(False)

            # Targets
            targets = np.array([
                d0[si], z0[si], phi[si], theta[si], qop[si],
            ], dtype=np.float32)

            all_track_targets.append(targets)
            all_track_lengths.append(n_track_hits)
            all_track_event_idx.append(ev)
            all_track_meta.append(np.array([pt[si], float(vertex_primary[si])], dtype=np.float32))

    # ---- Write output -------------------------------------------------
    output_dir.mkdir(parents=True, exist_ok=True)
    sel_dir = output_dir / "selected_tracks"
    sel_dir.mkdir(exist_ok=True)

    # Compact hits array
    n_selected = len(all_track_targets)
    if all_compact_hits:
        compact_hits = np.concatenate(all_compact_hits, axis=0)  # (total_compact_hits, N_HIT_FEATURES)
        compact_hit_times = np.concatenate(all_compact_hit_times, axis=0).astype(np.float32)
    else:
        compact_hits = np.zeros((0, N_HIT_FEATURES), dtype=np.float32)
        compact_hit_times = np.zeros(0, dtype=np.float32)

    # Build sequential CSR indices into compact array
    if n_selected > 0:
        track_targets = np.stack(all_track_targets, axis=0)  # (N, 5)
        track_hit_offsets = np.zeros(n_selected + 1, dtype=np.int32)
        np.cumsum(all_track_lengths, out=track_hit_offsets[1:])
        # CSR values are just sequential indices into compact_hits
        track_hit_indices = np.arange(len(compact_hits), dtype=np.int32)
        track_event_idx = np.array(all_track_event_idx, dtype=np.int32)
    else:
        track_targets = np.zeros((0, 5), dtype=np.float32)
        track_hit_indices = np.zeros(0, dtype=np.int32)
        track_hit_offsets = np.zeros(1, dtype=np.int32)
        track_event_idx = np.zeros(0, dtype=np.int32)

    # Save compact hits + parallel truth-time sidecar at shard level. The
    # 12-column ``hits.npy`` schema is unchanged — ``hit_times.npy`` is a
    # separate (total_compact_hits,) float32 array used only as the
    # encoder sort key.
    _safe_save(output_dir / "hits.npy", compact_hits)
    _safe_save(output_dir / "hit_times.npy", compact_hit_times)

    # Save selected tracks (track_targets last so its presence alone is NOT
    # treated as completion — the real completion marker is written at end).
    _safe_save(sel_dir / "track_hit_indices.npy", track_hit_indices)
    _safe_save(sel_dir / "track_hit_offsets.npy", track_hit_offsets)
    _safe_save(sel_dir / "track_event_idx.npy", track_event_idx)

    # Always save particle IDs
    track_particle_ids = np.array(all_track_particle_ids, dtype=np.int64) if all_track_particle_ids else np.zeros(0, dtype=np.int64)
    _safe_save(sel_dir / "track_particle_ids.npy", track_particle_ids)

    # Save ACTS reco if computed
    if all_track_acts_reco:
        acts_reco_arr = np.stack(all_track_acts_reco, axis=0)  # (N, 5)
        acts_dm_arr = np.array(all_track_acts_dm, dtype=bool)  # (N,)
        _safe_save(sel_dir / "acts_reco.npy", acts_reco_arr)
        _safe_save(sel_dir / "acts_dm_mask.npy", acts_dm_arr)

    # Save track metadata (pt, vertex_primary) for tight selection filtering
    if all_track_meta:
        track_meta = np.stack(all_track_meta, axis=0)  # (N, 2): [pt, vertex_primary]
    else:
        track_meta = np.zeros((0, 2), dtype=np.float32)
    _safe_save(sel_dir / "track_meta.npy", track_meta)

    # Save track_targets.npy and the _complete sentinel last.  An atomic(-ish)
    # completion check can then use either, but _complete is the primary.
    _safe_save(sel_dir / "track_targets.npy", track_targets)
    (output_dir / _COMPLETE_MARKER).touch()

    total_compact_hits = len(compact_hits)

    return {
        "n_events": n_events,
        "n_selected_tracks": n_selected,
        "n_selected_hits": total_compact_hits,
        "n_acts_double_matched": sum(all_track_acts_dm) if all_track_acts_dm else 0,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Compact preprocess ColliderML to memmap format (selected track hits only)"
    )
    parser.add_argument("--data-dir", type=str,
                        default=None, required=True,
                        help="Root directory containing parquet subdirectories")
    parser.add_argument("--output-dir", type=str,
                        default=None, required=True,
                        help="Output directory for preprocessed shards")
    parser.add_argument("--num-shards", type=int, default=-1,
                        help="Number of shards to process (-1 for all)")
    parser.add_argument("--selection-file", type=str, default=None,
                        help="Path to multi-variant selection YAML "
                             "(default: selection_p200_datasets.yaml in the package root).")
    parser.add_argument("--selection-variant", type=str, default=None,
                        help="Named variant to load from a multi-variant YAML "
                             "(e.g. 'loose', 'core', 'core_kf_matched', 'core_kf_hits')")
    parser.add_argument("--selection", type=str, default=None,
                        help="JSON string of selection overrides")
    parser.add_argument("--particles-subdir", type=str,
                        default="ttbar_pu200_particles_recorded_only")
    parser.add_argument("--hits-subdir", type=str,
                        default="ttbar_pu200_tracker_hits")
    parser.add_argument("--num-workers", type=int, default=1,
                        help="Number of parallel workers (default: 1 = sequential)")
    parser.add_argument("--tracks-subdir", type=str,
                        default="ttbar_pu200_tracks",
                        help="Subdirectory name for ACTS reconstructed tracks parquet files")
    parser.add_argument("--no-acts", action="store_true",
                        help="Disable ACTS augmentation (enabled by default)")
    parser.add_argument("--no-split", action="store_true",
                        help="Skip building split.json after preprocessing. "
                             "By default a deterministic 90/5/5 train/val/test "
                             "split is written next to the shards.")
    parser.add_argument("--train-frac", type=float, default=0.9,
                        help="Train fraction for the auto-built split (default 0.90)")
    parser.add_argument("--val-frac", type=float, default=0.05,
                        help="Val fraction for the auto-built split (default 0.05)")
    parser.add_argument("--test-frac", type=float, default=0.05,
                        help="Test fraction for the auto-built split (default 0.05)")
    args = parser.parse_args()
    augment_acts = not args.no_acts

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load selection
    sel_file = Path(args.selection_file) if args.selection_file else _SELECTION_FILE
    if not args.selection_variant:
        sys.exit("ERROR: --selection-variant is required (e.g. 'core' or 'core_kf_matched').")
    selection = load_selection_variant(sel_file, args.selection_variant)
    if args.selection:
        selection.update(json.loads(args.selection))

    # Validate ACTS requirements
    if selection.get("require_acts_dm") and not augment_acts:
        sys.exit("ERROR: require_acts_dm=true requires ACTS augmentation (remove --no-acts)")
    if selection.get("use_acts_hits_only") and not augment_acts:
        sys.exit("ERROR: use_acts_hits_only=true requires ACTS augmentation (remove --no-acts)")

    particles_dir = data_dir / args.particles_subdir
    hits_dir = data_dir / args.hits_subdir

    # Support both flat (*.parquet) and nested HuggingFace dataset layouts
    particle_files = sorted(particles_dir.glob("*.parquet"))
    if not particle_files:
        particle_files = sorted(particles_dir.rglob("*.parquet"))
    hits_files = sorted(hits_dir.glob("*.parquet"))
    if not hits_files:
        hits_files = sorted(hits_dir.rglob("*.parquet"))

    # Match by name
    pf_by_name = {f.name: f for f in particle_files}
    hf_by_name = {f.name: f for f in hits_files}

    tf_by_name: dict[str, Path] = {}
    if augment_acts:
        tracks_dir = data_dir / args.tracks_subdir
        tracks_files_list = sorted(tracks_dir.glob("*.parquet"))
        if not tracks_files_list:
            tracks_files_list = sorted(tracks_dir.rglob("*.parquet"))
        tf_by_name = {f.name: f for f in tracks_files_list}
        if not tf_by_name:
            raise FileNotFoundError(
                f"No tracks parquet files found under {tracks_dir}. "
                "Check --tracks-subdir (e.g. 'ttbar_pu200_tracks')."
            )
        print(f"Found {len(tf_by_name)} tracks parquet files for ACTS augmentation")

    common = sorted(set(pf_by_name) & set(hf_by_name))
    if not common:
        raise FileNotFoundError(
            f"No matching shards in {particles_dir} and {hits_dir}.\n"
            f"  Found {len(particle_files)} particle files, {len(hits_files)} hit files."
        )

    if args.num_shards > 0:
        common = common[: args.num_shards]

    print(f"Processing {len(common)} shards (compact mode — selected track hits only)")
    print(f"Selection: {selection}")
    print(f"Output: {output_dir}")
    print(f"Workers: {args.num_workers}")

    totals = {
        "n_events": 0,
        "n_selected_tracks": 0,
        "n_selected_hits": 0,
        "n_acts_double_matched": 0,
    }

    # Backfill `_complete` markers for shards written by the old version of
    # this script (which did not write the sentinel).  A shard is considered
    # "legacy-complete" if it has a full, well-formed set of output files.
    # This keeps the new skip-check compatible with older output directories.
    def _legacy_is_complete(shard_out: Path) -> bool:
        sel_d = shard_out / "selected_tracks"
        required = [
            shard_out / "hits.npy",
            shard_out / "hit_times.npy",  # v2 sidecar — required by data.py
            sel_d / "track_targets.npy",
            sel_d / "track_hit_indices.npy",
            sel_d / "track_hit_offsets.npy",
            sel_d / "track_event_idx.npy",
            sel_d / "track_particle_ids.npy",
            sel_d / "track_meta.npy",
        ]
        if augment_acts:
            required += [sel_d / "acts_reco.npy", sel_d / "acts_dm_mask.npy"]
        return all(p.exists() for p in required)

    n_backfilled = 0
    n_v1_invalidated = 0
    for shard_name in common:
        shard_idx = int(shard_name.split("-")[1])
        shard_out = output_dir / f"shard_{shard_idx:04d}"
        marker = shard_out / _COMPLETE_MARKER
        if marker.exists():
            # v2 invariant: every complete shard must carry hit_times.npy.
            # If a marker is present without the sidecar, this is a v1
            # (s-sorted) shard left over from the previous preprocessor —
            # drop the marker so the shard is re-processed under v2 rules.
            if not (shard_out / "hit_times.npy").exists():
                marker.unlink()
                n_v1_invalidated += 1
            else:
                continue
        if shard_out.exists() and _legacy_is_complete(shard_out):
            marker.touch()
            n_backfilled += 1
    if n_backfilled:
        print(f"Backfilled {_COMPLETE_MARKER} marker for {n_backfilled} legacy-complete shards")
    if n_v1_invalidated:
        print(
            f"Invalidated {_COMPLETE_MARKER} marker on {n_v1_invalidated} v1 "
            "(pre-time-sort) shards — these will be re-processed."
        )

    # Build job list, skipping already-completed shards.
    # A shard is considered complete iff its `_complete` sentinel file exists.
    # This sentinel is written atomically at the very end of process_shard(),
    # so partially-written shards (e.g. from an EOS-crashed previous run) are
    # automatically re-processed.  Any stale partial files are silently
    # overwritten by the new run.
    jobs = []
    n_skipped = 0
    for shard_name in common:
        shard_idx = int(shard_name.split("-")[1])
        shard_out = output_dir / f"shard_{shard_idx:04d}"
        if (shard_out / _COMPLETE_MARKER).exists():
            n_skipped += 1
            continue
        tracks_f = tf_by_name.get(shard_name) if augment_acts else None
        jobs.append((pf_by_name[shard_name], hf_by_name[shard_name], shard_out, selection, tracks_f))

    if n_skipped:
        print(f"Skipping {n_skipped} already-processed shards, {len(jobs)} remaining")

    t0 = time.time()

    if args.num_workers <= 1:
        for i, (pf, hf, shard_out, sel_cfg, tf) in enumerate(tqdm(jobs, desc="Processing shards")):
            stats = process_shard(pf, hf, shard_out, sel_cfg, tf)
            for k in totals:
                totals[k] += stats[k]
    else:
        with ProcessPoolExecutor(max_workers=args.num_workers) as pool:
            futures = {
                pool.submit(process_shard, pf, hf, shard_out, sel_cfg, tf): i
                for i, (pf, hf, shard_out, sel_cfg, tf) in enumerate(jobs)
            }
            with tqdm(total=len(futures), desc="Processing shards") as pbar:
                for future in as_completed(futures):
                    stats = future.result()
                    for k in totals:
                        totals[k] += stats[k]
                    pbar.update(1)

    elapsed = time.time() - t0

    # Write manifest
    manifest = {
        "num_shards": len(common),
        "format": "compact",
        "format_version": 2,  # v2 = adds per-shard hit_times.npy time-sort sidecar
        "sort_key": "time",   # was "s" in v1; truth time is linear in arc length
        "selection": selection,
        "hit_feature_names": HIT_FEATURE_NAMES,
        "target_names": TARGET_NAMES,
        "total_tracks": totals["n_selected_tracks"],
        "totals": totals,
        "processing_time_s": elapsed,
    }
    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Compact preprocessing complete in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"  Events:              {totals['n_events']:>12,}")
    print(f"  Selected tracks:     {totals['n_selected_tracks']:>12,}")
    print(f"  Selected hits:       {totals['n_selected_hits']:>12,}")
    print(f"  ACTS double-matched: {totals['n_acts_double_matched']:>12,}")
    print(f"  Output:              {output_dir}")
    print(f"  Manifest:            {output_dir / 'manifest.json'}")

    # Auto-create split.json (default 90/5/5) unless --no-split. If a split
    # file already exists (e.g. user pre-built it or a prior run wrote it),
    # we leave it alone so downstream training keeps the same split across
    # incremental re-runs.
    split_path = output_dir / "split.json"
    if args.no_split:
        print(f"\n--no-split passed; skipping split.json creation.")
    elif split_path.exists():
        print(f"\nSplit file already exists, leaving it untouched: {split_path}")
    else:
        from track_regression.scripts.create_split import create_split

        print(
            f"\nCreating default train/val/test split "
            f"({args.train_frac:.0%}/{args.val_frac:.0%}/{args.test_frac:.0%})..."
        )
        try:
            create_split(
                preprocessed_dir=output_dir,
                train_frac=args.train_frac,
                val_frac=args.val_frac,
                test_frac=args.test_frac,
            )
        except ValueError as e:
            # Fewer than 3 shards on disk — common for smoke tests with
            # --num-shards 1 or 2. Don't poison the exit status of an
            # otherwise-successful preprocessing run.
            print(f"WARNING: skipping split.json — {e}")


if __name__ == "__main__":
    main()
