#!/usr/bin/env python3
# ruff: noqa: TID252, PLR0915, C901
"""Preprocess the ``drift_beamspot`` campaign into the flat training format.

Differences from ``preprocess_colliderml_compact.py``, all forced by the new
campaign or measured to be worth it:

*Schema.*  ``tracker_hits`` carries a list-valued ``particle_ids`` per hit (a
merged cluster belongs to several particles) instead of a scalar
``particle_id``; ``particles.perigee_d0``/``perigee_z0`` are all-NULL so the
positional targets are re-derived by :func:`track_regression.perigee.truth_perigee`;
and ``detector`` is written as 255 for every strip volume, so it is rebuilt
from ``volume_id``.

*Selection.*  The d0/z0 window cuts are gone.  The drift beamspot is wider than
the windows in ``selection_p200_datasets.yaml``, so they truncated the targets
behind a hard wall instead of removing bad tracks.

*Layout.*  One flat store per split instead of 1000 shard directories:

    <out>/<split>/part_XXXX/{hits,hit_times,offsets,lengths,targets,...}.npy

Hits within a track are stored in ``--sort-key`` order (default ``geometry``); the
``hit_times`` sidecar is kept for diagnostics only.

``hits`` is contiguous per track, so a track is a slice and the legacy
``track_hit_indices`` (which was exactly ``arange``) is gone.  Tracks are
shuffled within each part at write time, which is what makes contiguous-block
sampling a valid substitute for a global shuffle at train time.  Splits are cut
at the *input shard* level so no event ever straddles train and val.

Everything is vectorised over a whole input shard; there is no per-event Python
loop, which is what makes 201.6 M events tractable.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from track_regression.perigee import truth_perigee  # noqa: E402

N_HIT_FEATURES = 12
HIT_FEATURE_NAMES = [
    "x", "y", "z", "r", "phi_hit", "theta_hit", "s",
    "volume_id", "layer_id", "surface_id", "detector", "eta_hit",
]
TARGET_NAMES = ["d0", "z0", "phi", "theta", "qop"]
MAX_HITS_HARD = 20

# ``detector`` is 255 (the uint8 "unset" sentinel) for every strip volume in
# this campaign and shifted to 6/7/8 for the pixel volumes.  volume_id is
# correct, so the feature is rebuilt from this map -- the one the previous
# production wrote.
VOLUME_TO_DETECTOR = {16: 0, 17: 1, 18: 2, 23: 3, 24: 4, 25: 5, 28: 6, 29: 7, 30: 8}

_SELECTION_FILE = Path(__file__).resolve().parent.parent / "selection_p200_datasets.yaml"


# ---------------------------------------------------------------------------
# arrow helpers
# ---------------------------------------------------------------------------

def _flat(col):
    """List column -> (flat values ndarray, per-row offsets int64)."""
    ca = col.combine_chunks()
    if hasattr(ca, "chunks"):
        ca = ca.chunk(0) if ca.num_chunks == 1 else ca.combine_chunks()
    off = np.asarray(ca.offsets, dtype=np.int64)
    vals = np.asarray(ca.values.to_numpy(zero_copy_only=False))
    return vals, off


def _rowid(off, n):
    """Row index for each flattened element."""
    return np.repeat(np.arange(len(off) - 1, dtype=np.int64), np.diff(off))[:n]


# ---------------------------------------------------------------------------
# one input shard -> selected tracks
# ---------------------------------------------------------------------------

def select_shard(pf: Path, hf: Path, tf: Path | None, sel: dict, augment_acts: bool,
                 ttf: Path | None = None, shf: Path | None = None):
    """Return the selected tracks of one parquet shard, fully vectorised."""
    ptab = pq.read_table(pf, columns=[
        "event_id", "particle_id", "charge", "px", "py", "pz",
        "vx", "vy", "vz", "primary", "vertex_primary"])
    n_events = ptab.num_rows
    ev_ids = ptab.column("event_id").to_numpy(zero_copy_only=False).astype(np.int64)

    p_pid, p_off = _flat(ptab.column("particle_id"))
    n_part = len(p_pid)
    if n_part == 0:
        return None
    p_ev = _rowid(p_off, n_part)
    g = lambda c: _flat(ptab.column(c))[0].astype(np.float64)          # noqa: E731
    q = g("charge")
    px, py, pz = g("px"), g("py"), g("pz")
    vx, vy, vz = g("vx"), g("vy"), g("vz")
    prim = _flat(ptab.column("primary"))[0].astype(bool)
    vprim = _flat(ptab.column("vertex_primary"))[0].astype(np.int32)

    d0, z0, phi, theta, qop = truth_perigee(vx, vy, vz, px, py, pz, q, Bz=float(sel.get("_bz", 2.0)))
    pt = np.hypot(px, py)
    with np.errstate(divide="ignore", invalid="ignore"):
        eta = -np.log(np.tan(np.clip(theta, 1e-8, np.pi - 1e-8) / 2.0))

    base = np.isfinite(d0) & np.isfinite(z0) & np.isfinite(theta) & (q != 0) & (pt > 0)
    if sel["primary"]:
        base &= prim
    if sel.get("hard_scatter"):
        base &= vprim == 1
    base &= pt >= sel["pt_min"]
    if sel.get("pt_max") is not None:
        base &= pt <= float(sel["pt_max"])
    base &= (eta >= sel["eta_min"]) & (eta <= sel["eta_max"])
    # d0/z0 windows are omitted by default (see module docstring); --apply-d0z0-windows
    # puts them back, e.g. to reproduce the legacy selection on new data.
    if sel.get("_apply_d0z0_windows"):
        base &= (d0 >= sel["d0_min"]) & (d0 <= sel["d0_max"])
        base &= (z0 >= sel["z0_min"]) & (z0 <= sel["z0_max"])

    # ---- hits ----
    need_tt = sel.get("_sort_key") == "true_time"
    htab = pq.read_table(hf, columns=[
        "event_id", "particle_ids", "x", "y", "z",
        "volume_id", "layer_id", "surface_id", "time"] + (["simhit_ids"] if need_tt else []))
    hx, h_off = _flat(htab.column("x"))
    n_hits = len(hx)
    if n_hits == 0:
        return None
    hx = hx.astype(np.float64)
    hy = _flat(htab.column("y"))[0].astype(np.float64)
    hz = _flat(htab.column("z"))[0].astype(np.float64)
    hvol = _flat(htab.column("volume_id"))[0].astype(np.int32)
    hlay = _flat(htab.column("layer_id"))[0].astype(np.int32)
    hsur = _flat(htab.column("surface_id"))[0].astype(np.int64)
    htime = _flat(htab.column("time"))[0].astype(np.float32)
    # The particles and tracker_hits tables are NOT guaranteed to hold their
    # events in the same row order (single_muon_uniform shard 0: 4.1 % of rows
    # differ; the fixed-pT shards 0.4-9 %).  Joining by row index paired the
    # hits of event B with the targets of event A for those rows -- ~4 % of the
    # muon-gun tracks in the 2026-08-21 stores were mislabelled.  Map every
    # hit-table row to the particles-table row with the SAME event_id value.
    h_ev_ids = htab.column("event_id").to_numpy(zero_copy_only=False).astype(np.int64)
    p_order = np.argsort(ev_ids, kind="stable")
    pos_e = np.searchsorted(ev_ids[p_order], h_ev_ids)
    pos_e = np.clip(pos_e, 0, len(ev_ids) - 1)
    row_ok = ev_ids[p_order][pos_e] == h_ev_ids
    h_row2p = np.where(row_ok, p_order[pos_e], -1)     # -1: event has no particles row
    h_ev = h_row2p[_rowid(h_off, n_hits)]              # particles-table row per hit

    # rebuild detector from volume_id
    hdet = np.full(n_hits, -1, np.int32)
    for v, dcode in VOLUME_TO_DETECTOR.items():
        hdet[hvol == v] = dcode

    # list<list<uint64>> -> (hit index, particle id) pairs
    inner = pc.list_flatten(htab.column("particle_ids"))     # list<uint64> per hit
    pair_pid, pair_off = _flat(inner)
    pair_hit = _rowid(pair_off, len(pair_pid))

    # ---- ACTS reference order: tracker_simhits.true_time per (hit, particle) pair ----
    # `simhit_ids` is aligned with `particle_ids` (one sim hit per contributing particle)
    # and indexes the event's sim-hit list positionally (hit_sorting_study.truth_time_table).
    pair_tt = None
    if need_tt:
        if shf is None:
            raise FileNotFoundError(f"{hf}: --sort-key true_time needs the tracker_simhits table next to it")
        sid_vals, sid_off = _flat(pc.list_flatten(htab.column("simhit_ids")))
        if len(sid_vals) != len(pair_pid) or not np.array_equal(sid_off, pair_off):
            raise RuntimeError(f"{hf.name}: particle_ids and simhit_ids are not aligned")
        stab = pq.read_table(shf, columns=["event_id", "true_time"])
        s_tt, s_off = _flat(stab.column("true_time"))
        s_ev_ids = stab.column("event_id").to_numpy(zero_copy_only=False).astype(np.int64)
        # hits-table row -> simhits-table row of the same event_id
        s_order = np.argsort(s_ev_ids, kind="stable")
        pos_s = np.clip(np.searchsorted(s_ev_ids[s_order], h_ev_ids), 0, len(s_ev_ids) - 1)
        s_ok = s_ev_ids[s_order][pos_s] == h_ev_ids
        h_row2s = np.where(s_ok, s_order[pos_s], -1)
        srow = h_row2s[_rowid(h_off, n_hits)][pair_hit]           # simhits row per pair
        sidx = sid_vals.astype(np.int64)
        valid = (srow >= 0) & (sidx >= 0) & (sidx < np.diff(s_off)[np.clip(srow, 0, len(s_off) - 2)])
        flat_idx = np.where(valid, s_off[np.clip(srow, 0, len(s_off) - 2)] + sidx, 0)
        pair_tt = np.where(valid, s_tt[np.clip(flat_idx, 0, len(s_tt) - 1)].astype(np.float64), np.nan)

    # ---- match hits to particles on (event, particle_id) ----
    # A dense rank keeps the composite key inside int64 exactly.
    uniq, inv = np.unique(np.concatenate([p_pid, pair_pid]), return_inverse=True)
    if len(uniq) >= 2**31:
        raise RuntimeError("particle-id rank overflows int32 packing")
    p_rank = inv[:n_part].astype(np.int64)
    pair_rank = inv[n_part:].astype(np.int64)
    p_key = (p_ev << 32) | p_rank
    pair_key = (h_ev[pair_hit] << 32) | pair_rank

    order = np.argsort(p_key, kind="stable")
    skey = p_key[order]
    pos = np.searchsorted(skey, pair_key)
    ok = pos < len(skey)
    pos_c = np.where(ok, pos, 0)
    ok &= skey[pos_c] == pair_key
    pidx = order[pos_c[ok]]        # -> index into particles
    hidx = pair_hit[ok]            # -> index into hits

    nh = np.bincount(pidx, minlength=n_part)
    mask = base & (nh >= sel["min_hits"]) & (nh <= sel["max_hits"])

    # ---- ACTS reco, matched on majority_particle_id ----
    acts_reco = acts_dm = None
    if augment_acts and tf is not None:
        ttab = pq.read_table(tf, columns=[
            "event_id", "d0", "z0", "phi", "theta", "qop",
            "majority_particle_id", "hit_ids"])
        t_maj, t_off = _flat(ttab.column("majority_particle_id"))
        if len(t_maj):
            t_row = _rowid(t_off, len(t_maj))
            t_ev_ids = ttab.column("event_id").to_numpy(zero_copy_only=False).astype(np.int64)
            ev_to_local = {int(e): i for i, e in enumerate(ev_ids)}
            t_ev = np.array([ev_to_local.get(int(t_ev_ids[r]), -1) for r in t_row], np.int64)
            keep_t = t_ev >= 0
            tr = np.searchsorted(uniq, t_maj)
            tr_ok = keep_t & (tr < len(uniq)) & (uniq[np.clip(tr, 0, len(uniq)-1)] == t_maj)
            t_key = np.where(tr_ok, (t_ev << 32) | tr.astype(np.int64), -1)
            tpos = np.searchsorted(skey, t_key)
            tok = tr_ok & (tpos < len(skey))
            tpos_c = np.where(tok, tpos, 0)
            tok &= skey[tpos_c] == t_key
            tgt_particle = order[tpos_c[tok]]
            acts_reco = np.full((n_part, 5), np.nan, np.float32)
            for j, name in enumerate(TARGET_NAMES):
                acts_reco[tgt_particle, j] = _flat(ttab.column(name))[0][tok].astype(np.float32)
            # double match: purity and efficiency > 0.75 on hit-index sets
            acts_dm = np.zeros(n_part, bool)
            hid_vals, hid_off = _flat(pc.list_flatten(ttab.column("hit_ids")))
            hid_track = _rowid(hid_off, len(hid_vals))       # -> index into t_maj
            n_reco = np.diff(hid_off)
            # truth hit sets, as a sorted (particle, hit) pair list
            po = np.lexsort((hidx, pidx))
            sp, sh = pidx[po], hidx[po]
            tstart = np.searchsorted(sp, np.arange(n_part))
            tend = np.searchsorted(sp, np.arange(n_part), side="right")
            sel_t = np.nonzero(tok)[0]
            for k, tk in enumerate(sel_t):
                pi = tgt_particle[k]
                a, b = tstart[pi], tend[pi]
                if b <= a:
                    continue
                truth = sh[a:b]
                lo = hid_off[tk]
                reco_local = hid_vals[lo:lo + n_reco[tk]].astype(np.int64) + h_off[t_ev[tk]]
                nmaj = np.intersect1d(truth, reco_local, assume_unique=False).size
                if nmaj / max(n_reco[tk], 1) > 0.75 and nmaj / (b - a) > 0.75:
                    acts_dm[pi] = True

    # ---- truth-tracking KF (``truth_tracks`` table), matched the same way ----
    # Done here, per shard, because event ids restart at 0 in every runs/<N>/
    # directory, so a post-hoc (event_id, particle_id) join over a whole store
    # is ambiguous (scripts/extract_truth_kf.py).  Written as the same
    # ``truth_kf_reco.npy`` side-car that fast_rms_eval / kf_baselines read.
    truth_kf = None
    if augment_acts and ttf is not None:
        ttab2 = pq.read_table(ttf, columns=["event_id", "d0", "z0", "phi", "theta", "qop",
                                            "majority_particle_id"])
        t_maj2, t_off2 = _flat(ttab2.column("majority_particle_id"))
        truth_kf = np.full((n_part, 5), np.nan, np.float32)
        if len(t_maj2):
            t_row2 = _rowid(t_off2, len(t_maj2))
            t_ev_ids2 = ttab2.column("event_id").to_numpy(zero_copy_only=False).astype(np.int64)
            ev_to_local2 = {int(e): i for i, e in enumerate(ev_ids)}
            t_ev2 = np.array([ev_to_local2.get(int(t_ev_ids2[r]), -1) for r in t_row2], np.int64)
            tr2 = np.searchsorted(uniq, t_maj2)
            tr_ok2 = (t_ev2 >= 0) & (tr2 < len(uniq)) & (uniq[np.clip(tr2, 0, len(uniq)-1)] == t_maj2)
            t_key2 = np.where(tr_ok2, (t_ev2 << 32) | tr2.astype(np.int64), -1)
            tpos2 = np.searchsorted(skey, t_key2)
            tok2 = tr_ok2 & (tpos2 < len(skey))
            tpos2_c = np.where(tok2, tpos2, 0)
            tok2 &= skey[tpos2_c] == t_key2
            tgt2 = order[tpos2_c[tok2]]
            for j, name in enumerate(TARGET_NAMES):
                truth_kf[tgt2, j] = _flat(ttab2.column(name))[0][tok2].astype(np.float32)

    if sel.get("require_acts_dm"):
        mask &= acts_dm if acts_dm is not None else False
    if sel.get("use_acts_hits_only"):
        raise NotImplementedError("use_acts_hits_only is not supported by the flat writer")

    sel_idx = np.nonzero(mask)[0]
    if len(sel_idx) == 0:
        return None

    # ---- gather hits of selected tracks, sorted along the track, CSR ----
    # Sort key per hit within each track.  ``time`` is the legacy choice and is
    # only valid when tracker_hits.time is a real per-hit time; on the
    # drift_beamspot campaign it is broken (strip volumes carry time == 0 and the
    # pixel time is not referenced to the bunch crossing, see
    # BUGREPORT_drift_beamspot_hit_time.md), so the default is the radial
    # magnitude ``s = sqrt(x^2+y^2+z^2)`` — the legacy pre-time ordering.  The
    # packed collate/encoder consume the stored order verbatim, so this is THE
    # sequence order the model sees.
    keep = mask[pidx]
    keep_idx = np.nonzero(ok)[0][keep]          # pair positions of the kept (hit, particle) pairs
    pk, hk = pidx[keep], hidx[keep]
    sort_key = sel.get("_sort_key", "geometry")
    if sort_key == "time":
        o = np.lexsort((htime[hk], pk))
    elif sort_key == "true_time":
        tt = pair_tt[keep_idx]
        n_nan = int(np.isnan(tt).sum())
        if n_nan:
            print(f"  [true_time] {n_nan} of {len(tt)} selected hits without a sim-hit time -> placed last", flush=True)
        o = np.lexsort((np.where(np.isnan(tt), np.inf, tt), pk))
    elif sort_key == "s":
        o = np.lexsort((np.sqrt(hx ** 2 + hy ** 2 + hz ** 2)[hk], pk))
    elif sort_key == "geometry":
        # Detector order (track_regression.hit_sorting): pixel -> short strip ->
        # long strip, barrel before endcap, barrel by r, discs by z along the
        # track's own flight direction.  Truth-free; reproduces the ACTS
        # truth-time order on 100 % of muon-gun and 99.7 % of ttbar tracks
        # (docs/HIT_SORTING_ACTS_vs_radial.md).  The direction is read off each
        # track's own hits: sign(z at max r - z at min r).
        from track_regression.hit_sorting import geometry_keys
        r_pair = np.hypot(hx[hk], hy[hk])
        o_r = np.lexsort((r_pair, pk))
        pk_r = pk[o_r]
        first = np.r_[True, pk_r[1:] != pk_r[:-1]]
        last = np.r_[pk_r[1:] != pk_r[:-1], True]
        z_r = hz[hk][o_r]
        dir_track = np.where(z_r[last] >= z_r[first], 1.0, -1.0)
        dir_pair = np.empty(len(pk))
        dir_pair[o_r] = np.repeat(dir_track, np.diff(np.r_[np.nonzero(first)[0], len(pk_r)]))
        primary, secondary = geometry_keys(
            np.stack([hx[hk], hy[hk], hz[hk]], 1), hvol[hk], direction=dir_pair)
        o = np.lexsort((secondary, primary, pk))
    else:
        raise ValueError(f"unknown sort key {sort_key!r}")
    pk, hk = pk[o], hk[o]
    new_id = np.full(n_part, -1, np.int64)
    new_id[sel_idx] = np.arange(len(sel_idx))
    tid = new_id[pk]
    lens = np.bincount(tid, minlength=len(sel_idx)).astype(np.int32)
    offsets = np.zeros(len(sel_idx) + 1, np.int64)
    np.cumsum(lens, out=offsets[1:])

    r = np.hypot(hx[hk], hy[hk])
    ss = np.sqrt(hx[hk] ** 2 + hy[hk] ** 2 + hz[hk] ** 2)
    th = np.arccos(np.clip(hz[hk] / np.maximum(ss, 1e-12), -1.0, 1.0))
    et = np.clip(-np.log(np.tan(np.clip(th, 1e-8, np.pi - 1e-8) / 2.0)), -10.0, 10.0)
    H = np.empty((len(hk), N_HIT_FEATURES), np.float32)
    H[:, 0] = hx[hk]; H[:, 1] = hy[hk]; H[:, 2] = hz[hk]; H[:, 3] = r
    H[:, 4] = np.arctan2(hy[hk], hx[hk]); H[:, 5] = th; H[:, 6] = ss
    H[:, 7] = hvol[hk]; H[:, 8] = hlay[hk]; H[:, 9] = hsur[hk]
    H[:, 10] = hdet[hk]; H[:, 11] = et

    out = dict(
        hits=H,
        hit_times=htime[hk].astype(np.float32),
        offsets=offsets,
        lengths=lens,
        targets=np.stack([d0, z0, phi, theta, qop], 1)[sel_idx].astype(np.float32),
        track_meta=np.stack([pt, vprim.astype(np.float64)], 1)[sel_idx].astype(np.float32),
        particle_ids=p_pid[sel_idx].astype(np.int64),
        event_ids=ev_ids[p_ev[sel_idx]].astype(np.int64),
        n_events=n_events,
        n_particles=n_part,
        n_raw_hits=n_hits,
    )
    if acts_reco is not None:
        out["acts_reco"] = acts_reco[sel_idx]
        out["acts_dm"] = acts_dm[sel_idx]
    if truth_kf is not None:
        out["truth_kf_reco"] = truth_kf[sel_idx]
    return out


# ---------------------------------------------------------------------------
# one output part = one group of input shards
# ---------------------------------------------------------------------------

def write_part(args):
    part_dir, triples, sel, augment_acts, seed = args
    part_dir = Path(part_dir)
    if (part_dir / "_complete").exists():
        m = json.load(open(part_dir / "meta.json"))
        return m
    t0 = time.time()
    chunks, stats = [], dict(n_events=0, n_particles=0, n_raw_hits=0)
    for pf, hf, tf, *rest in triples:
        ttf = rest[0] if len(rest) > 0 else None
        shf = rest[1] if len(rest) > 1 else None
        try:
            c = select_shard(Path(pf), Path(hf), Path(tf) if tf else None, sel, augment_acts,
                             Path(ttf) if ttf else None, Path(shf) if shf else None)
        except Exception:
            return {"error": f"{pf}: {traceback.format_exc()}"}
        if c is None:
            continue
        for k in stats:
            stats[k] += c.pop(k)
        chunks.append(c)
    if not chunks:
        return {"error": f"{part_dir}: no tracks selected"}

    n_tracks = sum(len(c["lengths"]) for c in chunks)
    n_hits = sum(len(c["hits"]) for c in chunks)
    # Shuffle within the part.  This is what makes a contiguous block at train
    # time statistically the same draw as a global shuffle; measured on real
    # data, per-batch target means then match the CLT prediction to within 9%,
    # against 5.6x over-dispersion in natural event order.
    rng = np.random.default_rng(seed)
    order = rng.permutation(n_tracks)

    lens_all = np.concatenate([c["lengths"] for c in chunks])
    src_off = np.zeros(n_tracks + 1, np.int64)
    np.cumsum(lens_all, out=src_off[1:])
    # per-chunk base offsets into the concatenated hit stream
    chunk_of = np.repeat(np.arange(len(chunks)), [len(c["lengths"]) for c in chunks])
    local_of = np.concatenate([np.arange(len(c["lengths"])) for c in chunks])

    lens = lens_all[order].astype(np.int32)
    off = np.zeros(n_tracks + 1, np.int64)
    np.cumsum(lens, out=off[1:])

    part_dir.mkdir(parents=True, exist_ok=True)
    H = np.lib.format.open_memmap(part_dir / "hits.npy", mode="w+",
                                  dtype=np.float32, shape=(n_hits, N_HIT_FEATURES))
    T = np.lib.format.open_memmap(part_dir / "hit_times.npy", mode="w+",
                                  dtype=np.float32, shape=(n_hits,))
    for out_i in range(n_tracks):
        src = order[out_i]
        c = chunks[chunk_of[src]]
        li = local_of[src]
        a, b = int(c["offsets"][li]), int(c["offsets"][li + 1])
        d, e = int(off[out_i]), int(off[out_i + 1])
        H[d:e] = c["hits"][a:b]
        T[d:e] = c["hit_times"][a:b]
    H.flush(); T.flush(); del H, T

    def cat(key, dtype=None):
        a = np.concatenate([c[key] for c in chunks])[order]
        return a.astype(dtype) if dtype else a

    np.save(part_dir / "offsets.npy", off)
    np.save(part_dir / "lengths.npy", lens)
    np.save(part_dir / "targets.npy", cat("targets", np.float32))
    np.save(part_dir / "track_meta.npy", cat("track_meta", np.float32))
    np.save(part_dir / "track_particle_ids.npy", cat("particle_ids", np.int64))
    np.save(part_dir / "track_event_ids.npy", cat("event_ids", np.int64))
    if "acts_reco" in chunks[0]:
        np.save(part_dir / "acts_reco.npy", cat("acts_reco", np.float32))
        np.save(part_dir / "acts_dm.npy", cat("acts_dm", np.bool_))
    if any("truth_kf_reco" in c for c in chunks):
        for c in chunks:
            c.setdefault("truth_kf_reco", np.full((len(c["lengths"]), 5), np.nan, np.float32))
        np.save(part_dir / "truth_kf_reco.npy", cat("truth_kf_reco", np.float32))
    meta = dict(name=part_dir.name, n_tracks=int(n_tracks), n_hits=int(n_hits),
                seconds=round(time.time() - t0, 1), **stats)
    json.dump(meta, open(part_dir / "meta.json", "w"))
    (part_dir / "_complete").touch()
    return meta


# ---------------------------------------------------------------------------
# dataset discovery + driver
# ---------------------------------------------------------------------------

def discover(root: Path):
    """Return ``[(particles, tracker_hits, tracks), ...]`` for either layout."""
    flat = root / "parquet" / "truth" / "particles"
    if flat.is_dir():
        out = []
        for pf in sorted(flat.glob("*.parquet")):
            tag = re.search(r"events\d+-\d+\.parquet", pf.name).group(0)
            hf = next((root / "parquet" / "reco" / "tracker_hits").glob(f"*{tag}"), None)
            tf = next((root / "parquet" / "reco" / "tracks").glob(f"*{tag}"), None)
            if hf and tf:
                ttd = root / "parquet" / "reco" / "truth_tracks"
                ttf = next(ttd.glob(f"*{tag}"), None) if ttd.is_dir() else None
                shd = root / "parquet" / "truth" / "tracker_simhits"
                shf = next(shd.glob(f"*{tag}"), None) if shd.is_dir() else None
                out.append((str(pf), str(hf), str(tf), str(ttf) if ttf else None, str(shf) if shf else None))
        return out
    out = []
    runs = sorted((root / "runs").glob("*/"),
                  key=lambda p: int(p.name) if p.name.isdigit() else 1 << 30)
    for rd in runs:
        for pf in sorted((rd / "particles").glob("*.parquet")):
            tag = pf.name.replace("particles_", "")
            hf, tf = rd / "tracker_hits" / f"tracker_hits_{tag}", rd / "tracks" / f"tracks_{tag}"
            ttf = rd / "truth_tracks" / f"truth_tracks_{tag}"      # truth-tracking KF (new ttbar runs)
            shf = rd / "tracker_simhits" / f"tracker_simhits_{tag}"  # sim hits (true_time for --sort-key true_time)
            if hf.exists() and tf.exists():
                out.append((str(pf), str(hf), str(tf), str(ttf) if ttf.exists() else None,
                            str(shf) if shf.exists() else None))
    return out


def group(items, n_groups):
    n_groups = max(1, min(n_groups, len(items)))
    bounds = np.linspace(0, len(items), n_groups + 1).astype(int)
    return [items[a:b] for a, b in zip(bounds[:-1], bounds[1:]) if b > a]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True, help="dataset root (holds parquet/ or runs/)")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--selection-variant", default="core")
    ap.add_argument("--selection-file", default=str(_SELECTION_FILE))
    ap.add_argument("--hard-scatter", action="store_true",
                    help="require vertex_primary == 1 (a no-op on this campaign)")
    ap.add_argument("--shards-per-part", type=int, default=1,
                    help="input parquet shards merged into one output part")
    ap.add_argument("--num-workers", type=int, default=32)
    ap.add_argument("--no-acts", action="store_true")
    ap.add_argument("--train-frac", type=float, default=0.90)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit-shards", type=int, default=-1)
    ap.add_argument("--apply-d0z0-windows", action="store_true",
                    help="re-apply the variant's d0/z0 windows (off by default)")
    ap.add_argument("--sort-key", choices=("geometry", "s", "time", "true_time"), default="geometry",
                    help="per-track hit order stored on disk: 'geometry' = detector order "
                         "(default since 2026-08-25, user decision; hit_sorting.geometry_keys, "
                         "truth-free, matches the ACTS truth-time order on ~100 %% of tracks), "
                         "'s' = sqrt(x^2+y^2+z^2) from the origin (legacy radial ordering, "
                         "wrong on 13-15 %% of drift_beamspot tracks at large |z0|), "
                         "'time' = tracker_hits.time (broken on the drift_beamspot campaign)")
    ap.add_argument("--d0-window", type=float, default=None, help="override |d0| max [mm]")
    ap.add_argument("--bz", type=float, default=2.0,
                    help="solenoid field [T] for the vertex->perigee transport of the truth targets. The campaign-2 "
                         "stores up to 2026-08-27 used 2.0; the hits imply 3.0 (CLAUDE.md §4.8)")
    ap.add_argument("--pt-max", type=float, default=None,
                    help="upper pT cut [GeV] (e.g. 110 for ttbar: test/train where the muon gun trains)")
    ap.add_argument("--pt-min", type=float, default=None,
                    help="override the selection's pt_min [GeV] (e.g. 1.0 for a ttbar eval store without the untrained < 1 GeV domain)")
    ap.add_argument("--z0-window", type=float, default=None, help="override |z0| max [mm]")
    a = ap.parse_args()

    sel = yaml.safe_load(open(a.selection_file))[a.selection_variant]
    sel["_bz"] = float(a.bz)
    if a.pt_min is not None:
        sel["pt_min"] = float(a.pt_min)          # recorded in manifest/dataset_meta via `selection`
    if a.pt_max is not None:
        sel["pt_max"] = float(a.pt_max)
    sel = dict(sel)
    sel["hard_scatter"] = bool(a.hard_scatter)
    sel["max_hits"] = min(sel.get("max_hits", MAX_HITS_HARD), MAX_HITS_HARD)
    sel["_apply_d0z0_windows"] = bool(a.apply_d0z0_windows)
    sel["_sort_key"] = a.sort_key
    if a.d0_window is not None:
        sel["d0_min"], sel["d0_max"] = -a.d0_window, a.d0_window
    if a.z0_window is not None:
        sel["z0_min"], sel["z0_max"] = -a.z0_window, a.z0_window

    root, out = Path(a.data_dir), Path(a.output_dir)
    triples = discover(root)
    if a.limit_shards > 0:
        triples = triples[: a.limit_shards]
    if not triples:
        sys.exit(f"no parquet triples found under {root}")

    # Split at the INPUT SHARD level so no event straddles two splits.
    rng = np.random.default_rng(a.seed)
    n = len(triples)
    perm = rng.permutation(n)
    # Give val and test at least one shard each whenever there are enough to go
    # round; rounding alone leaves test empty on small datasets like ttbar.
    if n >= 3:
        n_va = max(1, int(round(a.val_frac * n)))
        n_te = max(1, int(round((1.0 - a.train_frac - a.val_frac) * n)))
        n_tr = n - n_va - n_te
        if n_tr < 1:
            n_tr, n_va, n_te = n - 2, 1, 1
    else:
        n_tr, n_va, n_te = n, 0, 0
    split_idx = {"train": perm[:n_tr], "val": perm[n_tr:n_tr + n_va],
                 "test": perm[n_tr + n_va:n_tr + n_va + n_te]}

    print(f"{root.name}: {len(triples)} input shards -> "
          + ", ".join(f"{k} {len(v)}" for k, v in split_idx.items())
          + f"   [variant={a.selection_variant}, d0/z0 windows: "
          + ("ON" if a.apply_d0z0_windows else "OFF") + "]", flush=True)

    out.mkdir(parents=True, exist_ok=True)
    jobs, job_split = [], []
    for sp, idxs in split_idx.items():
        if len(idxs) == 0:
            continue
        sel_triples = [triples[i] for i in idxs]
        n_parts = max(1, int(round(len(sel_triples) / a.shards_per_part)))
        for pi, grp in enumerate(group(sel_triples, n_parts)):
            jobs.append((str(out / sp / f"part_{pi:04d}"), grp, sel,
                         not a.no_acts, a.seed + 1000 * pi + hash(sp) % 997))
            job_split.append(sp)

    results = {k: [] for k in split_idx}
    errors = []
    t0 = time.time()
    done = 0
    with ProcessPoolExecutor(max_workers=a.num_workers) as ex:
        futs = {ex.submit(write_part, j): s for j, s in zip(jobs, job_split)}
        for f in as_completed(futs):
            r = f.result(); done += 1
            if "error" in r:
                errors.append(r["error"]); print("ERROR " + r["error"][:400], flush=True)
            else:
                results[futs[f]].append(r)
            if done % 10 == 0 or done == len(jobs):
                print(f"  {done}/{len(jobs)} parts  ({time.time()-t0:.0f}s)", flush=True)

    for sp, parts in results.items():
        if not parts:
            continue
        parts.sort(key=lambda m: m["name"])
        man = dict(layout="flat_csr", version=3, n_feat=N_HIT_FEATURES,
                   hit_feature_names=HIT_FEATURE_NAMES, target_names=TARGET_NAMES,
                   shuffled_within_part=True, hit_sort_key=a.sort_key, bz=float(a.bz),
                   parts=[{k: p[k] for k in ("name", "n_tracks", "n_hits")} for p in parts],
                   n_tracks=sum(p["n_tracks"] for p in parts),
                   n_hits=sum(p["n_hits"] for p in parts))
        json.dump(man, open(out / sp / "manifest.json", "w"), indent=1)

    meta = dict(
        source=str(root), selection_variant=a.selection_variant, selection=sel,
        d0_z0_windows_applied=bool(a.apply_d0z0_windows), detector_rebuilt_from_volume_id=True,
        hit_sort_key=a.sort_key, bz=float(a.bz),
        perigee_recomputed=True, seed=a.seed,
        splits={sp: dict(n_input_shards=len(split_idx[sp]),
                         n_tracks=sum(p["n_tracks"] for p in results[sp]),
                         n_hits=sum(p["n_hits"] for p in results[sp]),
                         n_events=sum(p["n_events"] for p in results[sp]),
                         n_particles=sum(p["n_particles"] for p in results[sp]))
                for sp in results if results[sp]},
        errors=errors, seconds=round(time.time() - t0, 1))
    json.dump(meta, open(out / "dataset_meta.json", "w"), indent=1)
    tot = sum(v["n_tracks"] for v in meta["splits"].values())
    print(f"DONE {root.name}: {tot:,} tracks in {time.time()-t0:.0f}s"
          + (f"  [{len(errors)} ERRORS]" if errors else ""), flush=True)


if __name__ == "__main__":
    main()
