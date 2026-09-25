#!/usr/bin/env python3
"""Preprocess one ColliderML ``drift_beamspot`` dataset into a flat track store.

Input: the parquet tables of one dataset, in either portal layout

    <data-dir>/parquet/{truth/particles, truth/tracker_simhits, reco/tracker_hits,
                        reco/tracks, reco/truth_tracks}/*.events<a>-<b>.parquet
    <data-dir>/runs/<N>/{particles, tracker_simhits, tracker_hits, tracks,
                         truth_tracks}/<table>_<tag>.parquet        (ttbar)

Output: one store per split, each a list of parts,

    <output-dir>/<split>/manifest.json
    <output-dir>/<split>/part_XXXX/
        hits.npy            (n_hits, 12) float32, the hit features below, CSR by track
        offsets.npy         (n_tracks + 1,) int64, hits of track i = hits[offsets[i]:offsets[i+1]]
        lengths.npy         (n_tracks,) int32
        targets.npy         (n_tracks, 5) float32, truth perigee (d0, z0, phi, theta, q/p)
        truth_kf_reco.npy   (n_tracks, 5) float32, truth-seeded Kalman-filter fit (NaN if none)
        acts_reco.npy       (n_tracks, 5) float32, ACTS CKF fit (NaN if none)
        acts_dm.npy         (n_tracks,) bool, CKF track double-matched to this particle
        track_meta.npy      (n_tracks, 2) float32, [pT, vertex_primary]
        track_event_ids.npy, track_particle_ids.npy   (n_tracks,) int64, back-references
    <output-dir>/dataset_meta.json

Selection: charged primary particles with pT >= --pt-min (and <= --pt-max),
|eta| <= 3, |d0| <= --d0-max, |z0| <= --z0-max and 6-20 hits.  The perigee
targets are computed from the production vertex and momentum by
:func:`track_regression.perigee.truth_perigee` in the solenoid field --bz.

Hits within a track are stored in --sort-key order, which is the sequence order
the model consumes: ``true_time`` = simulated hit time (the order ACTS uses for
truth tracks; needs the tracker_simhits table), ``geometry`` = truth-free
detector order (:func:`track_regression.hit_sorting.geometry_keys`).

Splits are cut at the input-shard level (no event straddles two splits) and
tracks are shuffled within each part, so a contiguous block of a part is a
random draw at training time.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from track_regression.perigee import truth_perigee  # noqa: E402

N_HIT_FEATURES = 12
HIT_FEATURE_NAMES = [
    "x", "y", "z", "r", "phi_hit", "theta_hit", "s",
    "volume_id", "layer_id", "surface_id", "detector", "eta_hit",
]
TARGET_NAMES = ["d0", "z0", "phi", "theta", "qop"]
MIN_HITS, MAX_HITS = 6, 20
ETA_MAX = 3.0

# The ``detector`` column of tracker_hits is not filled for the strip volumes,
# so the feature is rebuilt from volume_id.
VOLUME_TO_DETECTOR = {16: 0, 17: 1, 18: 2, 23: 3, 24: 4, 25: 5, 28: 6, 29: 7, 30: 8}

SPLITS = ("train", "val", "test")


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


def _event_ids(tab):
    return tab.column("event_id").to_numpy(zero_copy_only=False).astype(np.int64)


def _rows_of_events(ev_ids, query):
    """Row of ``ev_ids`` holding each event id in ``query`` (-1 if absent).

    The tables of one shard do NOT store their events in the same row order, so
    they must be joined on the event_id VALUE; a join by row index silently pairs
    the hits of one event with the particles of another.
    """
    order = np.argsort(ev_ids, kind="stable")
    pos = np.clip(np.searchsorted(ev_ids[order], query), 0, len(ev_ids) - 1)
    return np.where(ev_ids[order][pos] == query, order[pos], -1)


# ---------------------------------------------------------------------------
# one input shard -> selected tracks
# ---------------------------------------------------------------------------

def select_shard(pf: Path, hf: Path, tf: Path, ttf: Path | None, shf: Path | None, cfg: dict):
    """Return the selected tracks of one parquet shard, fully vectorised."""
    ptab = pq.read_table(pf, columns=[
        "event_id", "particle_id", "charge", "px", "py", "pz",
        "vx", "vy", "vz", "primary", "vertex_primary"])
    n_events = ptab.num_rows
    ev_ids = _event_ids(ptab)

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

    d0, z0, phi, theta, qop = truth_perigee(vx, vy, vz, px, py, pz, q, Bz=cfg["bz"])
    pt = np.hypot(px, py)
    with np.errstate(divide="ignore", invalid="ignore"):
        eta = -np.log(np.tan(np.clip(theta, 1e-8, np.pi - 1e-8) / 2.0))

    base = np.isfinite(d0) & np.isfinite(z0) & np.isfinite(theta) & (q != 0) & (pt > 0) & prim
    base &= pt >= cfg["pt_min"]
    if cfg["pt_max"] is not None:
        base &= pt <= cfg["pt_max"]
    base &= np.abs(eta) <= ETA_MAX
    base &= (np.abs(d0) <= cfg["d0_max"]) & (np.abs(z0) <= cfg["z0_max"])

    # ---- hits ----
    need_tt = cfg["sort_key"] == "true_time"
    htab = pq.read_table(hf, columns=[
        "event_id", "particle_ids", "x", "y", "z",
        "volume_id", "layer_id", "surface_id"] + (["simhit_ids"] if need_tt else []))
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
    h_ev_ids = _event_ids(htab)
    h_ev = _rows_of_events(ev_ids, h_ev_ids)[_rowid(h_off, n_hits)]   # particles-table row per hit

    hdet = np.full(n_hits, -1, np.int32)
    for v, dcode in VOLUME_TO_DETECTOR.items():
        hdet[hvol == v] = dcode

    # list<list<uint64>> -> (hit index, particle id) pairs; a merged cluster
    # belongs to several particles.
    pair_pid, pair_off = _flat(pc.list_flatten(htab.column("particle_ids")))
    pair_hit = _rowid(pair_off, len(pair_pid))

    # ---- simulated time per (hit, particle) pair ----
    # `simhit_ids` is aligned with `particle_ids` (one sim hit per contributing
    # particle) and indexes the event's list in the tracker_simhits table.
    pair_tt = None
    if need_tt:
        if shf is None:
            raise FileNotFoundError(f"{hf}: --sort-key true_time needs the tracker_simhits table")
        sid_vals, sid_off = _flat(pc.list_flatten(htab.column("simhit_ids")))
        if len(sid_vals) != len(pair_pid) or not np.array_equal(sid_off, pair_off):
            raise RuntimeError(f"{hf.name}: particle_ids and simhit_ids are not aligned")
        stab = pq.read_table(shf, columns=["event_id", "true_time"])
        s_tt, s_off = _flat(stab.column("true_time"))
        srow = _rows_of_events(_event_ids(stab), h_ev_ids)[_rowid(h_off, n_hits)][pair_hit]
        sidx = sid_vals.astype(np.int64)
        srow_c = np.clip(srow, 0, len(s_off) - 2)
        valid = (srow >= 0) & (sidx >= 0) & (sidx < np.diff(s_off)[srow_c])
        flat_idx = np.where(valid, s_off[srow_c] + sidx, 0)
        pair_tt = np.where(valid, s_tt[np.clip(flat_idx, 0, len(s_tt) - 1)].astype(np.float64), np.nan)

    # ---- match hits to particles on (event, particle_id) ----
    # A dense rank keeps the composite key inside int64 exactly.
    uniq, inv = np.unique(np.concatenate([p_pid, pair_pid]), return_inverse=True)
    if len(uniq) >= 2**31:
        raise RuntimeError("particle-id rank overflows int32 packing")
    p_key = (p_ev << 32) | inv[:n_part].astype(np.int64)
    pair_key = (h_ev[pair_hit] << 32) | inv[n_part:].astype(np.int64)

    order = np.argsort(p_key, kind="stable")
    skey = p_key[order]

    def lookup(keys, valid):
        """Particle index for each (event row << 32 | particle rank) key, and a found-mask."""
        pos = np.searchsorted(skey, keys)
        ok = valid & (pos < len(skey))
        pos_c = np.where(ok, pos, 0)
        ok &= skey[pos_c] == keys
        return order[pos_c[ok]], ok

    pidx, ok = lookup(pair_key, np.ones(len(pair_key), bool))   # pidx -> particles
    hidx = pair_hit[ok]                                          # hidx -> hits

    nh = np.bincount(pidx, minlength=n_part)
    mask = base & (nh >= MIN_HITS) & (nh <= MAX_HITS)

    # ---- reconstructed tracks, matched to particles on majority_particle_id ----
    def match_reco(path, extra_cols=()):
        tab = pq.read_table(path, columns=["event_id", *TARGET_NAMES, "majority_particle_id", *extra_cols])
        maj, t_off = _flat(tab.column("majority_particle_id"))
        t_ev = _rows_of_events(ev_ids, _event_ids(tab))[_rowid(t_off, len(maj))]
        tr = np.searchsorted(uniq, maj)
        tr_ok = (t_ev >= 0) & (tr < len(uniq)) & (uniq[np.clip(tr, 0, len(uniq) - 1)] == maj)
        tgt, tok = lookup(np.where(tr_ok, (t_ev << 32) | tr.astype(np.int64), -1), tr_ok)
        reco = np.full((n_part, 5), np.nan, np.float32)
        for j, name in enumerate(TARGET_NAMES):
            reco[tgt, j] = _flat(tab.column(name))[0][tok].astype(np.float32)
        return tab, t_ev, tgt, tok, reco

    # ACTS CKF (`tracks` table) + double match: purity and efficiency > 0.75 on
    # the hit sets.  The evaluation compares on the double-matched subset.
    ttab, t_ev, tgt_particle, tok, acts_reco = match_reco(tf, ("hit_ids",))
    acts_dm = np.zeros(n_part, bool)
    hid_vals, hid_off = _flat(pc.list_flatten(ttab.column("hit_ids")))
    n_reco = np.diff(hid_off)
    po = np.lexsort((hidx, pidx))                  # truth hit sets as sorted (particle, hit) pairs
    sp, sh = pidx[po], hidx[po]
    tstart = np.searchsorted(sp, np.arange(n_part))
    tend = np.searchsorted(sp, np.arange(n_part), side="right")
    for k, tk in enumerate(np.nonzero(tok)[0]):
        pi = tgt_particle[k]
        a, b = tstart[pi], tend[pi]
        if b <= a:
            continue
        lo = hid_off[tk]
        # hit_ids index the event's hit list.  Its start is taken at
        # h_off[particles-table row], as in the stores used for the paper; this
        # differs from the hits-table row for the ~1 % of events that the two
        # tables order differently, and those tracks then fail the double match
        # (the evaluation subset is ~1 % smaller than with the hits-table row).
        reco_local = hid_vals[lo:lo + n_reco[tk]].astype(np.int64) + h_off[t_ev[tk]]
        nmaj = np.intersect1d(sh[a:b], reco_local).size
        if nmaj / max(n_reco[tk], 1) > 0.75 and nmaj / (b - a) > 0.75:
            acts_dm[pi] = True

    # Truth-seeded Kalman filter (`truth_tracks` table): the reference fit.
    truth_kf = match_reco(ttf)[4] if ttf is not None else np.full((n_part, 5), np.nan, np.float32)

    sel_idx = np.nonzero(mask)[0]
    if len(sel_idx) == 0:
        return None

    # ---- gather the hits of the selected tracks, sorted along the track ----
    keep = mask[pidx]
    keep_idx = np.nonzero(ok)[0][keep]          # pair positions of the kept (hit, particle) pairs
    pk, hk = pidx[keep], hidx[keep]
    if cfg["sort_key"] == "true_time":
        tt = pair_tt[keep_idx]
        n_nan = int(np.isnan(tt).sum())
        if n_nan:
            print(f"  [true_time] {n_nan} of {len(tt)} selected hits without a sim-hit time -> placed last", flush=True)
        o = np.lexsort((np.where(np.isnan(tt), np.inf, tt), pk))
    else:  # geometry
        # The flight direction along z is read off each track's own hits:
        # sign(z at max r - z at min r).
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
    pk, hk = pk[o], hk[o]
    new_id = np.full(n_part, -1, np.int64)
    new_id[sel_idx] = np.arange(len(sel_idx))
    lens = np.bincount(new_id[pk], minlength=len(sel_idx)).astype(np.int32)

    r = np.hypot(hx[hk], hy[hk])
    ss = np.sqrt(hx[hk] ** 2 + hy[hk] ** 2 + hz[hk] ** 2)
    th = np.arccos(np.clip(hz[hk] / np.maximum(ss, 1e-12), -1.0, 1.0))
    et = np.clip(-np.log(np.tan(np.clip(th, 1e-8, np.pi - 1e-8) / 2.0)), -10.0, 10.0)
    H = np.empty((len(hk), N_HIT_FEATURES), np.float32)
    H[:, 0] = hx[hk]; H[:, 1] = hy[hk]; H[:, 2] = hz[hk]; H[:, 3] = r
    H[:, 4] = np.arctan2(hy[hk], hx[hk]); H[:, 5] = th; H[:, 6] = ss
    H[:, 7] = hvol[hk]; H[:, 8] = hlay[hk]; H[:, 9] = hsur[hk]
    H[:, 10] = hdet[hk]; H[:, 11] = et

    return dict(
        hits=H,
        lengths=lens,
        targets=np.stack([d0, z0, phi, theta, qop], 1)[sel_idx].astype(np.float32),
        track_meta=np.stack([pt, vprim.astype(np.float64)], 1)[sel_idx].astype(np.float32),
        particle_ids=p_pid[sel_idx].astype(np.int64),
        event_ids=ev_ids[p_ev[sel_idx]].astype(np.int64),
        acts_reco=acts_reco[sel_idx],
        acts_dm=acts_dm[sel_idx],
        truth_kf_reco=truth_kf[sel_idx],
        n_events=n_events,
        n_particles=n_part,
        n_raw_hits=n_hits,
    )


# ---------------------------------------------------------------------------
# one output part = one group of input shards
# ---------------------------------------------------------------------------

def part_seed(seed: int, part_index: int, split: str) -> int:
    """Seed of the within-part shuffle."""
    return seed + 1000 * part_index + SPLITS.index(split)


def write_part(args):
    part_dir, shards, cfg, seed = args
    part_dir = Path(part_dir)
    if (part_dir / "_complete").exists():
        return json.load(open(part_dir / "meta.json"))
    t0 = time.time()
    chunks, stats = [], dict(n_events=0, n_particles=0, n_raw_hits=0)
    for pf, hf, tf, ttf, shf in shards:
        try:
            c = select_shard(Path(pf), Path(hf), Path(tf), Path(ttf) if ttf else None,
                             Path(shf) if shf else None, cfg)
        except Exception:
            return {"error": f"{pf}: {traceback.format_exc()}"}
        if c is None:
            continue
        for k in stats:
            stats[k] += c.pop(k)
        chunks.append(c)
    if not chunks:
        return {"error": f"{part_dir}: no tracks selected"}

    def cat(key):
        return np.concatenate([c[key] for c in chunks])

    # Shuffle the tracks within the part.
    lens_src = cat("lengths")
    n_tracks, n_hits = len(lens_src), int(lens_src.sum())
    order = np.random.default_rng(seed).permutation(n_tracks)
    start_src = np.concatenate([[0], np.cumsum(lens_src, dtype=np.int64)[:-1]])
    lens = lens_src[order].astype(np.int32)
    off = np.zeros(n_tracks + 1, np.int64)
    np.cumsum(lens, out=off[1:])
    hit_src = np.repeat(start_src[order] - off[:-1], lens) + np.arange(n_hits)

    part_dir.mkdir(parents=True, exist_ok=True)
    np.save(part_dir / "hits.npy", cat("hits")[hit_src])
    np.save(part_dir / "offsets.npy", off)
    np.save(part_dir / "lengths.npy", lens)
    np.save(part_dir / "targets.npy", cat("targets")[order])
    np.save(part_dir / "track_meta.npy", cat("track_meta")[order])
    np.save(part_dir / "track_particle_ids.npy", cat("particle_ids")[order])
    np.save(part_dir / "track_event_ids.npy", cat("event_ids")[order])
    np.save(part_dir / "acts_reco.npy", cat("acts_reco")[order])
    np.save(part_dir / "acts_dm.npy", cat("acts_dm")[order])
    np.save(part_dir / "truth_kf_reco.npy", cat("truth_kf_reco")[order])
    meta = dict(name=part_dir.name, n_tracks=int(n_tracks), n_hits=n_hits,
                seconds=round(time.time() - t0, 1), **stats)
    json.dump(meta, open(part_dir / "meta.json", "w"))
    (part_dir / "_complete").touch()
    return meta


# ---------------------------------------------------------------------------
# dataset discovery + driver
# ---------------------------------------------------------------------------

def discover(root: Path):
    """``[(particles, tracker_hits, tracks, truth_tracks|None, tracker_simhits|None), ...]``."""
    out = []
    flat = root / "parquet" / "truth" / "particles"
    if flat.is_dir():
        def find(table, tag):
            d = root / "parquet" / table
            f = next(d.glob(f"*{tag}"), None) if d.is_dir() else None
            return str(f) if f else None
        for pf in sorted(flat.glob("*.parquet")):
            tag = re.search(r"events\d+-\d+\.parquet", pf.name).group(0)
            hf, tf = find("reco/tracker_hits", tag), find("reco/tracks", tag)
            if hf and tf:
                out.append((str(pf), hf, tf, find("reco/truth_tracks", tag), find("truth/tracker_simhits", tag)))
        return out
    runs = sorted((root / "runs").glob("*/"), key=lambda p: int(p.name) if p.name.isdigit() else 1 << 30)
    for rd in runs:
        for pf in sorted((rd / "particles").glob("*.parquet")):
            tag = pf.name.replace("particles_", "")
            f = {t: rd / t / f"{t}_{tag}" for t in ("tracker_hits", "tracks", "truth_tracks", "tracker_simhits")}
            if f["tracker_hits"].exists() and f["tracks"].exists():
                out.append((str(pf), str(f["tracker_hits"]), str(f["tracks"]),
                            *(str(f[t]) if f[t].exists() else None for t in ("truth_tracks", "tracker_simhits"))))
    return out


def group(items, n_groups):
    n_groups = max(1, min(n_groups, len(items)))
    bounds = np.linspace(0, len(items), n_groups + 1).astype(int)
    return [items[a:b] for a, b in zip(bounds[:-1], bounds[1:]) if b > a]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True, help="dataset root (holds parquet/ or runs/)")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--sort-key", choices=("true_time", "geometry"), default="true_time",
                    help="per-track hit order stored on disk (see module docstring)")
    ap.add_argument("--bz", type=float, default=3.0, help="solenoid field [T] for the vertex -> perigee transport")
    ap.add_argument("--pt-min", type=float, default=0.5, help="lower pT cut [GeV]")
    ap.add_argument("--pt-max", type=float, default=None, help="upper pT cut [GeV]")
    ap.add_argument("--d0-max", type=float, default=7.1, help="|d0| cut [mm] (the target normalisation range)")
    ap.add_argument("--z0-max", type=float, default=270.0, help="|z0| cut [mm] (the target normalisation range)")
    ap.add_argument("--shards-per-part", type=int, default=1, help="input parquet shards merged into one output part")
    ap.add_argument("--num-workers", type=int, default=16)
    ap.add_argument("--train-frac", type=float, default=0.90)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    cfg = dict(sort_key=a.sort_key, bz=a.bz, pt_min=a.pt_min, pt_max=a.pt_max, d0_max=a.d0_max,
               z0_max=a.z0_max, eta_max=ETA_MAX, min_hits=MIN_HITS, max_hits=MAX_HITS)
    root, out = Path(a.data_dir), Path(a.output_dir)
    shards = discover(root)
    if not shards:
        sys.exit(f"no parquet shards found under {root}")

    # Split at the INPUT SHARD level so no event straddles two splits.  Val and
    # test get at least one shard each whenever there are three or more.
    n = len(shards)
    perm = np.random.default_rng(a.seed).permutation(n)
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
    print(f"{root}: {n} input shards -> " + ", ".join(f"{k} {len(v)}" for k, v in split_idx.items()), flush=True)

    out.mkdir(parents=True, exist_ok=True)
    jobs = []
    for sp, idxs in split_idx.items():
        sel = [shards[i] for i in idxs]
        if not sel:
            continue
        n_parts = max(1, int(round(len(sel) / a.shards_per_part)))
        for pi, grp in enumerate(group(sel, n_parts)):
            jobs.append((sp, (str(out / sp / f"part_{pi:04d}"), grp, cfg, part_seed(a.seed, pi, sp))))

    results = {k: [] for k in split_idx}
    errors = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.num_workers) as ex:
        futs = {ex.submit(write_part, j): sp for sp, j in jobs}
        for done, f in enumerate(as_completed(futs), 1):
            r = f.result()
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
                   shuffled_within_part=True, hit_sort_key=a.sort_key, bz=a.bz,
                   parts=[{k: p[k] for k in ("name", "n_tracks", "n_hits")} for p in parts],
                   n_tracks=sum(p["n_tracks"] for p in parts),
                   n_hits=sum(p["n_hits"] for p in parts))
        json.dump(man, open(out / sp / "manifest.json", "w"), indent=1)

    meta = dict(
        selection=cfg, seed=a.seed,
        splits={sp: dict(n_input_shards=len(split_idx[sp]),
                         **{k: sum(p[k] for p in results[sp])
                            for k in ("n_tracks", "n_hits", "n_events", "n_particles")})
                for sp in results if results[sp]},
        errors=errors, seconds=round(time.time() - t0, 1))
    json.dump(meta, open(out / "dataset_meta.json", "w"), indent=1)
    tot = sum(v["n_tracks"] for v in meta["splits"].values())
    print(f"DONE {out}: {tot:,} tracks in {time.time()-t0:.0f}s"
          + (f"  [{len(errors)} ERRORS]" if errors else ""), flush=True)
    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
