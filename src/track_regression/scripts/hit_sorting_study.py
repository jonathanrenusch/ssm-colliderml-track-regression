#!/usr/bin/env python3
"""Side-by-side hit-ordering study on the drift_beamspot flat test stores.

Reference order = what ACTS does.  ``TruthTrackFinder`` and
``TruthSeedingAlgorithm`` sort a truth track's measurements by
``SimHit::time()``; in the ColliderML parquet that key is
``tracker_simhits.true_time``, reached from a ``tracker_hits`` row through
``simhit_ids`` (positional index into the event's sim-hit list, aligned with
``particle_ids``).  Every candidate ordering from
:mod:`track_regression.hit_sorting` is compared against it on the tracks the
model actually sees (the flat ``test`` stores).  Nothing is modified.

Outputs (under ``dataset_plots/event_displays_acts_sorted/``)::

    summary.json, summary.md              agreement table + binned curves
    summary_disagreement.pdf              disagreement vs pT, |eta|, |d0|, per method
    lowpt_disagreements{,_zoom}.pdf       pT < 3 GeV tracks where s-order != ACTS order
    <dataset>/overlay_10events.pdf        muon sets, same events as the existing displays,
    <dataset>/event_NN_id<eid>.pdf        ttbar, one figure per event      (ACTS order)
    <dataset>/side_by_side_*.pdf          rows: legacy s / ACTS truth time / geometry order
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from track_regression import hit_sorting as hs                                   # noqa: E402
from track_regression.scripts.plot_preprocessed import (                          # noqa: E402
    RXY, RZ, _draw_event, _panels, save, style)
from track_regression.scripts.preprocess_flat import _flat, _rowid               # noqa: E402

import matplotlib.pyplot as plt                                                  # noqa: E402
from matplotlib.lines import Line2D                                              # noqa: E402

REPO = Path(__file__).resolve().parents[3]
ROOT_RAW = Path("/scratch/colliderml/drift_beamspot")
ROOT_FLAT = Path("/scratch/colliderml/ICLR_retraining")
DATASETS = ["single_muon_2GeV", "single_muon_10GeV", "single_muon_100GeV",
            "single_muon_uniform", "ttbar"]

# name -> (label, needs truth).  Fixed order; the summary figure keeps it.
METHODS = {
    "stored":    ("as stored on disk (digitised time)", False),
    "s_origin":  ("s = |X| from the origin (legacy)", False),
    "r":         ("cylindrical r", False),
    "geometry":  ("geometry order (group / barrel-endcap / r or z)", False),
    "s_perigee": ("|X - P|, P = truth perigee point", True),
    "helix_T":   ("helix arc length, transverse only", True),
    "helix":     ("helix arc length, barrel r-phi + endcap z", True),
}
# dataviz reference palette, categorical slots 1-6 in fixed order (validated adjacent-pair CVD)
PALETTE = {"s_origin": "#2a78d6", "r": "#eb6834", "geometry": "#1baf7a",
           "s_perigee": "#eda100", "helix_T": "#e87ba4", "helix": "#008300"}

PT_EDGES = np.array([0.5, 1, 2, 3, 5, 10, 20, 50, 100, 1000.0])
ETA_EDGES = np.arange(0, 3.01, 0.25)
D0_EDGES = np.arange(0, 7.01, 1.0)
Z0_EDGES = np.arange(0, 280.1, 40.0)


# ---------------------------------------------------------------------------
# raw parquet: truth time per (event, particle, hit position)
# ---------------------------------------------------------------------------

def raw_shards(ds: str):
    """[(event_lo, event_hi, tracker_hits path, tracker_simhits path or None)] for both layouts."""
    root = ROOT_RAW / ds / "v1"
    out = []
    for hf in sorted(root.glob("runs/*/tracker_hits/tracker_hits_*.parquet")):
        lo, hi = (int(v) for v in re.search(r"_(\d+)-(\d+)\.parquet$", hf.name).groups())
        sf = hf.parent.parent / "tracker_simhits" / hf.name.replace("tracker_hits_", "tracker_simhits_")
        out.append((lo, hi - 1, hf, sf if sf.exists() else None))
    for hf in sorted(root.glob("parquet/reco/tracker_hits/*.parquet")):
        lo, hi = (int(v) for v in re.search(r"events(\d+)-(\d+)\.parquet$", hf.name).groups())
        sf = root / "parquet/truth/tracker_simhits" / hf.name.replace(
            ".reco.tracker_hits.", ".truth.tracker_simhits.")
        out.append((lo, hi, hf, sf if sf.exists() else None))
    return out


def _bits(a):
    return np.ascontiguousarray(a, dtype=np.float32).view(np.uint32).astype(np.int64)


def truth_time_table(hits_path: Path, simhits_path: Path, events: np.ndarray):
    """``{(event, pid, xbits, ybits, zbits): true_time}`` for the requested events.

    A measurement can merge several sim hits of the same particle; the earliest
    one is what ACTS's time sort would see first, so keep the minimum.
    """
    ev_set = pa.array(np.unique(events).astype(np.uint32))
    ht = pq.read_table(hits_path, columns=["event_id", "x", "y", "z", "particle_ids", "simhit_ids"])
    ht = ht.filter(pc.is_in(ht.column("event_id"), value_set=ev_set))
    st = pq.read_table(simhits_path, columns=["event_id", "particle_id", "true_time"])
    st = st.filter(pc.is_in(st.column("event_id"), value_set=ev_set))
    stats = dict(pairs=0, matched=0, pid_mismatch=0)
    if ht.num_rows == 0 or st.num_rows == 0:
        return {}, stats
    h_ev = ht.column("event_id").to_numpy().astype(np.int64)
    x, h_off = _flat(ht.column("x"))
    y, z = _flat(ht.column("y"))[0], _flat(ht.column("z"))[0]
    xb, yb, zb = _bits(x), _bits(y), _bits(z)
    hit_ev = h_ev[_rowid(h_off, len(x))]
    pair_pid, pair_off = _flat(pc.list_flatten(ht.column("particle_ids")))
    pair_sid, pair_off2 = _flat(pc.list_flatten(ht.column("simhit_ids")))
    if not np.array_equal(pair_off, pair_off2):
        raise RuntimeError(f"{hits_path.name}: particle_ids and simhit_ids are not aligned")
    pair_hit = _rowid(pair_off, len(pair_pid))
    pair_pid = pair_pid.astype(np.int64)
    pair_sid = pair_sid.astype(np.int64)

    s_ev = st.column("event_id").to_numpy().astype(np.int64)
    s_pid, s_off = _flat(st.column("particle_id"))
    s_tt = _flat(st.column("true_time"))[0].astype(np.float64)
    o = np.argsort(s_ev, kind="stable")
    pos = np.searchsorted(s_ev[o], hit_ev[pair_hit])
    ok = pos < len(o)
    pos = np.where(ok, pos, 0)
    ok &= s_ev[o][pos] == hit_ev[pair_hit]
    s_row = o[pos]
    g = s_off[s_row] + pair_sid
    ok &= g < s_off[s_row + 1]
    g = np.where(ok, g, 0)
    pid_ok = s_pid[g].astype(np.int64) == pair_pid
    stats["pairs"] = int(len(pair_pid))
    stats["pid_mismatch"] = int((ok & ~pid_ok).sum())
    ok &= pid_ok
    stats["matched"] = int(ok.sum())

    hh = pair_hit[ok]
    keys = zip(hit_ev[hh].tolist(), pair_pid[ok].tolist(), xb[hh].tolist(), yb[hh].tolist(), zb[hh].tolist())
    table: dict = {}
    for k, t in zip(keys, s_tt[g[ok]].tolist()):
        prev = table.get(k)
        if prev is None or t < prev:
            table[k] = t
    return table, stats


# ---------------------------------------------------------------------------
# flat store
# ---------------------------------------------------------------------------

def pick_display_events(split_root: Path, n_events: int, seed: int):
    """Same RNG sequence as ``plot_preprocessed.Store.events`` -> same events as the existing figures."""
    man = json.loads((split_root / "manifest.json").read_text())
    parts = [p["name"] for p in man["parts"]]
    rng = np.random.default_rng(seed)
    nm = parts[rng.integers(0, len(parts))]
    ev = np.asarray(np.load(split_root / nm / "track_event_ids.npy"))
    uniq, _ = np.unique(ev, return_counts=True)
    pick = rng.choice(len(uniq), size=min(n_events, len(uniq)), replace=False)
    return nm, [(int(u), np.nonzero(ev == u)[0]) for u in uniq[pick]]


def load_tracks(part: Path, idx: np.ndarray):
    """Hits + per-track truth for the given track indices of one part, CSR-concatenated."""
    H = np.load(part / "hits.npy", mmap_mode="r")
    off = np.load(part / "offsets.npy", mmap_mode="r")
    lens = np.asarray(np.load(part / "lengths.npy", mmap_mode="r")[idx]).astype(np.int64)
    csum = np.cumsum(lens)
    pos = np.arange(int(csum[-1]), dtype=np.int64) - np.repeat(csum - lens, lens)
    g = np.repeat(np.asarray(off[idx]), lens) + pos
    tgt = np.asarray(np.load(part / "targets.npy", mmap_mode="r")[idx]).astype(np.float64)
    meta = np.asarray(np.load(part / "track_meta.npy", mmap_mode="r")[idx])
    return dict(
        H=np.asarray(H[g]), lens=lens, pos=pos,
        tid=np.repeat(np.arange(len(idx)), lens), start=csum - lens,
        targets=tgt, pt=meta[:, 0].astype(np.float64),
        ev=np.asarray(np.load(part / "track_event_ids.npy", mmap_mode="r")[idx]).astype(np.int64),
        pid=np.asarray(np.load(part / "track_particle_ids.npy", mmap_mode="r")[idx]).astype(np.int64),
        idx=idx,
    )


def attach_truth_time(tr: dict, ds: str):
    """Fill ``tr['tt']`` (nan where no sim hit was found) from every raw shard that can hold the events."""
    H, tid = tr["H"], tr["tid"]
    keys = list(zip(tr["ev"][tid].tolist(), tr["pid"][tid].tolist(),
                    _bits(H[:, 0]).tolist(), _bits(H[:, 1]).tolist(), _bits(H[:, 2]).tolist()))
    tt = np.full(len(keys), np.nan)
    events = np.unique(tr["ev"])
    agg = dict(pairs=0, matched=0, pid_mismatch=0, shards=[])
    for lo, hi, hf, sf in raw_shards(ds):
        need = events[(events >= lo) & (events <= hi)]
        if len(need) == 0 or np.isfinite(tt).all():
            continue
        if sf is None:
            print(f"    [warn] no tracker_simhits next to {hf}", flush=True)
            continue
        table, st = truth_time_table(hf, sf, need)
        n_new = 0
        for i, k in enumerate(keys):
            if np.isnan(tt[i]):
                t = table.get(k)
                if t is not None:
                    tt[i] = t
                    n_new += 1
        for k in ("pairs", "matched", "pid_mismatch"):
            agg[k] += st[k]
        agg["shards"].append(dict(hits=str(hf.relative_to(ROOT_RAW)), filled=n_new))
        print(f"    {hf.relative_to(ROOT_RAW)}: {n_new} hits matched", flush=True)
    tr["tt"] = tt
    return agg


# ---------------------------------------------------------------------------
# orderings and metrics
# ---------------------------------------------------------------------------

def ranks_from_perm(perm, tid, start):
    """perm sorts hits by (track, key); return each hit's rank inside its track."""
    r = np.empty(len(perm), np.int64)
    r[perm] = np.arange(len(perm)) - start[tid[perm]]
    return r


def all_ranks(tr: dict):
    H, tid, start = tr["H"], tr["tid"], tr["start"]
    xyz = H[:, :3].astype(np.float64)
    vol = H[:, 7]
    T = tr["targets"][tid]
    d0, z0, phi, theta, qop = (T[:, i] for i in range(5))
    keys = {
        "stored": tr["pos"].astype(np.float64),
        "s_origin": hs.distance_from_origin(xyz),
        "r": hs.cylindrical_radius(xyz),
        "s_perigee": hs.distance_from_perigee(xyz, d0, z0, phi),
        "helix_T": hs.helix_arc_length(xyz, d0, z0, phi, theta, qop, mode="transverse"),
        "helix": hs.helix_arc_length(xyz, d0, z0, phi, theta, qop, vol, mode="mixed"),
    }
    ranks = {m: ranks_from_perm(np.lexsort((k, tid)), tid, start) for m, k in keys.items()}
    # per-track direction of flight from the innermost / outermost hit (by r)
    r = np.hypot(xyz[:, 0], xyz[:, 1])
    o = np.lexsort((r, tid))
    inner, outer = o[start], o[start + tr["lens"] - 1]
    direction = np.where(xyz[outer, 2] >= xyz[inner, 2], 1.0, -1.0)
    prim, sec = hs.geometry_keys(xyz, vol, direction[tid])
    ranks["geometry"] = ranks_from_perm(np.lexsort((sec, prim, tid)), tid, start)
    ref = ranks_from_perm(np.lexsort((tr["tt"], tid)), tid, start)
    return ranks, ref


def compare(tr: dict, ranks: dict, ref: np.ndarray):
    """Per track and method: exact agreement, Kendall distance, same-layer share of discordant pairs."""
    n_trk, lens, tid = len(tr["lens"]), tr["lens"], tr["tid"]
    Lmax = int(lens.max())
    lay = (tr["H"][:, 7] * 100 + tr["H"][:, 8]).astype(np.int64)
    LAY = np.full((n_trk, Lmax), -1, np.int64)
    LAY[tid, ref] = lay
    iu = np.triu_indices(Lmax, 1)
    same_layer = LAY[:, iu[0]] == LAY[:, iu[1]]
    npairs = lens * (lens - 1) / 2.0
    out = {}
    for m, rk in ranks.items():
        RK = np.full((n_trk, Lmax), np.inf)
        RK[tid, ref] = rk                    # method rank, laid out in reference order
        disc = RK[:, iu[0]] > RK[:, iu[1]]   # pads are +inf -> never discordant
        n_disc = disc.sum(1)
        out[m] = dict(
            exact=(n_disc == 0),
            kendall=n_disc / npairs,
            same_layer_disc=(disc & same_layer).sum(1),
            n_disc=n_disc,
        )
    return out


def binned(x, edges, flags):
    """(fraction, n) of ``flags`` per bin of ``x``."""
    b = np.digitize(x, edges) - 1
    ok = (b >= 0) & (b < len(edges) - 1)
    n = np.bincount(b[ok], minlength=len(edges) - 1).astype(float)
    k = np.bincount(b[ok], weights=flags[ok].astype(float), minlength=len(edges) - 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, k / n, np.nan), n


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

def _rows(fig, n):
    return fig.subfigures(n, 1, hspace=0.08)


def _panels_zoom(fig, xyz):
    ax = fig.subplots(1, 3, gridspec_kw={"wspace": 0.3})
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    for a, (u, v, xl, yl) in zip(ax, [(x, y, "x [mm]", "y [mm]"), (z, x, "z [mm]", "x [mm]"),
                                      (z, y, "z [mm]", "y [mm]")]):
        m = 0.06 * max(np.ptp(u), np.ptp(v), 1.0)
        a.set_xlim(u.min() - m, u.max() + m); a.set_ylim(v.min() - m, v.max() + m)
        a.set_xlabel(xl); a.set_ylabel(yl); a.set_aspect("equal", adjustable="box"); a.grid(alpha=0.25)
    ax[0].set_title("x–y (beam view)"); ax[1].set_title("z–x"); ax[2].set_title("z–y")
    return ax


ROW_LABELS = {"s_origin": "hit order: legacy s = sqrt(x²+y²+z²) from the origin",
              "acts": "hit order: ACTS reference — tracker_simhits.true_time (SimHit::time)",
              "geometry": "hit order: geometry order, truth-free (pixel→sstrip→lstrip; barrel r, endcap z along flight)",
              "helix": "hit order: helix arc length from the truth perigee"}


def ordered(tr, i, rank):
    """Hits of track ``i`` sorted by ``rank`` (a per-hit rank array)."""
    sl = slice(int(tr["start"][i]), int(tr["start"][i] + tr["lens"][i]))
    h = tr["H"][sl]
    return h[np.argsort(rank[sl], kind="stable")]


def fig_multi_row(track_ids, tr, ranks, ref, rows, colours, title, out_path, zoom=False):
    fig = plt.figure(figsize=(21, 4.8 * len(rows)))
    subs = _rows(fig, len(rows))
    subs = np.atleast_1d(subs)
    allxyz = tr["H"][np.isin(tr["tid"], track_ids)][:, :3]
    for sf, row in zip(subs, rows):
        ax = _panels_zoom(sf, allxyz) if zoom else _panels(sf)
        rk = ref if row == "acts" else ranks[row]
        for j, i in enumerate(track_ids):
            _draw_event(ax, [ordered(tr, i, rk)], colours[j], lw=0.9, ms=2.4)
        sf.suptitle(ROW_LABELS[row], y=0.995, fontsize=11)
    fig.suptitle(title, y=1.012)
    save(fig, out_path)


def dataset_displays(ds, tr, ranks, ref, disp, out: Path):
    """ACTS-ordered displays in the style of the existing ones + 3-row side-by-side figures."""
    cmap = plt.get_cmap("tab10")
    tag = "hit order: ACTS reference — tracker_simhits.true_time via simhit_ids"
    if "ttbar" not in ds:
        fig = plt.figure(figsize=(21, 5.2))
        ax = _panels(fig)
        for k, (eid, tids) in enumerate(disp):
            for i in tids:
                _draw_event(ax, [ordered(tr, i, ref)], cmap(k % 10))
        fig.legend([Line2D([0], [0], color=cmap(k % 10), lw=2) for k in range(len(disp))],
                   [f"event {eid}" for eid, _ in disp], loc="upper center", ncol=min(len(disp), 10),
                   frameon=False, bbox_to_anchor=(0.5, 1.00), fontsize=8)
        ntr = sum(len(t) for _, t in disp)
        fig.suptitle(f"{ds} — {len(disp)} events overlaid, {ntr} tracks — {tag}\n"
                     f"axes fixed to the ODD envelope: |r| < {RXY:.0f} mm, |z| < {RZ:.0f} mm", y=1.13)
        fig.tight_layout()
        save(fig, out / f"overlay_{len(disp)}events")
        tids = np.concatenate([t for _, t in disp])
        cols = [cmap(k % 10) for k, (_, t) in enumerate(disp) for _ in t]
        fig_multi_row(tids, tr, ranks, ref, ["s_origin", "acts", "geometry"], cols,
                      f"{ds} — the same {len(disp)} events under three hit orderings "
                      f"(a wrong order shows as a zig-zag)", out / f"side_by_side_overlay_{len(disp)}events")
    else:
        for k, (eid, tids) in enumerate(disp):
            fig = plt.figure(figsize=(21, 5.2))
            ax = _panels(fig)
            for j, i in enumerate(tids):
                _draw_event(ax, [ordered(tr, i, ref)], cmap(j % 10), lw=0.7, ms=1.6)
            fig.suptitle(f"{ds} — event {eid}, {len(tids)} selected tracks — {tag}", y=1.02)
            fig.tight_layout()
            save(fig, out / f"event_{k:02d}_id{eid}")
            cols = [cmap(j % 10) for j in range(len(tids))]
            fig_multi_row(tids, tr, ranks, ref, ["s_origin", "acts", "geometry"], cols,
                          f"{ds} — event {eid}, {len(tids)} tracks under three hit orderings",
                          out / f"side_by_side_event_{k:02d}_id{eid}")


def summary_figure(results: dict, out: Path):
    style()
    methods = [m for m in METHODS if m != "stored"]
    rows = [("pt", "$p_T$ [GeV]", PT_EDGES, True), ("eta", r"$|\eta|$", ETA_EDGES, False),
            ("d0", "$|d_0|$ [mm]", D0_EDGES, False), ("z0", "$|z_0|$ [mm]", Z0_EDGES, False)]
    fig, axes = plt.subplots(len(rows), len(DATASETS), figsize=(4.2 * len(DATASETS), 3.6 * len(rows)),
                             squeeze=False)
    for c, ds in enumerate(DATASETS):
        R = results[ds]
        for r, (key, xl, edges, logx) in enumerate(rows):
            ax = axes[r, c]
            centres = np.sqrt(edges[:-1] * edges[1:]) if logx else 0.5 * (edges[:-1] + edges[1:])
            for m in methods:
                frac = np.asarray(R["binned"][key][m]["frac"], float)
                n = np.asarray(R["binned"][key][m]["n"], float)
                ok = n > 0
                y = 100.0 * frac
                floor = 100.0 * 0.5 / np.maximum(n, 1)
                ls = "--" if METHODS[m][1] else "-"
                nz = ok & (frac > 0)
                ax.plot(centres[nz], y[nz], ls, color=PALETTE[m], lw=1.6, marker="o", ms=4)
                zero = ok & (frac == 0)
                ax.plot(centres[zero], floor[zero], ls="none", marker="v", ms=5, mfc="white",
                        mec=PALETTE[m], mew=1.2)
            ax.set_yscale("log"); ax.set_ylim(3e-3, 150)
            if logx:
                ax.set_xscale("log")
            ax.set_xlabel(xl)
            if c == 0:
                ax.set_ylabel("tracks ordered differently\nfrom ACTS [%]")
            if r == 0:
                ax.set_title(f"{ds}\n{R['n_tracks_matched']:,} tracks", fontsize=11)
    handles = [Line2D([0], [0], color=PALETTE[m], lw=1.6, ls="--" if METHODS[m][1] else "-", marker="o", ms=4)
               for m in methods]
    labels = [METHODS[m][0] + ("  [needs truth]" if METHODS[m][1] else "") for m in methods]
    handles.append(Line2D([0], [0], ls="none", marker="v", mfc="white", mec="0.3", ms=6))
    labels.append("no disagreement in bin (marker at 0.5/N)")
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.06), fontsize=9)
    fig.suptitle("drift_beamspot test stores — hit-ordering methods vs the ACTS reference order "
                 "(sort by tracker_simhits.true_time)", y=1.10)
    fig.tight_layout()
    save(fig, out / "summary_disagreement")


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def run_dataset(ds: str, n_sample: int, n_events: int, seed: int, out: Path, rng):
    split_root = ROOT_FLAT / ds / "test"
    part_name, disp = pick_display_events(split_root, n_events, seed=1)
    part = split_root / part_name
    n_all = len(np.load(part / "lengths.npy", mmap_mode="r"))
    disp_tracks = np.concatenate([t for _, t in disp])
    sample = rng.choice(n_all, size=min(n_sample, n_all), replace=False)
    idx = np.unique(np.concatenate([sample, disp_tracks]))
    print(f"=== {ds}: part {part_name}, {n_all:,} tracks, using {len(idx):,}", flush=True)
    tr = load_tracks(part, idx)
    # display track indices -> positions in the sample
    remap = {int(v): i for i, v in enumerate(idx)}
    disp = [(eid, np.array([remap[int(i)] for i in t])) for eid, t in disp]

    t0 = time.time()
    join = attach_truth_time(tr, ds)
    n_miss = np.bincount(tr["tid"], weights=np.isnan(tr["tt"]), minlength=len(idx))
    matched_trk = n_miss == 0
    # A track with NO hit in its own raw event is a mislabelled sample: the
    # preprocessor joined hits to particles by parquet row index, and the
    # particles / tracker_hits tables do not list events in the same row order.
    n_mislab = int((n_miss == tr["lens"]).sum())
    n_partial = int(((n_miss > 0) & (n_miss < tr["lens"])).sum())
    print(f"    truth time attached to {matched_trk.sum():,}/{len(idx):,} tracks "
          f"({time.time()-t0:.0f} s; pairs {join['pairs']:,}, matched {join['matched']:,}, "
          f"pid mismatch {join['pid_mismatch']})", flush=True)
    print(f"    tracks whose hits belong to another event (row-order mis-join): {n_mislab:,}; "
          f"partially unmatched: {n_partial:,}", flush=True)
    # unmatched tracks: give them a harmless key so ranks are defined, but exclude from stats
    tr["tt"] = np.where(np.isnan(tr["tt"]), tr["pos"].astype(np.float64), tr["tt"])

    ranks, ref = all_ranks(tr)
    cmp = compare(tr, ranks, ref)
    T = tr["targets"]
    with np.errstate(all="ignore"):
        eta = -np.log(np.tan(np.clip(T[:, 3], 1e-8, np.pi - 1e-8) / 2.0))
    pt, ad0 = tr["pt"], np.abs(T[:, 0])
    m = matched_trk
    res = dict(part=part_name, n_tracks_used=int(len(idx)), n_tracks_matched=int(m.sum()),
               join=join, methods={}, binned={"pt": {}, "eta": {}, "d0": {}, "z0": {}},
               n_tracks_mislabelled=int(n_mislab), n_tracks_partial=int(n_partial))
    for meth, c in cmp.items():
        ex = c["exact"][m]
        dis = ~ex
        res["methods"][meth] = dict(
            agree_frac=float(ex.mean()),
            n_disagree=int(dis.sum()),
            mean_kendall_when_disagree=float(c["kendall"][m][dis].mean()) if dis.any() else 0.0,
            same_layer_share_of_discordant=float(c["same_layer_disc"][m].sum() / max(c["n_disc"][m].sum(), 1)),
            disagree_lowpt_lt3=float(dis[pt[m] < 3].mean()) if (pt[m] < 3).any() else None,
        )
        for key, x, edges in (("pt", pt, PT_EDGES), ("eta", np.abs(eta), ETA_EDGES), ("d0", ad0, D0_EDGES),
                              ("z0", np.abs(T[:, 1]), Z0_EDGES)):
            frac, n = binned(x[m], edges, dis)
            res["binned"][key][meth] = dict(frac=[None if np.isnan(v) else float(v) for v in frac],
                                            n=n.astype(int).tolist(), edges=edges.tolist())
    for meth, r in res["methods"].items():
        print(f"    {meth:10s} agree {100*r['agree_frac']:7.3f}%   n_dis {r['n_disagree']:6d}   "
              f"same-layer share {100*r['same_layer_share_of_discordant']:5.1f}%", flush=True)

    dataset_displays(ds, tr, ranks, ref, disp, out / ds)
    # low-pT candidates for the cross-dataset figure
    low = np.nonzero(m & (pt < 3) & ~cmp["s_origin"]["exact"])[0]
    np.savez(out / ds / "per_track.npz", pt=pt, eta=eta, d0=T[:, 0], lens=tr["lens"], matched=m,
             **{f"exact_{k}": v["exact"] for k, v in cmp.items()},
             **{f"kendall_{k}": v["kendall"] for k, v in cmp.items()})
    return res, (tr, ranks, ref, low)


def lowpt_figure(cands: list, out: Path, n_show: int = 8):
    """Up to n_show pT<3 GeV tracks (one colour each) where the s-order differs from ACTS."""
    picks = []
    for ds, tr, ranks, ref, low in cands:
        for i in low[:3]:
            picks.append((ds, tr, ranks, ref, int(i)))
    picks = picks[:n_show]
    if not picks:
        print("no low-pT disagreements to draw", flush=True)
        return
    cmap = plt.get_cmap("tab10")
    for zoom in (False, True):
        fig = plt.figure(figsize=(21, 4.8 * 3))
        subs = np.atleast_1d(_rows(fig, 3))
        allxyz = np.concatenate([ordered(tr, i, ref)[:, :3] for _, tr, _, ref, i in picks])
        for sf, row in zip(subs, ["s_origin", "acts", "geometry"]):
            ax = _panels_zoom(sf, allxyz) if zoom else _panels(sf)
            for j, (ds, tr, ranks, ref, i) in enumerate(picks):
                rk = ref if row == "acts" else ranks[row]
                _draw_event(ax, [ordered(tr, i, rk)], cmap(j % 10), lw=0.9, ms=2.6)
            sf.suptitle(ROW_LABELS[row], y=0.995, fontsize=11)
        fig.legend([Line2D([0], [0], color=cmap(j % 10), lw=2) for j in range(len(picks))],
                   [f"{ds.replace('single_muon_', 'mu ')}  pT={tr['pt'][i]:.2f} GeV  "
                    f"eta={-np.log(np.tan(tr['targets'][i, 3] / 2)):+.2f}  d0={tr['targets'][i, 0]:+.1f} mm"
                    for ds, tr, _, _, i in picks], loc="upper center", ncol=4, frameon=False,
                   bbox_to_anchor=(0.5, 1.06), fontsize=8)
        fig.suptitle("pT < 3 GeV tracks whose legacy s-order differs from the ACTS order"
                     + (" — axes zoomed to the drawn hits" if zoom else " — fixed ODD envelope"), y=1.08)
        save(fig, out / ("lowpt_disagreements_zoom" if zoom else "lowpt_disagreements"))
    for k, (ds, tr, ranks, ref, i) in enumerate(picks[:6]):
        eta = -np.log(np.tan(tr["targets"][i, 3] / 2))
        fig_multi_row([i], tr, ranks, ref, ["s_origin", "acts", "geometry"], [cmap(k % 10)],
                      f"{ds} — one pT < 3 GeV track, pT = {tr['pt'][i]:.2f} GeV, eta = {eta:+.2f}, "
                      f"d0 = {tr['targets'][i, 0]:+.2f} mm, z0 = {tr['targets'][i, 1]:+.1f} mm — axes zoomed to its hits",
                      out / f"lowpt_track_{k:02d}_{ds}", zoom=True)


def write_markdown(results: dict, out: Path):
    lines = ["# Hit ordering vs the ACTS reference (sort by `tracker_simhits.true_time`)", "",
             "Exact agreement of the whole per-track permutation, flat `test` stores, one part per dataset.", ""]
    hdr = "| method | " + " | ".join(DATASETS) + " |"
    lines += [hdr, "|" + "---|" * (len(DATASETS) + 1)]
    for m, (lab, truth) in METHODS.items():
        row = [f"`{m}` {'(truth)' if truth else ''}"]
        for ds in DATASETS:
            r = results[ds]["methods"][m]
            row.append(f"{100*r['agree_frac']:.3f} % ({r['n_disagree']})")
        lines.append("| " + " | ".join(row) + " |")
    lines += ["", "Tracks used / with a truth time for every hit / whose hits belong to another event "
              "(row-order mis-join in preprocess_flat.select_shard; excluded above):",
              "", "| dataset | part | used | matched | mislabelled | partially unmatched |", "|---|---|---|---|---|---|"]
    for ds in DATASETS:
        R = results[ds]
        lines.append(f"| {ds} | {R['part']} | {R['n_tracks_used']:,} | {R['n_tracks_matched']:,} | "
                     f"{R['n_tracks_mislabelled']:,} ({100*R['n_tracks_mislabelled']/R['n_tracks_used']:.2f} %) | "
                     f"{R['n_tracks_partial']:,} |")
    lines += ["", "Share of discordant hit pairs that sit in the same (volume, layer) — i.e. swaps of the two "
              "staggered sensors of one layer rather than cross-layer errors:", "",
              hdr, "|" + "---|" * (len(DATASETS) + 1)]
    for m in METHODS:
        row = [f"`{m}`"] + [f"{100*results[ds]['methods'][m]['same_layer_share_of_discordant']:.1f} %"
                            for ds in DATASETS]
        lines.append("| " + " | ".join(row) + " |")
    (out / "summary.md").write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="*", default=DATASETS)
    ap.add_argument("--n-sample", type=int, default=20000, help="tracks per dataset (all if fewer)")
    ap.add_argument("--n-events", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(REPO / "dataset_plots" / "event_displays_acts_sorted"))
    a = ap.parse_args()
    style()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    results, cands = {}, []
    for ds in a.datasets:
        res, (tr, ranks, ref, low) = run_dataset(ds, a.n_sample, a.n_events, a.seed, out, rng)
        results[ds] = res
        cands.append((ds, tr, ranks, ref, low))
    json.dump(results, open(out / "summary.json", "w"), indent=1)
    if set(a.datasets) == set(DATASETS):
        summary_figure(results, out)
        write_markdown(results, out)
    # low-pT figure: prefer the datasets with a real pT spectrum, then the 2 GeV gun
    order_pref = ["single_muon_uniform", "ttbar", "single_muon_2GeV"]
    cands.sort(key=lambda c: order_pref.index(c[0]) if c[0] in order_pref else 9)
    lowpt_figure(cands, out)
    print("done", flush=True)


if __name__ == "__main__":
    main()
