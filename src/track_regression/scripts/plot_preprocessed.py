#!/usr/bin/env python3
"""Distribution and event-display plots straight from a preprocessed flat store.

Reads what the model is actually trained on -- no selection variants, no extra
cuts -- so what you see is exactly the preprocessed dataset (variant `core`,
d0/z0 windows removed). PDF only; y axes are absolute counts, log-scaled where
the quantity spans orders of magnitude.
"""
from __future__ import annotations

import argparse, json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

TARGETS = [("d0", "$d_0$ [mm]", False), ("z0", "$z_0$ [mm]", False),
           ("phi", r"$\phi$ [rad]", False), ("theta", r"$\theta$ [rad]", False),
           ("qop", "$q/p$ [1/GeV]", True)]
# (column, label, log_y)
HITF = [("x", "hit $x$ [mm]", True), ("y", "hit $y$ [mm]", True), ("z", "hit $z$ [mm]", True),
        ("r", "hit $r$ [mm]", True), ("phi_hit", r"hit $\phi$ [rad]", False),
        ("theta_hit", r"hit $\theta$ [rad]", False), ("s", "hit $s$ [mm]", True),
        ("volume_id", "volume_id", True), ("layer_id", "layer_id", True),
        ("surface_id", "surface_id", True), ("detector", "detector", True),
        ("eta_hit", r"hit $\eta$", False)]
# detector envelope (ODD): r < ~1030 mm, |z| < ~3030 mm
RXY, RZ = 1100.0, 3100.0


def style():
    plt.rcParams.update({
        "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
        "legend.fontsize": 9, "axes.grid": True, "grid.alpha": 0.25,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.dpi": 110, "savefig.bbox": "tight", "pdf.fonttype": 42,
    })


def save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def counts(ax, v, *, bins=120, rng=None, log="auto", discrete=False,
           colour="#1f4e79", ylabel="tracks"):
    """Absolute counts, never a density.

    ``log="auto"`` picks the y scale from the actual dynamic range: a steeply
    falling spectrum (ttbar pT, q/p, surface_id) needs log, a flat one (uniform
    d0, muon-gun pT) is destroyed by it -- a fixed per-quantity flag gets one of
    the two datasets wrong every time.
    """
    v = np.asarray(v, np.float64); v = v[np.isfinite(v)]
    if v.size == 0:
        return
    if discrete:
        u = np.unique(v)
        b = np.arange(u.min() - 0.5, u.max() + 1.5) if len(u) <= 80 else bins
    else:
        lo, hi = (np.percentile(v, [0.02, 99.98]) if rng is None else rng)
        # A fixed-pT gun gives one value whose float32 spread is ~1 ulp (1e-7 at
        # pT = 2). An absolute epsilon lets matplotlib auto-range into that noise
        # and the axis reads "1e-7+2", which looks like a physical spread and is
        # not one. Use a RELATIVE test and open a readable window instead.
        mid = 0.5 * (lo + hi)
        if not np.isfinite(lo) or (hi - lo) < 1e-5 * max(1.0, abs(mid)):
            half = max(0.05 * abs(mid), 1e-3)
            lo, hi = mid - half, mid + half
        b = np.linspace(lo, hi, bins + 1)
    h, edges = np.histogram(v, bins=b)
    if log == "auto":
        nz = h[h > 0]
        log = nz.size > 3 and (nz.max() / max(np.percentile(nz, 10), 1)) > 30.0
    ax.stairs(h, edges, fill=True, color=colour, alpha=0.85, linewidth=0.7,
              edgecolor=colour)
    if log:
        ax.set_yscale("log")
        ax.set_ylim(bottom=max(0.5, nz.min() * 0.7) if (h > 0).any() else 0.5)
    ax.set_ylabel(f"{ylabel} (log)" if log else ylabel)


class Store:
    """Sampling reader for a flat store split."""
    def __init__(self, root: Path, max_parts: int | None = None):
        self.root = root
        self.man = json.loads((root / "manifest.json").read_text())
        self.parts = [p["name"] for p in self.man["parts"]][:max_parts]

    def targets(self, max_tracks: int):
        T, PT, ETA, NH = [], [], [], []
        per = max(1, max_tracks // max(len(self.parts), 1))
        for nm in self.parts:
            d = self.root / nm
            t = np.load(d / "targets.npy", mmap_mode="r")[:per]
            m = np.load(d / "track_meta.npy", mmap_mode="r")[:per]
            L = np.load(d / "lengths.npy", mmap_mode="r")[:per]
            T.append(np.asarray(t)); PT.append(np.asarray(m[:, 0])); NH.append(np.asarray(L))
        T = np.concatenate(T); PT = np.concatenate(PT); NH = np.concatenate(NH)
        with np.errstate(all="ignore"):
            ETA = -np.log(np.tan(np.clip(T[:, 3], 1e-8, np.pi - 1e-8) / 2.0))
        return T, PT, ETA, NH

    def hits(self, max_hits: int):
        H = []
        per = max(1, max_hits // max(len(self.parts), 1))
        for nm in self.parts:
            H.append(np.asarray(np.load(self.root / nm / "hits.npy", mmap_mode="r")[:per]))
        return np.concatenate(H)

    def events(self, n_events: int, seed: int = 0, sort: str = "stored"):
        """Return [(event_id, [track_hit_arrays])] for n_events random events.

        ``sort="s"`` re-orders each track's hits by s = sqrt(x^2+y^2+z^2) instead
        of using the on-disk order. The on-disk order is by `tracker_hits.time`,
        which is undefined (exactly 0) for every strip hit and ~300x too large
        where it is written, so the stored sequence is not physical.
        """
        rng = np.random.default_rng(seed)
        nm = self.parts[rng.integers(0, len(self.parts))]
        d = self.root / nm
        ev = np.asarray(np.load(d / "track_event_ids.npy"))
        off = np.load(d / "offsets.npy", mmap_mode="r")
        hits = np.load(d / "hits.npy", mmap_mode="r")
        uniq, counts_ = np.unique(ev, return_counts=True)
        pick = rng.choice(len(uniq), size=min(n_events, len(uniq)), replace=False)
        out = []
        for u in uniq[pick]:
            idx = np.nonzero(ev == u)[0]
            trks = []
            for i in idx:
                t = np.asarray(hits[int(off[i]):int(off[i + 1])])
                if sort == "s":
                    t = t[np.argsort(t[:, 6], kind="stable")]
                trks.append(t)
            out.append((int(u), trks))
        return out


def plot_targets(st: Store, out: Path, ds: str, n: int):
    T, PT, ETA, NH = st.targets(n)
    fig, ax = plt.subplots(2, 4, figsize=(22, 10))
    a = ax.flatten()
    for i, (nm, lab, lg) in enumerate(TARGETS):
        counts(a[i], T[:, i]); a[i].set_xlabel(lab); a[i].set_title(nm)
    counts(a[5], PT); a[5].set_xlabel("$p_T$ [GeV]"); a[5].set_title("$p_T$")
    counts(a[6], ETA, rng=(-3.05, 3.05)); a[6].set_xlabel(r"$\eta$"); a[6].set_title(r"$\eta$")
    counts(a[7], NH, discrete=True); a[7].set_xlabel("hits per track"); a[7].set_title("sequence length")
    fig.suptitle(f"{ds} — regression targets and kinematics, as preprocessed "
                 f"(variant core, no d0/z0 windows) — {len(T):,} tracks sampled", y=1.01)
    fig.tight_layout()
    save(fig, out / "targets_and_kinematics")


def plot_hits(st: Store, out: Path, ds: str, n: int):
    H = st.hits(n)
    fig, ax = plt.subplots(3, 4, figsize=(24, 15))
    a = ax.flatten()
    for i, (nm, lab, lg) in enumerate(HITF):
        disc = nm in ("volume_id", "layer_id", "detector")
        counts(a[i], H[:, i], discrete=disc, ylabel="hits")
        a[i].set_xlabel(lab); a[i].set_title(nm)
    fig.suptitle(f"{ds} — the 12 hit input features, as preprocessed — {len(H):,} hits sampled", y=1.005)
    fig.tight_layout()
    save(fig, out / "hit_features")


def _panels(fig):
    # equal aspect with |z| 2.8x wider than |r|: give the z panels the matching
    # width, or the x-y panel is squeezed to a sliver
    ax = fig.subplots(1, 3, gridspec_kw={"width_ratios": [RXY, RZ, RZ], "wspace": 0.25})
    for a, (xl, yl, xr, yr) in zip(ax, [("x [mm]", "y [mm]", RXY, RXY),
                                        ("z [mm]", "x [mm]", RZ, RXY),
                                        ("z [mm]", "y [mm]", RZ, RXY)]):
        a.set_xlabel(xl); a.set_ylabel(yl)
        a.set_xlim(-xr, xr); a.set_ylim(-yr, yr)
        a.set_aspect("equal", adjustable="box")
        a.grid(alpha=0.25)
    ax[0].set_title("x–y (beam view)"); ax[1].set_title("z–x"); ax[2].set_title("z–y")
    return ax


def _draw_event(ax, trks, colour, lw=0.8, ms=2.0):
    for t in trks:
        x, y, z = t[:, 0], t[:, 1], t[:, 2]
        for a, (u, v) in zip(ax, [(x, y), (z, x), (z, y)]):
            a.plot(u, v, "-", color=colour, lw=lw, alpha=0.85)
            a.plot(u, v, ".", color=colour, ms=ms)


def plot_events(st: Store, out: Path, ds: str, n_events: int, overlay: bool, seed: int,
                sort: str = "stored"):
    evs = st.events(n_events, seed=seed, sort=sort)
    tag = {"stored": "hit order: as stored (by tracker_hits.time)",
           "s": "hit order: re-sorted by s = sqrt(x^2+y^2+z^2)"}[sort]
    cmap = plt.get_cmap("tab10")
    if overlay:
        fig = plt.figure(figsize=(21, 5.2))
        ax = _panels(fig)
        for k, (eid, trks) in enumerate(evs):
            _draw_event(ax, trks, cmap(k % 10))
        fig.legend([Line2D([0], [0], color=cmap(k % 10), lw=2) for k in range(len(evs))],
                   [f"event {eid}" for eid, _ in evs], loc="upper center",
                   ncol=min(len(evs), 10), frameon=False, bbox_to_anchor=(0.5, 1.00), fontsize=8)
        ntr = sum(len(t) for _, t in evs)
        fig.suptitle(f"{ds} — {len(evs)} events overlaid, {ntr} tracks — {tag}\n"
                     f"axes fixed to the ODD envelope: |r| < {RXY:.0f} mm, |z| < {RZ:.0f} mm", y=1.13)
        fig.tight_layout()
        save(fig, out / f"overlay_{len(evs)}events")
    else:
        for k, (eid, trks) in enumerate(evs):
            fig = plt.figure(figsize=(21, 5.2))
            ax = _panels(fig)
            for j, t in enumerate(trks):
                _draw_event(ax, [t], cmap(j % 10), lw=0.7, ms=1.6)
            fig.suptitle(f"{ds} — event {eid}, {len(trks)} selected tracks — {tag}", y=1.02)
            fig.tight_layout()
            save(fig, out / f"event_{k:02d}_id{eid}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/scratch/colliderml/ICLR_retraining")
    ap.add_argument("--out", default="/shared/tracking/ssm-colliderml-track-regression/dataset_plots")
    ap.add_argument("--split", default="test")
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--max-tracks", type=int, default=1_500_000)
    ap.add_argument("--max-hits", type=int, default=3_000_000)
    ap.add_argument("--n-events", type=int, default=10)
    a = ap.parse_args()
    style()
    root, out = Path(a.root), Path(a.out)
    dsl = a.datasets or sorted(p.name for p in root.iterdir() if (p / a.split / "manifest.json").exists())
    for ds in dsl:
        st = Store(root / ds / a.split)
        print(f"=== {ds}  ({st.man['n_tracks']:,} tracks in {a.split}, {len(st.parts)} parts)", flush=True)
        plot_targets(st, out / "distributions" / ds, ds, a.max_tracks)
        plot_hits(st, out / "distributions" / ds, ds, a.max_hits)
        # one muon per event -> overlay is readable; ttbar has ~26 tracks/event -> one per figure
        for srt, sub in (("stored", "event_displays_stored_time_order"),
                         ("s", "event_displays_s_sorted")):
            plot_events(st, out / sub / ds, ds, a.n_events,
                        overlay=("ttbar" not in ds), seed=1, sort=srt)
        print(f"    done", flush=True)


if __name__ == "__main__":
    main()
