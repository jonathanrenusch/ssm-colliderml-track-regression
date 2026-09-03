#!/usr/bin/env python3
"""Visual check of the per-track hit order stored in a flat store.

For a store written in GEOMETRY order (``preprocess_flat.py --sort-key geometry``)
this draws, per dataset:

* ``<ds>__order_check_tracks.pdf`` — six tracks, one row each; columns are the
  z–r view with hits connected in ``s``-from-origin order, the same view in the
  stored geometry order, the x–y view in geometry order, and the distance of
  each hit from the TRUTH perigee plotted against its position in the sequence
  for both orders (a correct order is a monotonic staircase; the ``s`` order
  dips where it swaps hits).  Hits are numbered
  in sequence order and coloured by detector group (pixel / short strip / long
  strip); the star is the truth perigee (z0, |d0|), the cross the origin.  The
  first four tracks are ones where the two orders DISAGREE (chosen at large
  |z0|, where ``s`` fails), the last two are tracks where they agree.  A wrong
  order shows up as a line that runs back and forth in z–r.
* ``<ds>__order_disagreement_vs_z0.pdf`` — fraction of tracks whose ``s`` order
  differs from the geometry order, vs |z0| and vs pT (whole test part).
* ``<ds>__overlay_10events_geometry_order.pdf`` — the full-detector overlay in
  the style of ``dataset_plots/event_displays_*`` (x–y, z–x, z–y, fixed ODD
  axes), hits connected in the stored (geometry) order.

    python scripts/sorting_visual_check.py --store-root /scratch/colliderml/ICLR_retraining_geom \
        --out-dir eval_plots/baselines_KF_rebuilt_geom/sorting_check
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from track_regression.hit_sorting import geometry_order  # noqa: E402

GROUP = {16: 0, 17: 0, 18: 0, 23: 1, 24: 1, 25: 1, 28: 2, 29: 2, 30: 2}
GROUP_NAME = ["pixel", "short strip", "long strip"]
GROUP_COL = ["C0", "C1", "C3"]
RXY, RZ = 1100.0, 3100.0


def load_part(store_dir: Path, part: str | None = None):
    man = json.loads((store_dir / "manifest.json").read_text())
    name = part or man["parts"][0]["name"]
    d = store_dir / name
    return dict(name=name, hits=np.load(d / "hits.npy", mmap_mode="r"), off=np.load(d / "offsets.npy"),
                ln=np.load(d / "lengths.npy"), tg=np.load(d / "targets.npy"),
                ev=np.load(d / "track_event_ids.npy"), sort_key=man.get("hit_sort_key", "?"))


def track(P, i):
    a = int(P["off"][i]); return np.asarray(P["hits"][a:a + int(P["ln"][i])])


def kin(tg):
    d0, z0, phi, th, qop = tg
    pt = np.sin(th) / abs(qop); eta = -np.log(np.tan(th / 2))
    return pt, eta, d0, z0


def orders_differ(h, stored_is_geometry=True):
    s_order = np.argsort(h[:, 6], kind="stable")
    g_order = geometry_order(h[:, :3], h[:, 7])
    return not np.array_equal(s_order, np.arange(len(h))) if stored_is_geometry else not np.array_equal(s_order, g_order)


def draw_zr(ax, h, order, tg, title):
    r = np.hypot(h[:, 0], h[:, 1]); z = h[:, 2]
    hh = h[order]; rr = r[order]; zz = z[order]
    ax.plot(zz, rr, "-", color="0.35", lw=1.0, zorder=1)
    for g in range(3):
        m = np.array([GROUP.get(int(v), -1) == g for v in hh[:, 7]])
        if m.any():
            ax.plot(zz[m], rr[m], "o", color=GROUP_COL[g], ms=5, zorder=2, label=GROUP_NAME[g])
    for k, (a, b) in enumerate(zip(zz, rr)):
        ax.annotate(str(k + 1), (a, b), textcoords="offset points", xytext=(3, 3), fontsize=6)
    d0, z0 = tg[0], tg[1]
    ax.plot([z0], [abs(d0)], "*", color="k", ms=9, zorder=3, label="truth perigee")
    ax.plot([0], [0], "x", color="0.5", ms=7, zorder=3, label="origin")
    ax.set_xlabel("z [mm]"); ax.set_ylabel("r [mm]"); ax.set_title(title, fontsize=9)
    ax.grid(alpha=0.25)
    zlo, zhi = min(zz.min(), z0, 0), max(zz.max(), z0, 0); pad = 0.08 * (zhi - zlo + 1)
    ax.set_xlim(zlo - pad, zhi + pad); ax.set_ylim(-0.05 * rr.max(), 1.12 * rr.max())


def draw_xy(ax, h, order, tg, title):
    hh = h[order]
    ax.plot(hh[:, 0], hh[:, 1], "-", color="0.35", lw=1.0, zorder=1)
    for g in range(3):
        m = np.array([GROUP.get(int(v), -1) == g for v in hh[:, 7]])
        if m.any():
            ax.plot(hh[m, 0], hh[m, 1], "o", color=GROUP_COL[g], ms=5, zorder=2)
    for k, (a, b) in enumerate(zip(hh[:, 0], hh[:, 1])):
        ax.annotate(str(k + 1), (a, b), textcoords="offset points", xytext=(3, 3), fontsize=6)
    d0, z0, phi = tg[0], tg[1], tg[2]
    ax.plot([-d0 * np.sin(phi)], [d0 * np.cos(phi)], "*", color="k", ms=9, zorder=3)
    ax.plot([0], [0], "x", color="0.5", ms=7, zorder=3)
    ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]"); ax.set_title(title, fontsize=9)
    # square window around the hits and the origin (equal aspect, fixed box)
    cx, cy = 0.5 * (hh[:, 0].min() + hh[:, 0].max()), 0.5 * (hh[:, 1].min() + hh[:, 1].max())
    half = 0.6 * max(hh[:, 0].max() - hh[:, 0].min(), hh[:, 1].max() - hh[:, 1].min(), 60.0)
    half = max(half, abs(cx) + 30, abs(cy) + 30)
    ax.set_xlim(cx - half, cx + half); ax.set_ylim(cy - half, cy + half)
    ax.set_aspect("equal", adjustable="box"); ax.grid(alpha=0.25)


def draw_monotonic(ax, h, s_order, g_order, tg):
    """|X - P| (distance from the truth perigee, a truth-based proxy for the
    ACTS time order) against the position in the sequence, for both orders."""
    d0, z0, phi = tg[0], tg[1], tg[2]
    P = np.array([-d0 * np.sin(phi), d0 * np.cos(phi), z0])
    dist = np.sqrt(((h[:, :3] - P) ** 2).sum(1))
    k = np.arange(1, len(h) + 1)
    ax.plot(k, dist[s_order], "s--", color="C3", ms=5, lw=1.0, label="s = |X| from origin")
    ax.plot(k, dist[g_order], "o-", color="C0", ms=4, lw=1.2, label="geometry order (stored)")
    bad = np.nonzero(np.diff(dist[s_order]) < 0)[0]
    for b in bad:
        ax.axvspan(k[b] - 0.5, k[b] + 1.5, color="C3", alpha=0.12)
    ax.set_xlabel("position in the hit sequence"); ax.set_ylabel("|X − truth perigee| [mm]")
    ax.set_title("|X − truth perigee| vs position in sequence (must rise monotonically)", fontsize=9)
    ax.grid(alpha=0.25); ax.legend(fontsize=7, loc="upper left")


def plot_tracks(P, ds, out: Path, rng, n_dis=4, n_agree=2, scan=20000):
    n = min(scan, len(P["ln"]))
    dis, agree = [], []
    for i in range(n):
        h = track(P, i)
        (dis if orders_differ(h) else agree).append(i)
    frac = len(dis) / n
    # disagreeing tracks: prefer large |z0| (where s fails), but keep variety
    z0 = np.abs(P["tg"][:n, 1])
    dis_sorted = sorted(dis, key=lambda i: -z0[i])
    pick = list(dis_sorted[:2]) + list(rng.choice(dis, size=min(n_dis - 2, len(dis)), replace=False)) if dis else []
    pick += list(rng.choice(agree, size=min(n_agree, len(agree)), replace=False))
    fig, axes = plt.subplots(len(pick), 4, figsize=(24, 5.0 * len(pick)),
                             gridspec_kw={"width_ratios": [1.15, 1.15, 1.0, 1.15]})
    axes = np.atleast_2d(axes)
    for row, i in enumerate(pick):
        h = track(P, i); tg = P["tg"][i]; pt, eta, d0, z0i = kin(tg)
        s_order = np.argsort(h[:, 6], kind="stable"); g_order = np.arange(len(h))
        same = np.array_equal(s_order, g_order)
        info = (f"track {i} (event {int(P['ev'][i])}): pT={pt:.2f} GeV  η={eta:+.2f}  d0={d0:+.2f} mm  "
                f"z0={z0i:+.1f} mm  {len(h)} hits — s-order {'==' if same else '!='} geometry-order")
        draw_zr(axes[row, 0], h, s_order, tg, "z–r, hits connected in s = |X| order (from the ORIGIN)")
        draw_zr(axes[row, 1], h, g_order, tg, "z–r, hits connected in stored GEOMETRY order")
        draw_xy(axes[row, 2], h, g_order, tg, "x–y, geometry order")
        draw_monotonic(axes[row, 3], h, s_order, g_order, tg)
        axes[row, 0].text(0.0, 1.14, info, transform=axes[row, 0].transAxes, fontsize=10,
                          color=("C3" if not same else "C2"), fontweight="bold")
    axes[0, 0].legend(loc="lower right", fontsize=7)
    fig.suptitle(f"{ds} — per-track hit-order check on the rebuilt store ({P['name']}, stored order = {P['sort_key']})\n"
                 f"{100*frac:.1f} % of the first {n:,} tracks have an s-order ≠ geometry-order; "
                 f"rows 1–{min(n_dis,len(dis))}: disagreeing tracks (first two at the largest |z0|), last rows: agreeing tracks",
                 y=0.995, fontsize=11)
    fig.subplots_adjust(left=0.045, right=0.99, top=0.955, bottom=0.03, hspace=0.62, wspace=0.28)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{ds}__order_check_tracks.pdf"); plt.close(fig)
    return n, dis, agree


def plot_disagreement(P, ds, out: Path, n, dis):
    tg = P["tg"][:n]; z0 = np.abs(tg[:, 1]); pt = np.sin(tg[:, 3]) / np.abs(tg[:, 4])
    d = np.zeros(n, bool); d[dis] = True
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for a, x, edges, lab in [(ax[0], z0, np.linspace(0, 240, 13), "|z0| [mm]"),
                             (ax[1], pt, np.array([0.5, 1, 2, 3, 5, 10, 20, 50, 110, 300]), "pT [GeV]")]:
        idx = np.clip(np.digitize(x, edges) - 1, 0, len(edges) - 2)
        tot = np.bincount(idx, minlength=len(edges) - 1); bad = np.bincount(idx, weights=d, minlength=len(edges) - 1)
        f = np.where(tot > 0, bad / np.maximum(tot, 1), np.nan)
        a.step(edges, np.r_[f, f[-1]] * 100, where="post", color="C3")
        a.set_xlabel(lab); a.set_ylabel("% tracks with s-order ≠ geometry-order"); a.grid(alpha=0.3)
        if lab.startswith("pT"): a.set_xscale("log")
    fig.suptitle(f"{ds} — where the legacy s-from-origin order disagrees with the geometry order "
                 f"({100*d.mean():.1f} % overall, {n:,} tracks of {P['name']})", y=1.02)
    fig.tight_layout(); fig.savefig(out / f"{ds}__order_disagreement_vs_z0.pdf"); plt.close(fig)


def plot_overlay(P, ds, out: Path, rng, n_events=10):
    ev = np.asarray(P["ev"]); uniq = np.unique(ev)
    pick = rng.choice(len(uniq), size=min(n_events, len(uniq)), replace=False)
    fig = plt.figure(figsize=(22, 6.2))
    ax = fig.subplots(1, 3, gridspec_kw={"width_ratios": [RXY, RZ, RZ], "wspace": 0.25})
    for a, (xl, yl, xr, yr) in zip(ax, [("x [mm]", "y [mm]", RXY, RXY), ("z [mm]", "x [mm]", RZ, RXY), ("z [mm]", "y [mm]", RZ, RXY)]):
        a.set_xlabel(xl); a.set_ylabel(yl); a.set_xlim(-xr, xr); a.set_ylim(-yr, yr)
        a.set_aspect("equal", adjustable="box"); a.grid(alpha=0.25)
    ax[0].set_title("x–y (beam view)"); ax[1].set_title("z–x"); ax[2].set_title("z–y")
    cmap = plt.get_cmap("tab10")
    for k, u in enumerate(uniq[pick]):
        for i in np.nonzero(ev == u)[0]:
            t = track(P, i); x, y, z = t[:, 0], t[:, 1], t[:, 2]
            for a, (uu, vv) in zip(ax, [(x, y), (z, x), (z, y)]):
                a.plot(uu, vv, "-", color=cmap(k % 10), lw=0.8, alpha=0.85); a.plot(uu, vv, ".", color=cmap(k % 10), ms=2.0)
    fig.suptitle(f"{ds} — {len(pick)} events overlaid, hit order: as stored ({P['sort_key']} order) — {P['name']}", y=1.02)
    fig.tight_layout(); fig.savefig(out / f"{ds}__overlay_{len(pick)}events_{P['sort_key']}_order.pdf"); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--store-root", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    root, out = Path(a.store_root), Path(a.out_dir)
    names = a.datasets or sorted(p.name for p in root.iterdir() if (p / a.split / "manifest.json").exists())
    for ds in names:
        P = load_part(root / ds / a.split)
        if P["sort_key"] != "geometry":
            print(f"[skip] {ds}: store order is {P['sort_key']!r}, this check assumes a geometry-ordered store"); continue
        rng = np.random.default_rng(a.seed)
        n, dis, agree = plot_tracks(P, ds, out, rng)
        plot_disagreement(P, ds, out, n, dis)
        plot_overlay(P, ds, out, rng)
        print(f"{ds}: {len(dis)}/{n} tracks ({100*len(dis)/n:.1f} %) with s-order != geometry-order", flush=True)


if __name__ == "__main__":
    main()
