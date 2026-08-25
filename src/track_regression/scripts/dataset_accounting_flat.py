"""ICLR_retraining (drift_beamspot) flat-store accounting with the SAME bins as
legacy_accounting.py, read from every part's targets.npy / lengths.npy."""
import json, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np

ROOT = Path("/scratch/colliderml/ICLR_retraining")
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("docs")
DATASETS = ["single_muon_uniform", "single_muon_2GeV", "single_muon_10GeV", "single_muon_100GeV", "ttbar"]
PT_EDGES = np.array([0.5, 1, 2, 3, 5, 10, 20, 50, 110, np.inf])
D0_EDGES = np.array([0, 0.03, 0.1, 0.3, 1.0, 2.5, np.inf])
ETA_EDGES = np.array([-3, -2, -1, 0, 1, 2, 3])
Z0_EDGES = np.array([0, 50, 100, 150, 200, np.inf])
# finer grid for the coverage analysis (pT x eta x |d0|)
PT_F = np.array([0.5, 1, 1.5, 2, 3, 5, 7, 10, 15, 20, 30, 50, 75, 110, np.inf])
ETA_F = np.linspace(-3, 3, 13)
D0_F = np.array([0, 1, 2, 3, 4, 5, 6, 7.2])


def part_stats(p: Path):
    tg = np.load(p / "targets.npy", mmap_mode="r")
    ln = np.load(p / "lengths.npy", mmap_mode="r")
    tg = np.asarray(tg, dtype=np.float64)
    d0 = np.abs(tg[:, 0]); z0 = np.abs(tg[:, 1]); theta = tg[:, 3]; qop = tg[:, 4]
    pt = np.sin(theta) / np.abs(qop)
    eta = -np.log(np.tan(theta / 2))
    ev = np.load(p / "track_event_ids.npy", mmap_mode="r")
    return dict(
        n_tracks=int(len(tg)), n_hits=int(np.asarray(ln, dtype=np.int64).sum()),
        n_events=int(len(np.unique(np.asarray(ev)))),
        pt_hist=np.histogram(pt, PT_EDGES)[0].tolist(),
        d0_hist=np.histogram(d0, D0_EDGES)[0].tolist(),
        z0_hist=np.histogram(z0, Z0_EDGES)[0].tolist(),
        eta_hist=np.histogram(eta, ETA_EDGES)[0].tolist(),
        pt_d0_hist=np.histogram2d(pt, d0, [PT_EDGES, D0_EDGES])[0].astype(int).tolist(),
        pt_eta_hist=np.histogram2d(pt, eta, [PT_EDGES, ETA_EDGES])[0].astype(int).tolist(),
        fine=np.histogramdd(np.stack([pt, eta, d0], 1), [PT_F, ETA_F, D0_F])[0].astype(int).tolist(),
        nhits_hist=np.bincount(np.asarray(ln, dtype=np.int64), minlength=21)[:21].tolist(),
        pt_sum=float(pt.sum()), pt_min=float(pt.min()), pt_max=float(pt.max()),
    )


def merge(stats):
    m = {}
    for s in stats:
        for k, v in s.items():
            if isinstance(v, list):
                m[k] = (np.array(m[k]) + np.array(v)).tolist() if k in m else v
            elif k == "pt_min": m[k] = min(m.get(k, np.inf), v)
            elif k == "pt_max": m[k] = max(m.get(k, -np.inf), v)
            else: m[k] = m.get(k, 0) + v
    return m


def main():
    results = {}
    for ds in DATASETS:
        res = {"splits": {}}
        t0 = time.time()
        allp = []
        for sp in ("train", "val", "test"):
            man = json.load(open(ROOT / ds / sp / "manifest.json"))
            parts = [ROOT / ds / sp / p["name"] for p in man["parts"]]
            with ThreadPoolExecutor(32) as ex:
                st = list(ex.map(part_stats, parts))
            res["splits"][sp] = merge(st); res["splits"][sp]["n_parts"] = len(parts)
            allp += st
        res["all"] = merge(allp)
        results[ds] = res
        print(f"{ds}: {time.time()-t0:.0f}s tracks={res['all']['n_tracks']:,}", flush=True)
    json.dump({"pt_edges": PT_EDGES.tolist(), "d0_edges_mm": D0_EDGES.tolist(), "eta_edges": ETA_EDGES.tolist(),
               "z0_edges_mm": Z0_EDGES.tolist(), "fine_edges": {"pt": PT_F.tolist(), "eta": ETA_F.tolist(), "d0": D0_F.tolist()},
               "datasets": results}, open(OUT / "current_accounting.json", "w"), indent=1)
    lines = []
    for ds, res in results.items():
        lines.append(f"\n=== {ds} ===")
        lines.append(f"{'split':6s} {'parts':>6s} {'events':>11s} {'tracks':>13s} {'hits':>15s} {'trk/evt':>8s} {'hits/trk':>8s}")
        for sp in ("train", "val", "test", "all"):
            s = res["splits"][sp] if sp != "all" else res["all"]
            lines.append(f"{sp:6s} {str(s.get('n_parts','')):>6s} {s['n_events']:>11,} {s['n_tracks']:>13,} {s['n_hits']:>15,} "
                         f"{s['n_tracks']/max(s['n_events'],1):>8.2f} {s['n_hits']/max(s['n_tracks'],1):>8.2f}")
        tr = res["splits"]["train"]
        lines.append("pT bins [GeV]:      " + " ".join(f"{PT_EDGES[i]:>5g}-{PT_EDGES[i+1]:<5g}" for i in range(len(PT_EDGES)-1)))
        lines.append("  train tracks:     " + " ".join(f"{v:>11,}" for v in tr["pt_hist"]))
        lines.append("  train frac [%]:   " + " ".join(f"{100*v/tr['n_tracks']:>11.2f}" for v in tr["pt_hist"]))
        lines.append("|d0| bins [mm]:     " + " ".join(f"{D0_EDGES[i]:>5g}-{D0_EDGES[i+1]:<5g}" for i in range(len(D0_EDGES)-1)))
        lines.append("  train tracks:     " + " ".join(f"{v:>11,}" for v in tr["d0_hist"]))
        lines.append("  train frac [%]:   " + " ".join(f"{100*v/tr['n_tracks']:>11.2f}" for v in tr["d0_hist"]))
        lines.append("|z0| bins [mm]:     " + " ".join(f"{Z0_EDGES[i]:>5g}-{Z0_EDGES[i+1]:<5g}" for i in range(len(Z0_EDGES)-1)))
        lines.append("  train frac [%]:   " + " ".join(f"{100*v/tr['n_tracks']:>11.2f}" for v in tr["z0_hist"]))
        lines.append("eta bins:           " + " ".join(f"{ETA_EDGES[i]:>5g}-{ETA_EDGES[i+1]:<5g}" for i in range(len(ETA_EDGES)-1)))
        lines.append("  train frac [%]:   " + " ".join(f"{100*v/tr['n_tracks']:>11.2f}" for v in tr["eta_hist"]))
        a = res["all"]
        lines.append(f"  mean pT = {a['pt_sum']/a['n_tracks']:.3f} GeV ; pT range [{a['pt_min']:.3f}, {a['pt_max']:.3f}]")
        lines.append("  nhits histogram (6..20): " + " ".join(f"{k}:{v:,}" for k, v in enumerate(a['nhits_hist']) if v))
    txt = "\n".join(lines); print(txt); open(OUT / "current_accounting.txt", "w").write(txt)


if __name__ == "__main__":
    main()
