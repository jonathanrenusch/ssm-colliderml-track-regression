"""Legacy (arxiv_retraining) dataset accounting: exact per-split counts of
events / tracks / hits, per-pT / per-|d0| / per-eta breakdowns, read from
every shard's selected_tracks/{track_targets,track_meta,track_event_idx,
track_hit_offsets}.npy.  Writes JSON + a text table."""
import json, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np

EOS = Path("/eos/project/e/end-to-end-colliderml/data/arxiv_retraining")
SCR = Path("/scratch/colliderml/arxiv_retraining")
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("docs")
VARIANTS = ["p0_core_pretrain", "p0_core_kf_hits_pretrain",
            "p200_core_kf_matched_finetune", "p200_core_kf_hits_finetune"]

PT_EDGES = np.array([0.5, 1, 2, 3, 5, 10, 20, 50, 110, np.inf])
D0_EDGES = np.array([0, 0.03, 0.1, 0.3, 1.0, 2.5, np.inf])  # mm
ETA_EDGES = np.array([-3, -2, -1, 0, 1, 2, 3])
Z0_EDGES = np.array([0, 50, 100, 150, 200, np.inf])


def shard_stats(root: Path, i: int):
    d = root / f"shard_{i:04d}" / "selected_tracks"
    tg = np.load(d / "track_targets.npy")           # (N,5) d0 z0 phi theta qop
    meta = np.load(d / "track_meta.npy")            # (N,2) pt, vertex_primary
    ev = np.load(d / "track_event_idx.npy")
    off = np.load(d / "track_hit_offsets.npy")
    n = len(tg)
    pt = meta[:, 0].astype(np.float64)
    d0 = np.abs(tg[:, 0]).astype(np.float64)
    z0 = np.abs(tg[:, 1]).astype(np.float64)
    theta = tg[:, 3].astype(np.float64)
    eta = -np.log(np.tan(theta / 2))
    nh = np.diff(off).astype(np.int64)
    out = dict(
        n_tracks=int(n), n_hits=int(off[-1]), n_events=int(len(np.unique(ev))),
        vertex_primary_1=int((meta[:, 1] == 1).sum()),
        pt_hist=np.histogram(pt, PT_EDGES)[0].tolist(),
        d0_hist=np.histogram(d0, D0_EDGES)[0].tolist(),
        z0_hist=np.histogram(z0, Z0_EDGES)[0].tolist(),
        eta_hist=np.histogram(eta, ETA_EDGES)[0].tolist(),
        pt_d0_hist=np.histogram2d(pt, d0, [PT_EDGES, D0_EDGES])[0].astype(int).tolist(),
        pt_eta_hist=np.histogram2d(pt, eta, [PT_EDGES, ETA_EDGES])[0].astype(int).tolist(),
        nhits_hist=np.bincount(nh, minlength=21)[:21].tolist(),
        pt_sum=float(pt.sum()), pt_med_sample=float(np.median(pt)),
    )
    return out


def merge(stats):
    m = {}
    for s in stats:
        for k, v in s.items():
            if isinstance(v, list):
                m[k] = (np.array(m[k]) + np.array(v)).tolist() if k in m else v
            elif k == "pt_med_sample":
                m.setdefault("pt_med_samples", []).append(v)
            else:
                m[k] = m.get(k, 0) + v
    return m


def main():
    results = {}
    for var in VARIANTS:
        root = SCR / var if (SCR / var / "shard_0000").exists() else EOS / var
        split = json.load(open(EOS / var / "split.json"))
        man = json.load(open(EOS / var / "manifest.json"))
        t0 = time.time()
        with ThreadPoolExecutor(48) as ex:
            per_shard = list(ex.map(lambda i: shard_stats(root, i), range(man["num_shards"])))
        res = {"root": str(root), "manifest_totals": man["totals"], "selection": man["selection"],
               "splits": {}}
        for sp in ("train", "val", "test"):
            res["splits"][sp] = merge([per_shard[i] for i in split[sp]])
            res["splits"][sp]["n_shards"] = len(split[sp])
        res["all"] = merge(per_shard)
        results[var] = res
        print(f"{var}: {len(per_shard)} shards in {time.time()-t0:.0f}s  "
              f"tracks={res['all']['n_tracks']:,} hits={res['all']['n_hits']:,} "
              f"events={res['all']['n_events']:,}", flush=True)
    json.dump({"pt_edges": PT_EDGES.tolist(), "d0_edges_mm": D0_EDGES.tolist(),
               "eta_edges": ETA_EDGES.tolist(), "z0_edges_mm": Z0_EDGES.tolist(),
               "variants": results}, open(OUT / "legacy_accounting.json", "w"), indent=1)

    # ---- text table
    lines = []
    for var, res in results.items():
        lines.append(f"\n=== {var}  ({res['root']}) ===")
        lines.append(f"selection: primary={res['selection']['primary']} hard_scatter={res['selection']['hard_scatter']} "
                     f"pt>={res['selection']['pt_min']} |d0|<={res['selection']['d0_max']} |z0|<={res['selection']['z0_max']} "
                     f"dm={res['selection']['require_acts_dm']} kf_hits={res['selection']['use_acts_hits_only']}")
        lines.append(f"{'split':6s} {'shards':>6s} {'events':>10s} {'tracks':>13s} {'hits':>15s} {'trk/evt':>8s} {'hits/trk':>8s}")
        for sp in ("train", "val", "test", "all"):
            s = res["splits"][sp] if sp != "all" else res["all"]
            ns = s.get("n_shards", res["manifest_totals"].get("n_events", 0) and len(per := []) or "")
            lines.append(f"{sp:6s} {str(s.get('n_shards','1000')):>6s} {s['n_events']:>10,} {s['n_tracks']:>13,} {s['n_hits']:>15,} "
                         f"{s['n_tracks']/max(s['n_events'],1):>8.1f} {s['n_hits']/max(s['n_tracks'],1):>8.2f}")
        a = res["all"]; tr = res["splits"]["train"]
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
        lines.append(f"  mean pT (all) = {a['pt_sum']/a['n_tracks']:.3f} GeV ; median pT (median of per-shard medians) = {np.median(a['pt_med_samples']):.3f} GeV")
        lines.append(f"  vertex_primary==1: {100*a['vertex_primary_1']/a['n_tracks']:.2f}% of tracks")
        lines.append("  nhits histogram (6..20): " + " ".join(f"{k}:{v:,}" for k, v in enumerate(a['nhits_hist']) if v))
    txt = "\n".join(lines)
    print(txt)
    open(OUT / "legacy_accounting.txt", "w").write(txt)


if __name__ == "__main__":
    main()
