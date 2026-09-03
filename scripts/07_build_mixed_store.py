#!/usr/bin/env python3
"""Build a MIXED flat store by symlinking parts of two existing stores.

    python scripts/07_build_mixed_store.py --base <uniform store root> --extra <ttbar store root> \
        --extra-max-tracks 8000000 --out /scratch/colliderml/ICLR_retraining_geom_mixed8M

train = all base train parts + extra train parts (whole parts, in order) until
        `--extra-max-tracks` is reached;
val   = base val parts only (keeps the per-epoch val metrics comparable with the
        muon-only runs);  test = base test parts only.
Each part is a symlink to the original part directory (hits/hit_times/offsets/
lengths/targets[/acts_*].npy), so the store costs no disk.  Sanity checks: same
hit_sort_key, same hit feature width, same target width.
"""
import argparse, json, os, sys
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True); ap.add_argument("--extra", required=True)
ap.add_argument("--extra-max-tracks", type=int, default=8_000_000); ap.add_argument("--out", required=True)
ap.add_argument("--extra-val", action="store_true", help="also append the extra store's val parts to val (default: val = base only)")
a = ap.parse_args()
base, extra, out = Path(a.base), Path(a.extra), Path(a.out)

def man(root, sp): return json.loads((root / sp / "manifest.json").read_text())
mb, me = man(base, "train"), man(extra, "train")
for k in ("layout", "hit_sort_key"):
    assert mb.get(k) == me.get(k), (k, mb.get(k), me.get(k))
hb = np.load(base / "train" / mb["parts"][0]["name"] / "hits.npy", mmap_mode="r")
he = np.load(extra / "train" / me["parts"][0]["name"] / "hits.npy", mmap_mode="r")
assert hb.shape[1] == he.shape[1], (hb.shape, he.shape)

def link_parts(sp, sources):
    d = out / sp; d.mkdir(parents=True, exist_ok=True)
    parts, k, tot = [], 0, {}
    for root, m, cap in sources:
        taken = 0
        for p in m["parts"]:
            if cap is not None and taken >= cap: break
            name = f"part_{k:04d}"; k += 1
            dst = d / name
            if dst.is_symlink() or dst.exists(): dst.unlink()
            os.symlink((root / sp / p["name"]).resolve(), dst)
            parts.append({"name": name, "n_tracks": p["n_tracks"], "n_hits": p["n_hits"], "source": str(root / sp / p["name"])})
            taken += p["n_tracks"]
        tot[str(root)] = taken
    new = dict(mb); new["parts"] = parts
    new["n_tracks"] = sum(p["n_tracks"] for p in parts); new["n_hits"] = sum(p["n_hits"] for p in parts)
    new["mixed_from"] = tot
    (d / "manifest.json").write_text(json.dumps(new, indent=1))
    print(f"{sp}: {len(parts)} parts, {new['n_tracks']:,} tracks  <- " + ", ".join(f"{Path(r).name}={n:,}" for r, n in tot.items()))

link_parts("train", [(base, mb, None), (extra, me, a.extra_max_tracks)])
link_parts("val",   [(base, man(base, "val"), None)] + ([(extra, man(extra, "val"), None)] if a.extra_val else []))
link_parts("test",  [(base, man(base, "test"), None)])
meta = json.loads((base / "dataset_meta.json").read_text())
meta["mixed"] = {"base": str(base), "extra": str(extra), "extra_max_tracks": a.extra_max_tracks, "extra_val": a.extra_val,
                 "note": "train = base train + extra train parts (symlinks); test = base only; val = base" + (" + extra val" if a.extra_val else " only")}
(out / "dataset_meta.json").write_text(json.dumps(meta, indent=1))
print("wrote", out)
