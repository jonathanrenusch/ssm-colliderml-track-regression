#!/usr/bin/env python3
"""Create the eval symlink farm for a store root: <eval_root>/<ds>/test = union of the store's splits
(train+val+test parts of the fixed-pT / ttbar sets the model never trains on); uniform is linked as is.
    python scripts/build_eval_farm.py --store-root /scratch/colliderml/ICLR_retraining_geom_B3 --eval-root /scratch/colliderml/ICLR_eval_geom_B3 \
        --union single_muon_2GeV single_muon_10GeV single_muon_100GeV ttbar ttbar_new_pt1 --link single_muon_uniform
"""
import argparse, json, os
from pathlib import Path
ap = argparse.ArgumentParser(); ap.add_argument("--store-root", required=True); ap.add_argument("--eval-root", required=True)
ap.add_argument("--union", nargs="*", default=[]); ap.add_argument("--link", nargs="*", default=[])
a = ap.parse_args(); OUT, EV = Path(a.store_root), Path(a.eval_root); EV.mkdir(parents=True, exist_ok=True)
for ds in a.link:
    dst = EV / ds
    if dst.is_symlink() or dst.exists(): dst.unlink()
    os.symlink(OUT / ds, dst); print(ds, "-> linked")
for ds in a.union:
    if not (OUT / ds).exists(): print(ds, "MISSING in store root, skipped"); continue
    d = EV / ds; (d / "test").mkdir(parents=True, exist_ok=True)
    for sp in ("train", "val"):
        if not (d / sp).exists() and (OUT / ds / sp).exists(): os.symlink(OUT / ds / sp, d / sp)
    parts, k, man = [], 0, None
    for sp in ("train", "val", "test"):
        mp = OUT / ds / sp / "manifest.json"
        if not mp.exists(): continue
        m = json.load(open(mp)); man = man or m
        for p in m["parts"]:
            name = f"part_{k:04d}"; k += 1
            dst = d / "test" / name
            if dst.is_symlink() or dst.exists(): dst.unlink()
            os.symlink(OUT / ds / sp / p["name"], dst)
            parts.append({"name": name, "n_tracks": p["n_tracks"], "n_hits": p["n_hits"]})
    man["parts"] = parts; man["n_tracks"] = sum(p["n_tracks"] for p in parts); man["n_hits"] = sum(p["n_hits"] for p in parts)
    json.dump(man, open(d / "test" / "manifest.json", "w"), indent=1); print(ds, f"{man['n_tracks']:,} eval tracks")
