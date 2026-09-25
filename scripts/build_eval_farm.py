#!/usr/bin/env python3
"""Create the evaluation farm of a store root (symlinks only).

    python scripts/build_eval_farm.py --store-root <store_root> --eval-root <eval_root> \
        --union single_muon_2GeV single_muon_10GeV single_muon_50GeV --link single_muon_uniform

--union: <eval_root>/<ds>/test = the train + val + test parts of <store_root>/<ds>
         (the fixed-pT samples are never trained on, so all of it is test data);
--link:  <eval_root>/<ds> -> <store_root>/<ds> (its own test split is used).
"""
import argparse
import json
import os
from pathlib import Path

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--store-root", required=True)
ap.add_argument("--eval-root", required=True)
ap.add_argument("--union", nargs="*", default=[])
ap.add_argument("--link", nargs="*", default=[])
a = ap.parse_args()
OUT, EV = Path(a.store_root).resolve(), Path(a.eval_root)
EV.mkdir(parents=True, exist_ok=True)

for ds in a.link:
    dst = EV / ds
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    os.symlink(OUT / ds, dst)
    print(ds, "-> linked")

for ds in a.union:
    if not (OUT / ds).exists():
        raise SystemExit(f"{ds}: missing in {OUT}")
    d = EV / ds
    (d / "test").mkdir(parents=True, exist_ok=True)
    for sp in ("train", "val"):     # the data module recognises a store by its train/ manifest
        if not (d / sp).exists() and (OUT / ds / sp).exists():
            os.symlink(OUT / ds / sp, d / sp)
    parts, man = [], None
    for sp in ("train", "val", "test"):
        mp = OUT / ds / sp / "manifest.json"
        if not mp.exists():
            continue
        m = json.load(open(mp))
        man = man or m
        for p in m["parts"]:
            dst = d / "test" / f"part_{len(parts):04d}"
            if dst.is_symlink() or dst.exists():
                dst.unlink()
            os.symlink(OUT / ds / sp / p["name"], dst)
            parts.append({"name": dst.name, "n_tracks": p["n_tracks"], "n_hits": p["n_hits"]})
    man.update(parts=parts, n_tracks=sum(p["n_tracks"] for p in parts), n_hits=sum(p["n_hits"] for p in parts))
    json.dump(man, open(d / "test" / "manifest.json", "w"), indent=1)
    print(ds, f"{man['n_tracks']:,} eval tracks")
