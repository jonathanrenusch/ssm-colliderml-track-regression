#!/usr/bin/env python3
"""Build a mixed training store by symlinking the parts of two existing stores.

    python scripts/build_mixed_store.py --base <store> --extra <store> --out <mixed store>

train = base train parts + extra train parts,
val   = base val parts + extra val parts,
test  = base test parts only.
Each part is a symlink to the original part directory, so the mixed store costs
no disk; its manifests record the source of every part (``source``) and the
number of tracks taken from each store (``mixed_from``).  Both stores must use
the same layout, hit order and hit-feature width.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--base", required=True)
ap.add_argument("--extra", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
base, extra, out = Path(a.base), Path(a.extra), Path(a.out)


def man(root, sp):
    return json.loads((root / sp / "manifest.json").read_text())


mb, me = man(base, "train"), man(extra, "train")
for k in ("layout", "hit_sort_key"):
    assert mb.get(k) == me.get(k), (k, mb.get(k), me.get(k))
hb = np.load(base / "train" / mb["parts"][0]["name"] / "hits.npy", mmap_mode="r")
he = np.load(extra / "train" / me["parts"][0]["name"] / "hits.npy", mmap_mode="r")
assert hb.shape[1] == he.shape[1], (hb.shape, he.shape)


def link_parts(sp, roots):
    d = out / sp
    d.mkdir(parents=True, exist_ok=True)
    parts, taken = [], {}
    for root in roots:
        m = man(root, sp)
        taken[str(root)] = sum(p["n_tracks"] for p in m["parts"])
        for p in m["parts"]:
            dst = d / f"part_{len(parts):04d}"
            if dst.is_symlink() or dst.exists():
                dst.unlink()
            os.symlink((root / sp / p["name"]).resolve(), dst)
            parts.append({"name": dst.name, "n_tracks": p["n_tracks"], "n_hits": p["n_hits"],
                          "source": str(root / sp / p["name"])})
    new = dict(mb, parts=parts, n_tracks=sum(p["n_tracks"] for p in parts),
               n_hits=sum(p["n_hits"] for p in parts), mixed_from=taken)
    (d / "manifest.json").write_text(json.dumps(new, indent=1))
    print(f"{sp}: {len(parts)} parts, {new['n_tracks']:,} tracks  <- "
          + ", ".join(f"{Path(r).name}={n:,}" for r, n in taken.items()))


link_parts("train", [base, extra])
link_parts("val", [base, extra])
link_parts("test", [base])
meta = json.loads((base / "dataset_meta.json").read_text())
meta["mixed"] = {"base": str(base), "extra": str(extra),
                 "note": "train/val = base + extra parts (symlinks); test = base only"}
(out / "dataset_meta.json").write_text(json.dumps(meta, indent=1))
print("wrote", out)
