#!/usr/bin/env python3
"""Drop hits that fail the ACTS converter's sensor-bounds projection check.

The v2 production contains rare hits whose (volume, layer, surface) triplet maps
to a gen3 sensor the hit does not lie on (>25 mm outside its bounds); pyacts'
``ColliderMLRelease1InputConverter`` THROWS on the first such hit and kills the
whole run.  This filter replicates the exact check offline — project each hit
into its mapped surface's frame (frame derived from three ``localToGlobal``
probes) and test ``insideBounds`` with the same tolerance — and rewrites the
shimmed ``tracker_hits`` parquet without the offending hits.

Usage: filter_acts_hits.py <tolerance_mm> <tracker_hits.parquet> [...]
Rewrites each file in place; prints per-file drop counts.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

import acts
from acts.json import TrackingGeometryJsonConverter

ODD = Path("/scratch/colliderml/odd-json")

gctx = acts.GeometryContext.dangerouslyDefaultConstruct()
geom = TrackingGeometryJsonConverter().fromJson(gctx, (ODD / "odd.json").absolute())
smap = geom.geoIdSurfaceMap()
g3 = {(int(r["gen1_volume"]), int(r["gen1_layer"]), int(r["gen1_sensitive"])): int(r["gen3_packed"])
      for r in csv.DictReader(open(ODD / "geoid_map.csv"))}

_FRAMES: dict[tuple, tuple] = {}


def frame(tri):
    """(center, ex, ey, surface) for a ColliderML (vol, layer, surface) triplet."""
    hit = _FRAMES.get(tri)
    if hit is not None:
        return hit
    surf = smap[acts.GeometryIdentifier(g3[tri])]
    d = acts.Vector3(0.0, 0.0, 1.0)
    l2g = lambda x, y: np.array(surf.localToGlobal(gctx, acts.Vector2(x, y), d))  # noqa: E731
    c = l2g(0, 0)
    fr = (c, l2g(1, 0) - c, l2g(0, 1) - c, surf)
    _FRAMES[tri] = fr
    return fr


def filter_file(path: Path, tol_mm: float) -> None:
    bt = acts.BoundaryTolerance.absoluteEuclidean(tol_mm)
    t = pq.read_table(path)
    cols = {n: [] for n in t.column_names}
    n_drop = n_tot = 0
    for i in range(t.num_rows):
        x = np.asarray(t["x"][i].values, np.float64)
        y = np.asarray(t["y"][i].values, np.float64)
        z = np.asarray(t["z"][i].values, np.float64)
        v = np.asarray(t["volume_id"][i].values); l = np.asarray(t["layer_id"][i].values)
        s = np.asarray(t["surface_id"][i].values)
        keep = np.ones(len(x), bool)
        pts = np.stack([x, y, z], 1)
        for j in range(len(x)):
            tri = (int(v[j]), int(l[j]), int(s[j]))
            if tri not in g3:
                keep[j] = False
                continue
            c, ex, ey, surf = frame(tri)
            rel = pts[j] - c
            if not surf.insideBounds(acts.Vector2(float(rel @ ex), float(rel @ ey)), bt):
                keep[j] = False
        n_tot += len(x); n_drop += int((~keep).sum())
        for n in t.column_names:
            if n == "event_id":
                cols[n].append(t[n][i].as_py())
            else:
                vals = t[n][i].values.to_numpy(zero_copy_only=False)
                cols[n].append(vals[keep].tolist())
    schema = t.schema
    out = pa.table({n: (cols[n] if n == "event_id" else pa.array(cols[n], type=schema.field(n).type))
                    for n in t.column_names}, schema=schema)
    pq.write_table(out, path)
    print(f"[filter] {path}: dropped {n_drop} / {n_tot:,} hits", flush=True)


if __name__ == "__main__":
    tol = float(sys.argv[1])
    for f in sys.argv[2:]:
        filter_file(Path(f), tol)
