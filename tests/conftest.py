"""Shared fixture: a small synthetic flat store of helix tracks."""

from __future__ import annotations

import json

import numpy as np
import pytest

from tests.test_seed import helix_points, random_params
from track_regression.seed import DEFAULT_BZ, KAPPA


def _write_part(d, hits, lens, targets):
    d.mkdir(parents=True)
    off = np.zeros(len(lens) + 1, np.int64)
    np.cumsum(lens, out=off[1:])
    np.save(d / "hits.npy", hits.astype(np.float32))
    np.save(d / "offsets.npy", off)
    np.save(d / "lengths.npy", lens.astype(np.int32))
    np.save(d / "targets.npy", targets.astype(np.float32))
    return {"name": d.name, "n_tracks": int(len(lens)), "n_hits": int(off[-1])}


def make_store(root, n_parts=(137, 91), seed=0):
    """``root/{train,val,test}`` with helix tracks of 6-20 hits in 12 features."""
    rng = np.random.default_rng(seed)
    for split in ("train", "val", "test"):
        parts = []
        for pi, n in enumerate(n_parts):
            d0, z0, phi, theta, qop = random_params(n)
            lens = rng.integers(6, 21, size=n)
            hits = []
            for i in range(n):
                R = np.sin(theta[i]) / abs(qop[i]) / (KAPPA * DEFAULT_BZ)
                pts = helix_points(d0[i], z0[i], phi[i], theta[i], qop[i],
                                   np.sort(rng.uniform(30.0, 1000.0, lens[i])) / R)
                r = np.hypot(pts[:, 0], pts[:, 1])
                s = np.linalg.norm(pts, axis=1)
                f = np.zeros((lens[i], 12))
                f[:, :3], f[:, 3], f[:, 6] = pts, r, s
                f[:, 4] = np.arctan2(pts[:, 1], pts[:, 0])
                f[:, 5] = np.arccos(np.clip(pts[:, 2] / s, -1, 1))
                f[:, 7] = np.where(r < 200, 16, 23)
                f[:, 11] = -np.log(np.tan(f[:, 5] / 2))
                hits.append(f)
            parts.append(_write_part(root / split / f"part_{pi:04d}", np.concatenate(hits), lens,
                                     np.stack([d0, z0, phi, theta, qop], 1)))
        (root / split / "manifest.json").write_text(json.dumps({
            "layout": "flat_csr", "n_feat": 12, "parts": parts,
            "n_tracks": sum(p["n_tracks"] for p in parts), "n_hits": sum(p["n_hits"] for p in parts)}))
    return root


@pytest.fixture(scope="session")
def synthetic_store(tmp_path_factory):
    return make_store(tmp_path_factory.mktemp("store"))
