"""Checks of the truth-free detector ordering in track_regression.hit_sorting."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from track_regression import hit_sorting as hs  # noqa: E402


def test_geometry_order_forward_track():
    """A straight eta~1.6 track through pixel barrel, short-strip barrel and discs."""
    theta = 2 * np.arctan(np.exp(-1.6))
    # (r, |z|, volume) of the ODD elements this track crosses, in path order
    elems = [(32, 76, 17), (68, 162, 17), (114, 271, 17), (170, 405, 17),
             (260, 619, 24), (360, 857, 24), (545, 1298, 25), (650, 1548, 25), (923, 2198, 30)]
    l = np.array([np.hypot(r, z) for r, z, _ in elems])
    dirv = np.array([np.sin(theta) * np.cos(0.4), np.sin(theta) * np.sin(0.4), np.cos(theta)])
    xyz = l[:, None] * dirv[None, :]
    vol = np.array([v for _, _, v in elems])
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(elems))
    got = hs.geometry_order(xyz[perm], vol[perm])
    assert np.array_equal(perm[got], np.arange(len(elems)))
    # the mirror track on the -z side uses volumes 16/23/28 and must order the same
    xyz_m = xyz * np.array([1, 1, -1])
    vol_m = np.array([{17: 17, 24: 24, 25: 23, 30: 28}[v] for v in vol])
    got_m = hs.geometry_order(xyz_m[perm], vol_m[perm])
    assert np.array_equal(perm[got_m], np.arange(len(elems)))


def test_geometry_same_radius_pair_uses_flight_direction():
    """Two pixel-barrel hits at the same radius 0.7 mm apart in z on a track going to -z:
    the one with larger z is crossed first (z decreases along the track)."""
    theta = 2 * np.arctan(np.exp(2.0))          # eta = -2
    d = np.array([np.sin(theta), 0.0, np.cos(theta)])
    xyz = np.array([32.0 / d[0] * d, 32.0 / d[0] * d + [0.0001, 0, -0.7], 68.0 / d[0] * d])
    vol = np.array([17, 17, 17])
    assert hs.z_direction(xyz) == -1.0
    assert np.array_equal(hs.geometry_order(xyz, vol), [0, 1, 2])
    # a z-direction of +1 would swap the pair: the direction is what decides
    prim, sec = hs.geometry_keys(xyz, vol, direction=1.0)
    assert np.array_equal(np.lexsort((sec, prim)), [1, 0, 2])


def test_geometry_rejects_unknown_volume():
    with pytest.raises(ValueError):
        hs.geometry_keys(np.zeros((2, 3)), np.array([17, 99]))
