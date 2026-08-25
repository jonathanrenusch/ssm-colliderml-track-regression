"""Synthetic-helix checks for track_regression.hit_sorting.

The generator below integrates the equations of motion for ``q v x B`` with
``B = (0, 0, Bz)`` directly (``dphi/dl_T = -q / R``), so it is independent of the
centre/turning-angle construction in :func:`helix_arc_length`.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from track_regression import hit_sorting as hs  # noqa: E402
from track_regression.perigee import truth_perigee  # noqa: E402


def helix_points(l, d0, z0, phi, theta, qop, Bz=2.0):
    """Points at 3-D path length ``l`` from the perigee, from the equations of motion."""
    q = np.sign(qop)
    pt = np.sin(theta) / abs(qop)
    R = pt / (hs.KAPPA * Bz) * 1000.0
    lt = l * np.sin(theta)
    px, py = -d0 * np.sin(phi), d0 * np.cos(phi)
    x = px + q * R * (np.sin(phi) - np.sin(phi - q * lt / R))
    y = py + q * R * (np.cos(phi - q * lt / R) - np.cos(phi))
    z = z0 + l * np.cos(theta)
    return np.stack([x, y, z], 1)


def test_generator_direction_at_perigee():
    d0, z0, phi, theta, qop = 3.0, -50.0, 0.7, 1.1, -0.5
    eps = 1e-4
    p = helix_points(np.array([0.0, eps]), d0, z0, phi, theta, qop)
    d = (p[1] - p[0]) / eps
    want = np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])
    assert np.allclose(d, want, atol=1e-6)
    assert np.allclose(p[0], hs.perigee_point(d0, z0, phi))


@pytest.mark.parametrize("qop", [0.5, -0.5, 0.05, -1.9])
@pytest.mark.parametrize("theta", [0.4, 1.3, 1.57, 2.6])
def test_arc_length_recovers_path_length(qop, theta):
    rng = np.random.default_rng(int(abs(qop) * 100 + theta * 10))
    d0, z0, phi = rng.uniform(-7, 7), rng.uniform(-260, 260), rng.uniform(-np.pi, np.pi)
    l = np.sort(rng.uniform(30, 1400, 15))
    pts = helix_points(l, d0, z0, phi, theta, qop)
    perm = rng.permutation(15)
    pts, l = pts[perm], l[perm]
    lt = hs.helix_arc_length(pts, d0, z0, phi, theta, qop, mode="transverse")
    assert np.allclose(lt, l, rtol=1e-9, atol=1e-6)
    if abs(np.cos(theta)) > 1e-3:
        ll = hs.helix_arc_length(pts, d0, z0, phi, theta, qop, mode="longitudinal")
        assert np.allclose(ll, l, rtol=1e-9, atol=1e-6)
    vol = np.where(np.arange(15) % 2 == 0, 17, 25)   # arbitrary barrel/endcap mix
    lm = hs.helix_arc_length(pts, d0, z0, phi, theta, qop, vol, mode="mixed")
    if abs(np.cos(theta)) > 1e-3:
        assert np.allclose(lm, l, rtol=1e-9, atol=1e-6)
    assert np.array_equal(hs.order(lt), np.argsort(l))


def test_per_hit_parameter_broadcast():
    """Two tracks concatenated with per-hit parameters give the same as one at a time."""
    a = (2.0, 10.0, 0.3, 1.0, 0.7)
    b = (-4.0, -100.0, -2.0, 2.2, -0.2)
    la, lb = np.linspace(40, 900, 8), np.linspace(50, 1200, 11)
    pa, pb = helix_points(la, *a), helix_points(lb, *b)
    xyz = np.concatenate([pa, pb])
    par = [np.concatenate([np.full(8, va), np.full(11, vb)]) for va, vb in zip(a, b)]
    got = hs.helix_arc_length(xyz, *par, mode="transverse")
    assert np.allclose(got, np.concatenate([la, lb]), rtol=1e-9, atol=1e-6)


def test_looper_unwrap():
    """pT = 0.2 GeV (R = 333 mm): hits over 3.5 turns come back in path-length order."""
    d0, z0, phi, theta, qop = 1.0, 20.0, 0.2, 1.2, 1.0 / (0.2 / np.sin(1.2))
    R = 0.2 / (hs.KAPPA * 2.0) * 1000.0
    l = np.linspace(50, 3.5 * 2 * np.pi * R / np.sin(theta), 40)
    pts = helix_points(l, d0, z0, phi, theta, qop)
    got = hs.helix_arc_length(pts, d0, z0, phi, theta, qop, mode="transverse")
    assert np.allclose(got, l, rtol=1e-9, atol=1e-5)


def test_centre_convention_matches_perigee_module():
    """The circle centre used here is the one truth_perigee() propagates around."""
    rng = np.random.default_rng(3)
    for _ in range(20):
        vx, vy, vz = rng.uniform(-5, 5, 2).tolist() + [rng.uniform(-200, 200)]
        px, py, pz = rng.uniform(-3, 3, 3)
        q = rng.choice([-1.0, 1.0])
        d0, z0, phi, theta, qop = (float(v) for v in truth_perigee(vx, vy, vz, px, py, pz, q))
        # the vertex lies on the helix through the perigee: signed turning angle from
        # the perigee to the vertex reproduces vz
        l = hs.helix_arc_length(np.array([[vx, vy, vz]]), d0, z0, phi, theta, qop,
                                mode="transverse", unwrap_min_pitch=np.inf)[0]
        R = np.sin(theta) / abs(qop) / (hs.KAPPA * 2.0) * 1000.0
        # vertex may sit just behind the perigee -> dalpha wraps to ~2pi; undo that
        lt = l * np.sin(theta)
        if lt > np.pi * R:
            lt -= 2 * np.pi * R
        z_pred = z0 + lt / np.sin(theta) * np.cos(theta)
        assert abs(z_pred - vz) < 1e-6, (z_pred, vz)
        assert abs(hs.distance_from_perigee(np.array([[vx, vy, vz]]), d0, z0, phi)[0]
                   - np.hypot(np.hypot(vx + d0 * np.sin(phi), vy - d0 * np.cos(phi)), vz - z0)) < 1e-9


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
    vol_m = np.array({17: 17, 24: 24, 25: 23, 30: 28}[v] for v in vol) if False else \
        np.array([{17: 17, 24: 24, 25: 23, 30: 28}[v] for v in vol])
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
