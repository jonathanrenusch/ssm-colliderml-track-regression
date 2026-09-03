"""Synthetic-helix tests for track_regression.seed (the ACTS three-point seed)."""
import numpy as np
import pytest

from track_regression.seed import (DEFAULT_BZ, KAPPA, PIXEL_VOLUMES, compress_residuals, estimate_free,
                                   estimate_free_torch, perigee_from_free, seed_from_csr,
                                   seed_perigee, seed_perigee_torch, seed_residuals,
                                   seed_residuals_torch, select_triplet)

RNG = np.random.default_rng(7)


def helix_points(d0, z0, phi, theta, qop, alphas, bz=DEFAULT_BZ):
    """Points on the exact helix through the perigee (d0, z0, phi, theta, qop)
    at forward turning angles ``alphas`` (same conventions as perigee.py)."""
    q = np.sign(qop)
    pt = np.sin(theta) / abs(qop)
    R = pt / (KAPPA * bz)
    P = np.array([-d0 * np.sin(phi), d0 * np.cos(phi), z0])
    C = P[:2] + q * R * np.array([np.sin(phi), -np.cos(phi)])
    pts = []
    for a in alphas:
        psi = phi - q * a
        xy = C + q * R * np.array([-np.sin(psi), np.cos(psi)])
        z = z0 + R * a / np.tan(theta)
        pts.append([xy[0], xy[1], z])
    return np.array(pts)


def random_params(n):
    d0 = RNG.uniform(-7, 7, n)
    z0 = RNG.uniform(-250, 250, n)
    phi = RNG.uniform(-np.pi, np.pi, n)
    eta = RNG.uniform(-2.5, 2.5, n)
    theta = 2 * np.arctan(np.exp(-eta))
    pt = np.exp(RNG.uniform(np.log(1.0), np.log(110.0), n))
    q = RNG.choice([-1.0, 1.0], n)
    qop = q / (pt / np.sin(theta))
    return d0, z0, phi, theta, qop


def test_seed_reproduces_exact_helix():
    n = 500
    d0, z0, phi, theta, qop = random_params(n)
    sp = np.zeros((3, n, 3))
    for i in range(n):
        pt = np.sin(theta[i]) / abs(qop[i]); R = pt / (KAPPA * DEFAULT_BZ)
        # turning angles that put the three points at ~35, 100, 170 mm of transverse path
        alphas = np.array([35.0, 100.0, 170.0]) / R
        sp[:, i] = helix_points(d0[i], z0[i], phi[i], theta[i], qop[i], alphas)
    direction, qop_est, straight = estimate_free(sp[0], sp[1], sp[2])
    assert not straight.any()
    out = perigee_from_free(sp[0], direction, qop_est)
    dphi = np.angle(np.exp(1j * (out[:, 2] - phi)))
    assert np.allclose(out[:, 0], d0, atol=1e-6), np.abs(out[:, 0] - d0).max()
    assert np.allclose(out[:, 1], z0, atol=1e-5), np.abs(out[:, 1] - z0).max()
    assert np.allclose(dphi, 0, atol=1e-8)
    assert np.allclose(out[:, 3], theta, atol=1e-8)
    assert np.allclose(out[:, 4], qop, rtol=1e-6)


def test_torch_twin_matches_numpy():
    torch = pytest.importorskip("torch")
    n = 200
    d0, z0, phi, theta, qop = random_params(n)
    sp = np.zeros((3, n, 3))
    for i in range(n):
        pt = np.sin(theta[i]) / abs(qop[i]); R = pt / (KAPPA * DEFAULT_BZ)
        sp[:, i] = helix_points(d0[i], z0[i], phi[i], theta[i], qop[i], np.array([40.0, 90.0, 160.0]) / R)
    d_np, q_np, _ = estimate_free(sp[0], sp[1], sp[2])
    d_t, q_t = estimate_free_torch(*(torch.from_numpy(s) for s in sp))
    assert np.allclose(d_t.numpy(), d_np, atol=1e-10)
    assert np.allclose(q_t.numpy(), q_np, rtol=1e-10)


def test_triplet_selection_prefers_pixels_with_max_lever_arm():
    # one track, 10 hits: pixels at r=34,70,116,172 (vol 16) + strips beyond (vol 23/28)
    r = np.array([34, 70, 116, 172, 260, 360, 500, 660, 820, 1020], float)
    vol = np.array([16, 16, 16, 16, 23, 23, 23, 28, 28, 28], float)
    xyz = np.stack([r, np.zeros_like(r), 0.5 * r], axis=1)[None]
    valid = np.ones((1, 10), bool)
    b, m, t = select_triplet(xyz, valid, vol[None])[0]
    assert (b, t) == (0, 3)                     # innermost / outermost pixel
    assert m in (1, 2)                          # middle maximises (r_m-34)*(172-r_m): 70 -> 3672, 116 -> 4592
    assert m == 2
    # fewer than 3 pixel hits -> fall back to all hits
    vol2 = vol.copy(); vol2[:2] = 23
    b, m, t = select_triplet(xyz, valid, vol2[None])[0]
    assert (b, t) == (0, 9)


def test_seed_from_csr_and_padding_agree():
    n = 50
    d0, z0, phi, theta, qop = random_params(n)
    lens = RNG.integers(6, 13, n)
    hits, keep = [], []
    for i in range(n):
        pt = np.sin(theta[i]) / abs(qop[i]); R = pt / (KAPPA * DEFAULT_BZ)
        alphas = np.sort(RNG.uniform(30.0, 900.0, lens[i])) / R
        pts = helix_points(d0[i], z0[i], phi[i], theta[i], qop[i], alphas)
        H = np.zeros((lens[i], 12)); H[:, :3] = pts
        rr = np.hypot(pts[:, 0], pts[:, 1])
        H[:, 7] = np.where(rr < 200, 16, 23)     # pixels inside 200 mm
        hits.append(H)
    seeds = seed_from_csr(np.concatenate(hits), lens)
    assert seeds.shape == (n, 5)
    assert np.allclose(seeds[:, 0], d0, atol=1e-5)
    assert np.allclose(seeds[:, 1], z0, atol=1e-4)
    assert np.allclose(seeds[:, 4], qop, rtol=1e-5)


def test_straight_line_fallback_is_finite():
    sp0 = np.array([[30.0, 0.0, 10.0]]); sp1 = np.array([[100.0, 0.0, 33.0]]); sp2 = np.array([[170.0, 0.0, 56.0]])
    direction, qop, straight = estimate_free(sp0, sp1, sp2)
    assert straight.all() and np.isfinite(qop).all() and np.isfinite(direction).all()
    out = perigee_from_free(sp0, direction, qop)
    assert np.isfinite(out).all()


def test_seed_residuals_vanish_on_exact_helix():
    n = 200
    d0, z0, phi, theta, qop = random_params(n)
    xyz, track = [], []
    for i in range(n):
        pt = np.sin(theta[i]) / abs(qop[i]); R = pt / (KAPPA * DEFAULT_BZ)
        alphas = np.sort(RNG.uniform(30.0, 1000.0, 10)) / R
        xyz.append(helix_points(d0[i], z0[i], phi[i], theta[i], qop[i], alphas)); track += [i] * 10
    xyz = np.concatenate(xyz); track = np.asarray(track)
    truth = np.stack([d0, z0, phi, theta, qop], 1)
    res = seed_residuals(xyz, truth, track)
    assert res.shape == (10 * n, 3)
    assert np.abs(res[:, 0]).max() < 1e-6, np.abs(res[:, 0]).max()      # du [mm]
    assert np.abs(res[:, 1]).max() < 1e-5, np.abs(res[:, 1]).max()      # dv [mm]
    # s_helix = 3-D path length from the perigee = R * alpha / sin(theta)
    assert (res[:, 2] > 0).all()
    for i in range(n):
        assert np.all(np.diff(res[track == i, 2]) > 0)


def test_seed_residuals_zero_on_seed_hits_and_compression():
    n = 50
    d0, z0, phi, theta, qop = random_params(n)
    lens = np.full(n, 12)
    hits = []
    for i in range(n):
        pt = np.sin(theta[i]) / abs(qop[i]); R = pt / (KAPPA * DEFAULT_BZ)
        alphas = np.sort(RNG.uniform(30.0, 900.0, 12)) / R
        pts = helix_points(d0[i], z0[i], phi[i], theta[i], qop[i], alphas)
        pts[3:] += RNG.normal(0, 0.5, pts[3:].shape)          # smear the non-seed hits
        H = np.zeros((12, 12)); H[:, :3] = pts
        rr = np.hypot(pts[:, 0], pts[:, 1]); H[:, 7] = np.where(rr < 200, 16, 23)
        hits.append(H)
    H = np.concatenate(hits)
    seeds = seed_from_csr(H, lens)
    track = np.repeat(np.arange(n), lens)
    res = seed_residuals(H[:, :3], seeds, track)
    c = compress_residuals(res)
    assert c.shape == res.shape and np.isfinite(c).all()
    # the residuals to the seed's own three points: bottom and top are exactly on the circle
    for i in range(n):
        r_i = res[track == i]
        assert min(np.abs(r_i[:, 0]).min(), 1) < 1e-6
    assert np.abs(c[:, :2]).max() < 10.0                        # asinh range


def test_torch_seed_and_residuals_match_numpy():
    torch = pytest.importorskip("torch")
    n = 300
    d0, z0, phi, theta, qop = random_params(n)
    L = 12; xyz = np.zeros((n, L, 3)); vol = np.zeros((n, L)); valid = np.ones((n, L), bool)
    for i in range(n):
        pt = np.sin(theta[i]) / abs(qop[i]); R = pt / (KAPPA * DEFAULT_BZ)
        alphas = np.sort(RNG.uniform(30.0, 900.0, L)) / R
        pts = helix_points(d0[i], z0[i], phi[i], theta[i], qop[i], alphas)
        pts[3:] += RNG.normal(0, 0.3, pts[3:].shape)
        xyz[i] = pts; vol[i] = np.where(np.hypot(pts[:, 0], pts[:, 1]) < 200, 16, 23)
    s_np = seed_perigee(xyz, valid, vol)
    s_t = seed_perigee_torch(torch.from_numpy(xyz), torch.from_numpy(valid), torch.from_numpy(vol)).numpy()
    assert np.allclose(s_t[:, [0, 1, 3, 4]], s_np[:, [0, 1, 3, 4]], rtol=1e-9, atol=1e-9)
    assert np.allclose(np.angle(np.exp(1j * (s_t[:, 2] - s_np[:, 2]))), 0, atol=1e-9)
    track = np.repeat(np.arange(n), L); flat = xyz.reshape(-1, 3)
    r_np = compress_residuals(seed_residuals(flat, s_np, track))
    r_t = seed_residuals_torch(torch.from_numpy(flat), torch.from_numpy(s_np), torch.from_numpy(track)).numpy()
    # float64 rounding only (rho - R cancellation at R ~ 1e5 mm for the highest-pT synthetic tracks)
    assert np.allclose(r_t, r_np, rtol=1e-7, atol=1e-6)
