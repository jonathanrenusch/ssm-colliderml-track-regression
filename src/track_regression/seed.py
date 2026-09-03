"""Three-space-point helix seed, the ACTS way, from the hits of one track.

Port of ``Acts::estimateTrackParamsFromSeed`` (Core/src/Seeding/
EstimateTrackParamsFromSeed.cpp, the ATLAS ``SiTrackMaker_xk::getAtaPlane``
conformal-map estimate) plus the triplet choice of
``ActsExamples::TruthSeedingAlgorithm`` (pixel space points, largest radial
lever arm) and the transport of the result to the beamline perigee with
:func:`track_regression.perigee.truth_perigee`.

Inputs are only what every hit already carries -- ``x, y, z`` and
``volume_id`` -- plus the constant solenoid field.  No geometry tables, no
truth, no time: the same function runs at preprocessing, in the collate and at
inference.  Everything is closed-form and vectorised over tracks (numpy) with
an identical torch twin for in-model use.

Conventions (ACTS units: mm, GeV, e): ``q/pT = curvature / (KAPPA * Bz)``
with ``KAPPA = 0.299792458e-3 GeV / (e mm T)``; the output vector is
``[d0, z0, phi, theta, qop]`` at the perigee, the same as the training targets.
"""
from __future__ import annotations

import numpy as np

from track_regression.perigee import truth_perigee

KAPPA = 0.299792458e-3          # GeV / (e * mm * T)
DEFAULT_BZ = 3.0                # tesla.  MEASURED on the drift_beamspot hits 2026-08-27: the implied
                                # field from 3-pixel-point circles vs truth pT is 3.00 T in every pT bin
                                # (CLAUDE.md §4.8); the ODD default 2 T is NOT what ColliderML simulated.
                                # Only seed q/p depends on Bz (it cancels in d0/z0/phi/theta).
PIXEL_VOLUMES = (16, 17, 18)    # ODD pixel barrel + the two pixel endcaps
MIN_DELTA_R = 10.0              # mm, ACTS TruthSeedingAlgorithm deltaRMin
QOP_STRAIGHT = 1.0e-4           # e/GeV used when the triplet is (numerically) straight

PARAMS = ("d0", "z0", "phi", "theta", "qop")


# ---------------------------------------------------------------------------
# triplet selection
# ---------------------------------------------------------------------------

def select_triplet(xyz: np.ndarray, valid: np.ndarray, volume_id: np.ndarray,
                   min_delta_r: float = MIN_DELTA_R) -> np.ndarray:
    """Indices ``(N, 3)`` of the bottom / middle / top hit per track.

    ``xyz`` is ``(N, L, 3)`` padded, ``valid`` ``(N, L)`` bool, ``volume_id``
    ``(N, L)``.  Candidates are the pixel hits (fallback: all hits when fewer
    than three pixel hits exist).  Bottom = innermost candidate, top =
    outermost, middle = the candidate maximising ``(r_m - r_b) * (r_t - r_m)``
    -- the ACTS lever-arm score -- preferring middles at least ``min_delta_r``
    from both.  ACTS searches all ordered triplets; taking the radial extremes
    as bottom/top is the maximum-lever-arm member of that search.
    """
    r = np.hypot(xyz[..., 0], xyz[..., 1])
    pix = valid & np.isin(np.rint(volume_id).astype(np.int64), PIXEL_VOLUMES)
    use = np.where((pix.sum(1) >= 3)[:, None], pix, valid)
    n_idx = np.arange(xyz.shape[0])
    b = np.where(use, r, np.inf).argmin(1)
    t = np.where(use, r, -np.inf).argmax(1)
    rb, rt = r[n_idx, b][:, None], r[n_idx, t][:, None]
    score = (r - rb) * (rt - r)
    ok = use.copy()
    ok[n_idx, b] = False
    ok[n_idx, t] = False
    far = ok & (r - rb >= min_delta_r) & (rt - r >= min_delta_r)
    cand = np.where(far.any(1)[:, None], far, ok)
    m = np.where(cand, score, -np.inf).argmax(1)
    return np.stack([b, m, t], axis=1)


# ---------------------------------------------------------------------------
# ACTS conformal-map estimate at the bottom space point
# ---------------------------------------------------------------------------

def _sinc(x):
    return np.where(np.abs(x) < 1e-8, 1.0, np.sin(x) / np.where(np.abs(x) < 1e-8, 1.0, x))


def estimate_free(sp0: np.ndarray, sp1: np.ndarray, sp2: np.ndarray,
                  bz: float = DEFAULT_BZ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ACTS ``estimateTrackParamsFromSeed`` for ``(N, 3)`` triplets.

    Returns ``(direction (N,3) unit, qop (N,) e/GeV, straight (N,) bool)``;
    the position is ``sp0``.  ``straight`` flags triplets whose conformal
    line is degenerate (collinear points); their ``qop`` is set to
    ``QOP_STRAIGHT`` with positive sign and the direction to ``sp2 - sp0``.
    """
    sp0, sp1, sp2 = (np.asarray(a, np.float64) for a in (sp0, sp1, sp2))
    bvec = np.array([0.0, 0.0, 1.0])                        # field direction
    rel = sp1 - sp0
    zax = np.broadcast_to(bvec, rel.shape)
    yax = np.cross(zax, rel)
    yax = yax / np.maximum(np.linalg.norm(yax, axis=1, keepdims=True), 1e-300)
    xax = np.cross(yax, zax)
    # local = R^T (p - sp0), with R columns (xax, yax, zax)
    def to_local(p):
        d = p - sp0
        return np.stack([(d * xax).sum(1), (d * yax).sum(1), (d * zax).sum(1)], axis=1)
    l1, l2 = to_local(sp1), to_local(sp2)
    with np.errstate(divide="ignore", invalid="ignore"):
        uv1 = l1[:, :2] / (l1[:, :2] ** 2).sum(1, keepdims=True)
        uv2 = l2[:, :2] / (l2[:, :2] ** 2).sum(1, keepdims=True)
        duv = uv2 - uv1
        A = duv[:, 1] / duv[:, 0]
        Bc = uv1[:, 1] - A * uv1[:, 0]
        b_over_s = (uv1[:, 1] * uv2[:, 0] - uv2[:, 1] * uv1[:, 0]) / np.linalg.norm(duv, axis=1)
        # dz/ds with the sinc correction (Acts computeDzDs)
        def local_phi(l2d):
            rr = 2.0 * Bc[:, None] * l2d - np.stack([-A, np.ones_like(A)], axis=1)
            return np.arctan2(rr[:, 1], rr[:, 0])
        phi0 = local_phi(np.zeros_like(l1[:, :2]))
        phi2 = local_phi(l2[:, :2])
        dphi = phi2 - phi0
        dzds = _sinc(dphi / 2.0) * l2[:, 2] / np.linalg.norm(l2[:, :2], axis=1)
        # tangent at the bottom point (local (0,0)): r = (A, -1) -> t = (1, A, |r| dzds)
        rnorm = np.sqrt(1.0 + A ** 2)
        t_loc = np.stack([np.ones_like(A), A, rnorm * dzds], axis=1)
        t_loc = t_loc / np.linalg.norm(t_loc, axis=1, keepdims=True)
        direction = t_loc[:, :1] * xax + t_loc[:, 1:2] * yax + t_loc[:, 2:3] * zax
        q_over_pt = 2.0 * b_over_s / (KAPPA * bz)
        qop = q_over_pt / np.hypot(1.0, dzds)
    # Two ways a triplet can be "straight": exactly collinear points map to a
    # line through the (u, v) origin and give curvature 0 (finite everything,
    # qop == 0); a degenerate map (duv.x -> 0 or non-finite values) gives no
    # usable tangent.  Both get a tiny positive curvature so the perigee
    # transport stays finite; the degenerate case also gets the chord direction.
    bad = ~(np.isfinite(qop) & np.isfinite(direction).all(1)) | (np.abs(duv[:, 0]) < 1e-14)
    if bad.any():
        d = sp2[bad] - sp0[bad]
        direction[bad] = d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-300)
        qop[bad] = QOP_STRAIGHT
    tiny = np.abs(qop) < 1e-9
    qop = np.where(tiny, QOP_STRAIGHT, qop)
    return direction, qop, bad | tiny


def perigee_from_free(sp0: np.ndarray, direction: np.ndarray, qop: np.ndarray,
                      bz: float = DEFAULT_BZ) -> np.ndarray:
    """Transport (point, direction, q/p) to the beamline perigee -> ``(N, 5)``."""
    p_abs = 1.0 / np.maximum(np.abs(qop), 1e-12)
    p = direction * p_abs[:, None]
    q = np.where(qop >= 0, 1.0, -1.0)
    d0, z0, phi, theta, qop_out = truth_perigee(sp0[:, 0], sp0[:, 1], sp0[:, 2],
                                                p[:, 0], p[:, 1], p[:, 2], q, Bz=bz)
    return np.stack([d0, z0, phi, theta, qop_out], axis=1)


def seed_perigee(xyz: np.ndarray, valid: np.ndarray, volume_id: np.ndarray,
                 bz: float = DEFAULT_BZ) -> np.ndarray:
    """Seed perigee parameters ``(N, 5)`` = [d0, z0, phi, theta, qop] for a
    padded batch of tracks (``xyz (N,L,3)``, ``valid (N,L)``, ``volume_id (N,L)``)."""
    idx = select_triplet(xyz, valid, volume_id)
    n = np.arange(xyz.shape[0])
    sp0, sp1, sp2 = (xyz[n, idx[:, k]] for k in range(3))
    direction, qop, _ = estimate_free(sp0, sp1, sp2, bz)
    return perigee_from_free(sp0, direction, qop, bz)


def seed_from_csr(hits: np.ndarray, lens: np.ndarray, bz: float = DEFAULT_BZ) -> np.ndarray:
    """Convenience for the flat-store CSR layout: ``hits (total_L, >=8)`` with
    columns x, y, z at 0..2 and volume_id at 7; ``lens (N,)``."""
    lens64 = np.asarray(lens, np.int64)
    n, max_len = len(lens64), int(lens64.max())
    csum = np.cumsum(lens64)
    pos = np.arange(len(hits), dtype=np.int64) - np.repeat(csum - lens64, lens64)
    row = np.repeat(np.arange(n), lens64)
    xyz = np.zeros((n, max_len, 3), np.float64)
    vol = np.zeros((n, max_len), np.float64)
    valid = np.zeros((n, max_len), bool)
    xyz[row, pos] = hits[:, :3]
    vol[row, pos] = hits[:, 7]
    valid[row, pos] = True
    return seed_perigee(xyz, valid, vol, bz)


# ---------------------------------------------------------------------------
# torch twin (same math, for in-model use)
# ---------------------------------------------------------------------------

def estimate_free_torch(sp0, sp1, sp2, bz: float = DEFAULT_BZ):
    """torch version of :func:`estimate_free` (no straight-line fallback
    branching: degenerate triplets give non-finite values the caller must mask)."""
    import torch
    dt = torch.float64 if sp0.dtype == torch.float64 else sp0.dtype
    rel = sp1 - sp0
    zax = torch.zeros_like(rel); zax[..., 2] = 1.0
    yax = torch.cross(zax, rel, dim=-1)
    yax = yax / yax.norm(dim=-1, keepdim=True).clamp_min(1e-300 if dt == torch.float64 else 1e-30)
    xax = torch.cross(yax, zax, dim=-1)
    def to_local(p):
        d = p - sp0
        return torch.stack([(d * xax).sum(-1), (d * yax).sum(-1), (d * zax).sum(-1)], dim=-1)
    l1, l2 = to_local(sp1), to_local(sp2)
    uv1 = l1[:, :2] / (l1[:, :2] ** 2).sum(-1, keepdim=True)
    uv2 = l2[:, :2] / (l2[:, :2] ** 2).sum(-1, keepdim=True)
    duv = uv2 - uv1
    A = duv[:, 1] / duv[:, 0]
    Bc = uv1[:, 1] - A * uv1[:, 0]
    b_over_s = (uv1[:, 1] * uv2[:, 0] - uv2[:, 1] * uv1[:, 0]) / duv.norm(dim=-1)
    def local_phi(l2d):
        rr = 2.0 * Bc[:, None] * l2d - torch.stack([-A, torch.ones_like(A)], dim=-1)
        return torch.atan2(rr[:, 1], rr[:, 0])
    dphi = local_phi(l2[:, :2]) - local_phi(torch.zeros_like(l1[:, :2]))
    x = dphi / 2.0
    sinc = torch.where(x.abs() < 1e-8, torch.ones_like(x), torch.sin(x) / torch.where(x.abs() < 1e-8, torch.ones_like(x), x))
    dzds = sinc * l2[:, 2] / l2[:, :2].norm(dim=-1)
    rnorm = torch.sqrt(1.0 + A ** 2)
    t_loc = torch.stack([torch.ones_like(A), A, rnorm * dzds], dim=-1)
    t_loc = t_loc / t_loc.norm(dim=-1, keepdim=True)
    direction = t_loc[:, :1] * xax + t_loc[:, 1:2] * yax + t_loc[:, 2:3] * zax
    qop = (2.0 * b_over_s / (KAPPA * bz)) / torch.hypot(torch.ones_like(dzds), dzds)
    return direction, qop


# ---------------------------------------------------------------------------
# per-hit residuals to the seed helix (the KF's representation, CLAUDE.md §4.8)
# ---------------------------------------------------------------------------

RESIDUAL_SCALE_MM = 0.1     # asinh knee: linear below 0.1 mm, logarithmic above


def seed_residuals(xyz: np.ndarray, seed: np.ndarray, track: np.ndarray,
                   bz: float = DEFAULT_BZ) -> np.ndarray:
    """Residuals of every hit to its track's seed helix, ``(n_hits, 3)``.

    ``xyz`` ``(n_hits, 3)`` [mm], ``seed`` ``(n_tracks, 5)`` = [d0, z0, phi,
    theta, qop] at the perigee, ``track`` ``(n_hits,)`` hit -> track index.
    Columns: ``du`` = signed transverse distance from the helix circle
    (along U = Z x T, mm), ``dv`` = residual along V = T x U (= sin(theta) *
    dz at the same azimuth, mm), ``s_helix`` = 3-D path length from the
    perigee to the hit's azimuth on the helix (mm).  Helix conventions as in
    :mod:`track_regression.perigee` / ``tests/test_seed.py::helix_points``:
    P0 = (-d0 sin(phi), d0 cos(phi), z0), centre C = P0 + q R (sin(phi),
    -cos(phi)), forward turning angle a >= 0, z(a) = z0 + R a cot(theta).
    Hits of the seed triplet come out at exactly zero (bottom / top) --
    the seed is self-identifying to the encoder.
    """
    xyz = np.asarray(xyz, np.float64)
    d0, z0, phi, theta, qop = (np.asarray(seed[:, i], np.float64)[track] for i in range(5))
    q = np.where(qop >= 0, 1.0, -1.0)
    sin_t = np.clip(np.sin(theta), 1e-6, None)
    pt = sin_t / np.maximum(np.abs(qop), 1e-12)
    R = pt / (KAPPA * bz)                                         # mm
    cx = -d0 * np.sin(phi) + q * R * np.sin(phi)
    cy = d0 * np.cos(phi) - q * R * np.cos(phi)
    rho = np.hypot(xyz[:, 0] - cx, xyz[:, 1] - cy)
    beta = np.arctan2(xyz[:, 1] - cy, xyz[:, 0] - cx)
    a = np.mod(q * (phi - beta) + 0.5 * np.pi, 2.0 * np.pi)        # forward turning angle to the hit's azimuth
    t = R * a                                                     # transverse path length
    dz = xyz[:, 2] - (z0 + t * np.cos(theta) / sin_t)
    du = q * (rho - R)
    dv = sin_t * dz
    s_helix = t / sin_t
    return np.stack([du, dv, s_helix], axis=1)


def compress_residuals(res: np.ndarray) -> np.ndarray:
    """asinh-compress du, dv (mm -> dimensionless, +-~7.5); keep s_helix in mm."""
    out = np.empty_like(res)
    out[:, 0] = np.arcsinh(res[:, 0] / RESIDUAL_SCALE_MM)
    out[:, 1] = np.arcsinh(res[:, 1] / RESIDUAL_SCALE_MM)
    out[:, 2] = res[:, 2]
    return out


# ---------------------------------------------------------------------------
# torch (GPU) path for inference: triplet -> free estimate -> perigee -> residuals
# ---------------------------------------------------------------------------

def select_triplet_torch(xyz, valid, volume_id, min_delta_r: float = MIN_DELTA_R):
    """torch twin of :func:`select_triplet` (``xyz (N,L,3)``, ``valid (N,L)`` bool,
    ``volume_id (N,L)``) -> ``(N, 3)`` long indices [bottom, middle, top]."""
    import torch
    r = torch.hypot(xyz[..., 0], xyz[..., 1])
    vol = torch.round(volume_id).long()
    pix = valid & ((vol == 16) | (vol == 17) | (vol == 18))
    use = torch.where((pix.sum(1) >= 3)[:, None], pix, valid)
    inf = torch.tensor(float("inf"), dtype=r.dtype, device=r.device)
    b = torch.where(use, r, inf).argmin(1)
    t = torch.where(use, r, -inf).argmax(1)
    n = torch.arange(xyz.shape[0], device=xyz.device)
    rb, rt = r[n, b][:, None], r[n, t][:, None]
    score = (r - rb) * (rt - r)
    ok = use.clone(); ok[n, b] = False; ok[n, t] = False
    far = ok & (r - rb >= min_delta_r) & (rt - r >= min_delta_r)
    cand = torch.where(far.any(1)[:, None], far, ok)
    m = torch.where(cand, score, -inf).argmax(1)
    return torch.stack([b, m, t], dim=1)


def perigee_from_free_torch(sp0, direction, qop, bz: float = DEFAULT_BZ):
    """torch twin of :func:`perigee_from_free` (= perigee.truth_perigee on the seed point)."""
    import torch
    p_abs = 1.0 / qop.abs().clamp_min(1e-12)
    px, py, pz = (direction[:, i] * p_abs for i in range(3))
    q = torch.where(qop >= 0, torch.ones_like(qop), -torch.ones_like(qop))
    vx, vy, vz = sp0[:, 0], sp0[:, 1], sp0[:, 2]
    pt = torch.hypot(px, py); p = torch.sqrt(px * px + py * py + pz * pz)
    theta = torch.arccos((pz / p).clamp(-1.0, 1.0))
    qop_out = q / p
    R = pt / (KAPPA * bz)
    cx, cy = vx + R * q * py / pt, vy - R * q * px / pt
    rc = torch.hypot(cx, cy)
    f = (rc - R) / rc
    pxp, pyp = cx * f, cy * f
    a0 = torch.atan2(vy - cy, vx - cx); a1 = torch.atan2(pyp - cy, pxp - cx)
    two_pi = 2.0 * torch.pi
    dphi = torch.remainder(a1 - a0 + torch.pi, two_pi) - torch.pi
    z0 = vz - q * (pz / pt) * R * dphi
    phi = torch.remainder(torch.atan2(py, px) + dphi + torch.pi, two_pi) - torch.pi
    d0 = -(pxp * torch.sin(phi) - pyp * torch.cos(phi))
    return torch.stack([d0, z0, phi, theta, qop_out], dim=1)


def seed_perigee_torch(xyz, valid, volume_id, bz: float = DEFAULT_BZ):
    """GPU seed for a padded batch: ``(N,5)`` = [d0, z0, phi, theta, qop].  Degenerate
    (straight) triplets get ``QOP_STRAIGHT`` and the chord direction, as in numpy."""
    import torch
    idx = select_triplet_torch(xyz, valid, volume_id)
    n = torch.arange(xyz.shape[0], device=xyz.device)
    sp0, sp1, sp2 = (xyz[n, idx[:, k]] for k in range(3))
    direction, qop = estimate_free_torch(sp0, sp1, sp2, bz)
    bad = ~(torch.isfinite(qop) & torch.isfinite(direction).all(1)) | (qop.abs() < 1e-9)
    if bad.any():
        d = sp2 - sp0
        direction = torch.where(bad[:, None], d / d.norm(dim=1, keepdim=True).clamp_min(1e-30), direction)
        qop = torch.where(bad, torch.full_like(qop, QOP_STRAIGHT), qop)
    return perigee_from_free_torch(sp0, direction, qop, bz)


def seed_residuals_torch(xyz, seed, track, bz: float = DEFAULT_BZ):
    """torch twin of :func:`seed_residuals` (``xyz (n_hits,3)``, ``seed (N,5)``, ``track (n_hits,)``)."""
    import torch
    d0, z0, phi, theta, qop = (seed[:, i][track] for i in range(5))
    q = torch.where(qop >= 0, torch.ones_like(qop), -torch.ones_like(qop))
    sin_t = torch.sin(theta).clamp_min(1e-6)
    pt = sin_t / qop.abs().clamp_min(1e-12)
    R = pt / (KAPPA * bz)
    cx = -d0 * torch.sin(phi) + q * R * torch.sin(phi)
    cy = d0 * torch.cos(phi) - q * R * torch.cos(phi)
    rho = torch.hypot(xyz[:, 0] - cx, xyz[:, 1] - cy)
    beta = torch.atan2(xyz[:, 1] - cy, xyz[:, 0] - cx)
    a = torch.remainder(q * (phi - beta) + 0.5 * torch.pi, 2.0 * torch.pi)
    t = R * a
    dz = xyz[:, 2] - (z0 + t * torch.cos(theta) / sin_t)
    res = torch.stack([q * (rho - R), sin_t * dz, t / sin_t], dim=1)
    out = res.clone()
    out[:, 0] = torch.asinh(res[:, 0] / RESIDUAL_SCALE_MM)
    out[:, 1] = torch.asinh(res[:, 1] / RESIDUAL_SCALE_MM)
    return out
