"""GPU twin of :mod:`track_regression.seed` for inference: seed + per-hit residual
features computed in torch from the packed hit features, so nothing runs on the CPU.

    seed, res = gpu_seed_features(hit_features, cu_seqlens)   # (B, 5), (n_hits, 3)

``hit_features`` is the packed ``(n_hits, 12)`` tensor the collate emits (x, y, z
in columns 0-2, volume_id in column 7); ``cu_seqlens`` the ``(B+1,)`` int32
offsets.  The maths is float64 (as in numpy) and the results are cast to the
input dtype.  Parity with the numpy path is tested in tests/test_seed_torch.py.
"""
from __future__ import annotations

import math
import torch

from track_regression.seed import (DEFAULT_BZ, KAPPA, MIN_DELTA_R, PIXEL_VOLUMES, QOP_STRAIGHT,
                                   RESIDUAL_SCALE_MM, estimate_free_torch)


def _pad(hit_features: torch.Tensor, cu: torch.Tensor, max_len: int | None = None):
    """Packed (n_hits, F) -> padded xyz (B, L, 3), volume (B, L), valid (B, L), plus row/pos.

    ``max_len``: static pad length (e.g. 20) — skips the ``lens.max().item()``
    host sync so the whole seed path is CUDA-graph capturable."""
    B = cu.numel() - 1
    lens = (cu[1:] - cu[:-1]).long()
    L = max_len if max_len is not None else (int(lens.max().item()) if B > 0 else 0)
    # bucketize, not repeat_interleave(lens): tensor-repeats have a data-dependent
    # output size (host sync) — bucketize keeps the shape static (n_hits,), so the
    # path stays CUDA-graph capturable. Identical mapping.
    tok = torch.arange(hit_features.shape[0], device=cu.device)
    row = torch.bucketize(tok, cu[1:].long(), right=True)
    pos = tok - cu[:-1].long()[row]
    xyz = torch.zeros(B, L, 3, dtype=torch.float64, device=cu.device)
    vol = torch.zeros(B, L, dtype=torch.float64, device=cu.device)
    valid = torch.zeros(B, L, dtype=torch.bool, device=cu.device)
    xyz[row, pos] = hit_features[:, :3].double(); vol[row, pos] = hit_features[:, 7].double()
    # device-tensor RHS, not the Python scalar True: a scalar put is a
    # synchronizing H2D memcpy (breaks CUDA-graph capture)
    valid[row, pos] = torch.ones_like(row, dtype=torch.bool)
    return xyz, vol, valid, row, pos


_PIXVOL_CACHE: dict = {}


def _pixvol(device):
    """Device-cached PIXEL_VOLUMES tensor (also keeps the seed path CUDA-graph
    capturable: creating it per call is a synchronizing H2D copy)."""
    t = _PIXVOL_CACHE.get(device)
    if t is None:
        t = torch.tensor(PIXEL_VOLUMES, device=device)
        _PIXVOL_CACHE[device] = t
    return t


def select_triplet_torch(xyz, valid, vol, min_delta_r: float = MIN_DELTA_R):
    """torch port of :func:`seed.select_triplet` -> (B, 3) indices [bottom, middle, top]."""
    r = torch.hypot(xyz[..., 0], xyz[..., 1])
    pix = valid & torch.isin(torch.round(vol).long(), _pixvol(vol.device))
    use = torch.where((pix.sum(1) >= 3)[:, None], pix, valid)
    # scalar overloads throughout (where/scatter_ with Python scalars are kernel
    # constants): a torch.tensor(inf) or a scalar index-put is a synchronizing
    # H2D memcpy and breaks CUDA-graph capture
    b = torch.where(use, r, float("inf")).argmin(1)
    t = torch.where(use, r, float("-inf")).argmax(1)
    n = torch.arange(xyz.shape[0], device=xyz.device)
    rb, rt = r[n, b][:, None], r[n, t][:, None]
    score = (r - rb) * (rt - r)
    ok = use.clone()
    ok.scatter_(1, b[:, None], False)
    ok.scatter_(1, t[:, None], False)
    far = ok & (r - rb >= min_delta_r) & (rt - r >= min_delta_r)
    cand = torch.where(far.any(1)[:, None], far, ok)
    m = torch.where(cand, score, float("-inf")).argmax(1)
    return torch.stack([b, m, t], 1)


def perigee_from_free_torch(sp0, direction, qop, bz: float = DEFAULT_BZ):
    """torch port of :func:`perigee.truth_perigee` applied to (point, direction, q/p)."""
    p_abs = 1.0 / qop.abs().clamp_min(1e-12)
    p = direction * p_abs[:, None]
    q = torch.where(qop >= 0, 1.0, -1.0).to(sp0.dtype)
    vx, vy, vz = sp0[:, 0], sp0[:, 1], sp0[:, 2]
    px, py, pz = p[:, 0], p[:, 1], p[:, 2]
    pt = torch.hypot(px, py); pn = torch.sqrt(px * px + py * py + pz * pz)
    theta = torch.acos((pz / pn).clamp(-1.0, 1.0))
    qop_out = q / pn
    R = pt / (KAPPA * bz)                                          # mm (KAPPA is GeV/(e mm T))
    s = q
    cx, cy = vx + R * s * py / pt, vy - R * s * px / pt
    rc = torch.hypot(cx, cy)
    f = (rc - R) / rc
    px_per, py_per = cx * f, cy * f
    a0 = torch.atan2(vy - cy, vx - cx); a1 = torch.atan2(py_per - cy, px_per - cx)
    dphi = torch.remainder(a1 - a0 + math.pi, 2.0 * math.pi) - math.pi
    z0 = vz - s * (pz / pt) * R * dphi
    phi = torch.remainder(torch.atan2(py, px) + dphi + math.pi, 2.0 * math.pi) - math.pi
    d0 = -(px_per * torch.sin(phi) - py_per * torch.cos(phi))
    return torch.stack([d0, z0, phi, theta, qop_out], 1)


def seed_perigee_torch(xyz, valid, vol, bz: float = DEFAULT_BZ):
    """(B, 5) seed perigee parameters from padded float64 xyz/valid/volume, incl. the
    straight-line fallback of the numpy version."""
    idx = select_triplet_torch(xyz, valid, vol)
    n = torch.arange(xyz.shape[0], device=xyz.device)
    sp0, sp1, sp2 = (xyz[n, idx[:, k]] for k in range(3))
    direction, qop = estimate_free_torch(sp0, sp1, sp2, bz)
    bad = ~(torch.isfinite(qop) & torch.isfinite(direction).all(1))
    # degenerate map (as in numpy: |duv.x| < 1e-14 is folded into non-finite/tiny here)
    d = sp2 - sp0
    chord = d / d.norm(dim=1, keepdim=True).clamp_min(1e-300)
    direction = torch.where(bad[:, None], chord, direction)
    qop = torch.where(bad, torch.full_like(qop, QOP_STRAIGHT), qop)
    tiny = qop.abs() < 1e-9
    qop = torch.where(tiny, torch.full_like(qop, QOP_STRAIGHT), qop)
    return perigee_from_free_torch(sp0, direction, qop, bz)


def seed_residuals_torch(xyz_hits, seed, track, bz: float = DEFAULT_BZ):
    """torch port of :func:`seed.seed_residuals`: (n_hits, 3) = du, dv, s_helix [mm]."""
    d0, z0, phi, theta, qop = (seed[:, i][track] for i in range(5))
    q = torch.where(qop >= 0, 1.0, -1.0).to(seed.dtype)
    sin_t = torch.sin(theta).clamp_min(1e-6)
    pt = sin_t / qop.abs().clamp_min(1e-12)
    R = pt / (KAPPA * bz)
    cx = -d0 * torch.sin(phi) + q * R * torch.sin(phi)
    cy = d0 * torch.cos(phi) - q * R * torch.cos(phi)
    rho = torch.hypot(xyz_hits[:, 0] - cx, xyz_hits[:, 1] - cy)
    beta = torch.atan2(xyz_hits[:, 1] - cy, xyz_hits[:, 0] - cx)
    a = torch.remainder(q * (phi - beta) + 0.5 * math.pi, 2.0 * math.pi)
    t = R * a
    dz = xyz_hits[:, 2] - (z0 + t * torch.cos(theta) / sin_t)
    return torch.stack([q * (rho - R), sin_t * dz, t / sin_t], 1)


def compress_residuals_torch(res):
    out = res.clone()
    out[:, 0] = torch.asinh(res[:, 0] / RESIDUAL_SCALE_MM)
    out[:, 1] = torch.asinh(res[:, 1] / RESIDUAL_SCALE_MM)
    return out


@torch.no_grad()
def gpu_seed_features(hit_features: torch.Tensor, cu_seqlens: torch.Tensor, bz: float = DEFAULT_BZ,
                      max_len: int | None = None):
    """Seed (B, 5) and compressed residual features (n_hits, 3) on the device of the inputs.

    Pass ``max_len`` (the store's max hits per track, 20) to make the path
    CUDA-graph capturable (no host sync)."""
    xyz, vol, valid, row, pos = _pad(hit_features, cu_seqlens, max_len)
    seed = seed_perigee_torch(xyz, valid, vol, bz)
    res = compress_residuals_torch(seed_residuals_torch(hit_features[:, :3].double(), seed, row, bz))
    return seed.to(hit_features.dtype), res.to(hit_features.dtype)
