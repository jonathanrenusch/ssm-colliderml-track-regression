"""Truth perigee parameters from the production vertex and momentum.

The ``particles`` tables of the ColliderML ``drift_beamspot`` samples carry no
usable perigee impact parameters, so the regression targets are derived here:
the helix is propagated from the production vertex back to its point of
closest approach to the beamline in the solenoid field (3 T).
"""

from __future__ import annotations

import numpy as np

# R[m] = pT[GeV] / (KAPPA * B[T] * |q|)
KAPPA = 0.299792458

# solenoid field along +z [T]
DEFAULT_BZ = 3.0


def truth_perigee(vx, vy, vz, px, py, pz, q, Bz: float = DEFAULT_BZ):
    """Perigee ``(d0, z0, phi, theta, qop)`` in ACTS conventions.

    Lengths in mm, angles in rad, ``qop`` in 1/GeV.  Inputs are per-particle
    arrays of equal length; neutral or zero-momentum particles come back as
    NaN and are expected to be cut by the caller's ``isfinite`` mask.
    """
    vx, vy, vz, px, py, pz, q = (
        np.asarray(a, dtype=np.float64) for a in (vx, vy, vz, px, py, pz, q)
    )
    pt = np.hypot(px, py)
    p = np.sqrt(px * px + py * py + pz * pz)

    with np.errstate(divide="ignore", invalid="ignore"):
        theta = np.arccos(np.clip(pz / np.where(p > 0, p, np.nan), -1.0, 1.0))
        qop = np.where(p > 0, q / p, np.nan)

        R = pt / (KAPPA * Bz * np.abs(np.where(q != 0, q, np.nan))) * 1000.0
        s = np.sign(q)
        ptc = np.where(pt > 0, pt, np.nan)

        # Centre of the transverse circle through the vertex.
        cx, cy = vx + R * s * py / ptc, vy - R * s * px / ptc
        rc = np.hypot(cx, cy)

        # Perigee = the point of that circle nearest the origin.
        f = (rc - R) / rc
        px_per, py_per = cx * f, cy * f

        # Signed turn from vertex to perigee.  The vertex sits a few mm from the
        # perigee, so the shortest branch is always the right one.
        a0 = np.arctan2(vy - cy, vx - cx)
        a1 = np.arctan2(py_per - cy, px_per - cx)
        dphi = np.remainder(a1 - a0 + np.pi, 2.0 * np.pi) - np.pi

        # The forward-arc sense in the circle angle is -sign(q) for Bz > 0, so
        # the z displacement carries that factor; the momentum direction rotates
        # rigidly with the circle angle, so phi does not.  Dropping the sign(q)
        # here inflates the z0 residual from ~30 um to ~5 mm.
        z0 = vz - s * (pz / ptc) * R * dphi
        phi = np.remainder(np.arctan2(py, px) + dphi + np.pi, 2.0 * np.pi) - np.pi
        d0 = -(px_per * np.sin(phi) - py_per * np.cos(phi))

    return d0, z0, phi, theta, qop
