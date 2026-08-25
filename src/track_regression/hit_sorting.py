"""Time-independent hit orderings for the encoder input sequence.

Why this module exists
----------------------
ACTS orders the hits of a truth track by simulated hit *time*: both
``TruthTrackFinder`` and ``TruthSeedingAlgorithm`` ``std::sort`` the proto-track
on ``SimHit::time()`` (Geant4 global time of the sensor crossing).  In the
``drift_beamspot`` campaign that key is not usable from the reconstructed table:
``tracker_hits.time`` is 0 for every strip hit and smeared by ~0.1 ns for pixels,
and the truth ``tracker_simhits.true_time`` is not carried into the preprocessed
stores.  The functions here build orderings from what the stores DO hold -- the
hit positions, the volume ids, and (for the truth-based variants) the perigee
targets.

Everything is plain numpy over the hits of one track, ``xyz`` of shape
``(L, 3)`` in mm.  All track parameters may be scalars or per-hit arrays, so the
same functions run over a whole CSR store at once (``np.repeat`` the parameters
by track length).

Conventions (ACTS): perigee point ``P = (-d0 sin phi, d0 cos phi, z0)``, unit
direction at the perigee ``(sin theta cos phi, sin theta sin phi, cos theta)``,
charge ``q = sign(qop)``, solenoid field ``(0, 0, Bz)`` with ``Bz = +2 T`` for the
ODD.  Lengths mm, angles rad, ``qop`` in 1/GeV.  These are the conventions of
:mod:`track_regression.perigee`, and :func:`helix_arc_length` is tested against
it.
"""

from __future__ import annotations

import numpy as np

KAPPA = 0.299792458      # R[m] = pT[GeV] / (KAPPA * Bz[T] * |q|)
DEFAULT_BZ = 2.0         # ODD solenoid, along +z

# ODD silicon volumes -> radial group.  0 = pixel, 1 = short strip, 2 = long
# strip.  The groups are nested shells (pixel r < 175 mm, short strip
# 240-705 mm, long strip 810-1035 mm), so an outgoing track crosses them in
# this order, and within a group it has to leave the barrel cylinder through
# its end face before it can reach that group's discs (the discs sit beyond the
# barrel half-length at radii inside the barrel envelope).
VOLUME_GROUP = {16: 0, 17: 0, 18: 0, 23: 1, 24: 1, 25: 1, 28: 2, 29: 2, 30: 2}
BARREL_VOLUMES = (17, 24, 29)
# Two hits of one track on the same barrel layer at the same radius (a track
# crossing the boundary between two modules of a stave) differ in r only by
# digitisation noise; quantising r makes them an exact tie so the z direction
# decides.  Genuinely different sensors of a layer are >= 1 mm apart in r.
BARREL_R_QUANTUM = 0.1

_GROUP_TABLE = np.full(64, -1, dtype=np.int64)
for _v, _g in VOLUME_GROUP.items():
    _GROUP_TABLE[_v] = _g


def _xyz(xyz):
    xyz = np.asarray(xyz, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] < 3:
        raise ValueError(f"xyz must have shape (L, >=3), got {xyz.shape}")
    return xyz[:, 0], xyz[:, 1], xyz[:, 2]


def order(key) -> np.ndarray:
    """Permutation that sorts ``key`` ascending (stable, so ties keep input order)."""
    return np.argsort(np.asarray(key), kind="stable")


# ---------------------------------------------------------------------------
# truth-free keys
# ---------------------------------------------------------------------------

def distance_from_origin(xyz) -> np.ndarray:
    """``s = sqrt(x^2 + y^2 + z^2)`` -- the legacy key (hit feature column 6)."""
    x, y, z = _xyz(xyz)
    return np.sqrt(x * x + y * y + z * z)


def cylindrical_radius(xyz) -> np.ndarray:
    """``r = sqrt(x^2 + y^2)``.  Monotonic along a helix that started near the
    beam line as long as the track has not reached its maximum radius
    ``2R + |d0|``; in the ODD (r < 1035 mm, Bz = 2 T) that holds for every track
    with ``pT > 0.31 GeV``.  Poorly measured on endcap strips (radial strips)."""
    x, y, _ = _xyz(xyz)
    return np.hypot(x, y)


def z_direction(xyz):
    """+1 if the track runs towards +z, -1 towards -z, read off its own hits.

    ``z`` is monotonic along any track in a solenoid (``p_z`` is conserved), so
    the sign of ``z(outermost hit) - z(innermost hit)`` is the direction, with
    inner/outer taken by radius.  Truth-free.
    """
    x, y, z = _xyz(xyz)
    r = np.hypot(x, y)
    return 1.0 if z[np.argmax(r)] >= z[np.argmin(r)] else -1.0


def geometry_keys(xyz, volume_id, direction=None):
    """Detector-order keys ``(primary, secondary)`` without any truth input.

    ``primary = (2 * group + endcap) * 4096 + c`` with ``c = r`` (quantised to
    ``BARREL_R_QUANTUM``) on a barrel layer and ``c = direction * z`` on a disc,
    i.e. tracks are ordered pixel -> short strip -> long strip, inside each
    group barrel before endcap, inside the barrel by radius and inside the
    endcap by z along the track's direction of flight -- each time the
    coordinate that the module position fixes exactly (barrel sensors sit at a
    fixed radius, disc sensors at a fixed z), so the poorly measured strip
    coordinate never enters.  ``secondary = direction * z`` breaks radius ties
    in the barrel (``r`` breaks ties on a disc).  4096 > 3100 mm, the largest
    |z| in the ODD, keeps the blocks disjoint.  The ODD layer ids order the
    same way inside a volume (barrel layer id grows with r; disc layer id grows
    with |z| on the +z side and shrinks with |z| on the -z side), so this is
    the layer order without a hand-written table.

    ``direction`` is :func:`z_direction` of the track; pass it explicitly (a
    scalar or a per-hit array) when calling on many tracks at once.
    """
    x, y, z = _xyz(xyz)
    vol = np.rint(np.asarray(volume_id, dtype=np.float64)).astype(np.int64)
    if vol.min() < 0 or vol.max() >= len(_GROUP_TABLE):
        raise ValueError(f"unknown volume id(s): {np.unique(vol)}")
    group = _GROUP_TABLE[vol]
    if (group < 0).any():
        raise ValueError(f"unknown volume id(s): {np.unique(vol[group < 0])}")
    if direction is None:
        direction = z_direction(xyz)
    endcap = ~np.isin(vol, BARREL_VOLUMES)
    r = np.hypot(x, y)
    rq = np.round(r / BARREL_R_QUANTUM) * BARREL_R_QUANTUM
    sz = np.asarray(direction, dtype=np.float64) * z
    primary = (2 * group + endcap) * 4096.0 + np.where(endcap, sz, rq)
    secondary = np.where(endcap, r, sz)
    return primary, secondary


def geometry_order(xyz, volume_id) -> np.ndarray:
    """Permutation of one track's hits into detector order (see :func:`geometry_keys`)."""
    primary, secondary = geometry_keys(xyz, volume_id)
    return np.lexsort((secondary, primary))


# ---------------------------------------------------------------------------
# truth-based keys (need the perigee parameters)
# ---------------------------------------------------------------------------

def perigee_point(d0, z0, phi):
    """``P = (-d0 sin phi, d0 cos phi, z0)``, the point of closest approach to the beam line."""
    d0, z0, phi = (np.asarray(v, dtype=np.float64) for v in (d0, z0, phi))
    return -d0 * np.sin(phi), d0 * np.cos(phi), z0


def distance_from_perigee(xyz, d0, z0, phi) -> np.ndarray:
    """``|X - P|`` -- the legacy ``s`` re-anchored at the track's own perigee."""
    x, y, z = _xyz(xyz)
    px, py, pz = perigee_point(d0, z0, phi)
    return np.sqrt((x - px) ** 2 + (y - py) ** 2 + (z - pz) ** 2)


def helix_arc_length(xyz, d0, z0, phi, theta, qop, volume_id=None, *,
                     Bz: float = DEFAULT_BZ, mode: str = "mixed",
                     unwrap_min_pitch: float = 300.0) -> np.ndarray:
    """3-D path length from the perigee to each hit along the ideal helix.

    Time along a charged track in a uniform field is proportional to this path
    length, so sorting by it reproduces the ACTS truth-time order whenever the
    real trajectory stays close to the helix through the truth perigee.

    Two estimators of the same path length ``l``::

        R      = pT / (KAPPA * Bz)  [m -> mm],  pT = sin(theta) / |qop|
        C      = P + q R (sin phi, -cos phi)           circle centre
        dalpha = wrap[0, 2pi)( -q (atan2(X - C) - atan2(P - C)) )
                                                       forward turning angle
        l_T    = R dalpha / sin(theta)                 from the transverse plane
        l_L    = (z - z0) / cos(theta)                 from z alone

    ``mode="transverse"`` returns ``l_T`` for every hit, ``"longitudinal"``
    ``l_L``, and ``"mixed"`` (default, needs ``volume_id``) ``l_T`` on barrel
    volumes and ``l_L`` on endcap volumes -- barrel strip sensors measure the
    r-phi coordinate but not z, endcap strip sensors measure phi but not r, so
    the mixed estimator only ever uses the coordinate that the module fixes.

    Loopers: ``dalpha`` is unwrapped with the z coordinate,
    ``n = round((z - z0) / (R cot theta) / 2pi - dalpha / 2pi)`` extra turns,
    whenever a full turn advances z by more than ``unwrap_min_pitch`` (so a
    mis-measured strip z cannot flip ``n``).  Not needed in the ODD above
    pT = 0.31 GeV, where no selected track completes a half turn.
    """
    x, y, z = _xyz(xyz)
    d0, z0, phi, theta, qop = (np.asarray(v, dtype=np.float64)
                               for v in (d0, z0, phi, theta, qop))
    q = np.sign(qop)
    sin_t, cos_t = np.sin(theta), np.cos(theta)
    pt = sin_t / np.abs(qop)
    R = pt / (KAPPA * Bz) * 1000.0

    px, py, _ = perigee_point(d0, z0, phi)
    cx, cy = px + q * R * np.sin(phi), py - q * R * np.cos(phi)
    alpha = np.arctan2(y - cy, x - cx)
    alpha_p = np.arctan2(py - cy, px - cx)
    two_pi = 2.0 * np.pi
    dalpha = np.remainder(-q * (alpha - alpha_p), two_pi)

    with np.errstate(divide="ignore", invalid="ignore"):
        cot = cos_t / sin_t
        pitch = two_pi * R * np.abs(cot)
        n = np.floor((z - z0) / (R * cot) / two_pi - dalpha / two_pi + 0.5)
        n = np.where(pitch > unwrap_min_pitch, np.maximum(n, 0.0), 0.0)
        dalpha = dalpha + two_pi * n
        l_t = R * dalpha / sin_t
        l_l = np.where(np.abs(cos_t) > 1e-9, (z - z0) / cos_t, l_t)

    if mode == "transverse":
        return l_t
    if mode == "longitudinal":
        return l_l
    if mode != "mixed":
        raise ValueError(f"mode must be 'mixed', 'transverse' or 'longitudinal', got {mode!r}")
    if volume_id is None:
        raise ValueError("mode='mixed' needs volume_id")
    vol = np.rint(np.asarray(volume_id, dtype=np.float64)).astype(np.int64)
    endcap = ~np.isin(vol, BARREL_VOLUMES)
    return np.where(endcap, l_l, l_t)
