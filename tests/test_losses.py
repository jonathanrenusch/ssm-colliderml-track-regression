"""Seed-anchored quantile heads: the loss target and predict_physical are inverse."""

import math

import torch

from track_regression.losses import QuantileLoss, TrackParameterLoss

P = ("d0", "z0", "phi", "theta", "qop")
Q = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
RANGES = {"d0": 0.4, "z0": 3.5, "phi": 0.015, "theta": 0.01, "qop": 2.0}


def _loss():
    cfg = {p: {"norm_min": -r, "norm_max": r, "delta_anchor": f"seed_{p}", "quantiles": Q} for p, r in RANGES.items()}
    cfg["qop"]["scale_anchor_eps"] = 0.02
    return TrackParameterLoss(cfg)


def _raw_for(loss, residuals):
    """Raw outputs whose median is exactly the given (anchored) residual."""
    raw = torch.full((len(residuals["d0"]), loss.total_outputs), -30.0)   # ladder gaps ~ 0
    for p in P:
        s, _ = loss._output_slices[p]
        r = RANGES[p]
        raw[:, s] = 2.0 * (residuals[p] + r) / (2 * r) - 1.0
    return raw


def test_anchored_heads_roundtrip():
    torch.manual_seed(0)
    loss, n = _loss(), 256
    seed = {f"seed_{p}": torch.randn(n) * s for p, s in zip(P, (1.0, 50.0, 3.0, 1.0, 0.5))}
    truth = {p: seed[f"seed_{p}"] + torch.randn(n) * 0.1 * RANGES[p] for p in P}
    truth["phi"] = torch.remainder(truth["phi"] + math.pi, 2 * math.pi) - math.pi
    res = {p: truth[p] - seed[f"seed_{p}"] for p in P}
    res["phi"] = torch.remainder(res["phi"] + math.pi, 2 * math.pi) - math.pi
    res["qop"] = res["qop"] / (seed["seed_qop"].abs() + 0.02)
    pred = loss.predict_physical(_raw_for(loss, res), seed)
    for p in P:
        d = pred[p] - truth[p]
        if p == "phi":
            d = torch.remainder(d + math.pi, 2 * math.pi) - math.pi
        assert d.abs().max() < 1e-4 * RANGES[p], p
    assert torch.isfinite(loss(_raw_for(loss, res), {**truth, **seed})["total"])


def test_quantile_ladder_is_ordered():
    ql = QuantileLoss(Q)
    q = ql.predict_quantiles(torch.randn(100, 7))
    assert (q[:, 1:] > q[:, :-1]).all()
