"""Scale-free anchored head: target (t - a)/(|a| + eps) in the loss, inverse in predict_physical."""
import torch
from track_regression.losses import TrackParameterLoss

def _loss(scale):
    cfg = {p: {"type": "quantile", "weight": 1.0, "norm_min": -1.0, "norm_max": 1.0, "quantiles": [0.1, 0.5, 0.9]} for p in ("d0", "z0", "phi", "theta", "qop")}
    cfg["qop"] = dict(cfg["qop"], norm_min=-5.0, norm_max=5.0, delta_anchor="seed_qop", **({"scale_anchor_eps": 0.02} if scale else {}))
    return TrackParameterLoss(parameter_order=["d0", "z0", "phi", "theta", "qop"], config=cfg)

def test_scaled_anchor_roundtrip():
    lm = _loss(True); n = 64
    tgt = {p: torch.randn(n) * 0.1 for p in ("d0", "z0", "phi", "theta")}
    seed = torch.randn(n) * 0.3; tgt["qop"] = seed + torch.randn(n) * 0.01; tgt["seed_qop"] = seed
    tgt["track_valid"] = torch.ones(n, dtype=torch.bool)
    # a raw output whose q/p median channel equals the exact scaled residual must reconstruct q/p exactly
    start, end = lm._output_slices["qop"]; pred = torch.zeros(n, end - start + start)
    d_norm = ((tgt["qop"] - seed) / (seed.abs() + 0.02))            # physical scaled residual
    u = 2.0 * (d_norm - (-5.0)) / 10.0 - 1.0                          # -> [-1, 1] space of the head
    # ordered ladder: base channel = q10; deltas via softplus -> put the median exactly at u by using huge negative deltas ~0
    raw = torch.full((n, 3), -30.0); raw[:, 0] = u                    # q10 = u, increments ~ 0 -> median = u
    pred[:, start:end] = raw
    phys = lm.predict_physical(pred, tgt)
    assert torch.allclose(phys["qop"], tgt["qop"], atol=1e-4), (phys["qop"] - tgt["qop"]).abs().max()

def test_unscaled_anchor_unchanged():
    lm = _loss(False); assert "qop" not in lm._scale_eps and lm._delta_anchors["qop"] == "seed_qop"
