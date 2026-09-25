"""Seed-anchored quantile losses for the five perigee track parameters.

Every parameter has a pinball (quantile) head: the network emits ``Q`` raw
channels per parameter which are turned into a strictly increasing quantile
ladder (base + cumulative softplus gaps); the median is the point estimate.

Targets are anchored to the analytic seed (``delta_anchor: seed_<param>``):
the head regresses ``target - seed`` (azimuthal difference wrapped to
``(-pi, pi]``), linearly mapped from ``[norm_min, norm_max]`` to ``[-1, 1]``.
With ``scale_anchor_eps`` the residual is made scale-free,
``(target - seed) / (|seed| + eps)`` — used for q/p so that a 1 GeV and a
100 GeV track put the same relative demand on the output precision.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def _linear_normalise(x: Tensor, norm_min: Tensor, norm_max: Tensor) -> Tensor:
    """Map a physical value to [-1, 1]."""
    return 2.0 * (x - norm_min) / (norm_max - norm_min) - 1.0


def _linear_denormalise(u: Tensor, norm_min: Tensor, norm_max: Tensor) -> Tensor:
    """Map [-1, 1] back to a physical value."""
    return (u + 1.0) / 2.0 * (norm_max - norm_min) + norm_min


def _wrap(x: Tensor) -> Tensor:
    """Wrap an angle to [-pi, pi)."""
    return torch.remainder(x + math.pi, 2.0 * math.pi) - math.pi


class QuantileLoss(nn.Module):
    """Pinball loss on a linearly normalised target, one output per quantile."""

    def __init__(
        self,
        quantiles: list[float],
        norm_min: float = -1.0,
        norm_max: float = 1.0,
        weight: float = 1.0,
        monotone_eps: float = 1.0e-6,
    ):
        super().__init__()
        if any(q2 <= q1 for q1, q2 in zip(quantiles[:-1], quantiles[1:])) or not all(0 < q < 1 for q in quantiles):
            raise ValueError(f"quantiles must be strictly increasing in (0, 1), got {quantiles}")
        self.weight = weight
        self.monotone_eps = monotone_eps
        self.register_buffer("quantiles", torch.tensor(quantiles, dtype=torch.float32))
        self.register_buffer("norm_min", torch.tensor(norm_min, dtype=torch.float32))
        self.register_buffer("norm_max", torch.tensor(norm_max, dtype=torch.float32))
        self._median_idx = min(range(len(quantiles)), key=lambda i: abs(quantiles[i] - 0.5))

    @property
    def num_outputs(self) -> int:
        return len(self.quantiles)

    def _ordered_from_raw(self, raw: Tensor) -> Tensor:
        """Map raw channels to strictly ordered quantile values."""
        base = raw[..., :1]
        deltas = F.softplus(raw[..., 1:]) + self.monotone_eps
        return torch.cat([base, base + torch.cumsum(deltas, dim=-1)], dim=-1)

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """pred: (N, Q) raw channels, target: (N,) physical (anchored) value."""
        t_norm = _linear_normalise(target, self.norm_min, self.norm_max).unsqueeze(-1)
        diff = t_norm - self._ordered_from_raw(pred)
        tau = self.quantiles.unsqueeze(0)
        return self.weight * torch.max(tau * diff, (tau - 1) * diff).mean(dim=-1).mean()

    def predict(self, raw: Tensor) -> Tensor:
        """Median quantile in physical (anchored) units."""
        ordered = self._ordered_from_raw(raw)
        return _linear_denormalise(ordered[..., self._median_idx], self.norm_min, self.norm_max)

    def predict_quantiles(self, raw: Tensor) -> Tensor:
        """All ordered quantiles in physical (anchored) units."""
        return _linear_denormalise(self._ordered_from_raw(raw), self.norm_min, self.norm_max)


class TrackParameterLoss(nn.Module):
    """Sum of the per-parameter quantile losses.

    Parameters
    ----------
    config : dict[str, dict]
        Per-parameter kwargs of :class:`QuantileLoss` (``quantiles``,
        ``norm_min``, ``norm_max``, ``weight``) plus the optional anchoring
        keys ``delta_anchor`` (name of the target entry holding the anchor,
        e.g. ``seed_d0``) and ``scale_anchor_eps``.
    parameter_order : list[str]
        Order of the parameters' channel slices in the model output.
    """

    def __init__(self, config: dict[str, dict[str, Any]],
                 parameter_order: list[str] | None = None):
        super().__init__()
        self.parameter_order = parameter_order or ["d0", "z0", "phi", "theta", "qop"]
        self.losses = nn.ModuleDict()
        self._delta_anchors: dict[str, str] = {}
        self._scale_eps: dict[str, float] = {}
        for name in self.parameter_order:
            cfg = dict(config[name])
            anchor = cfg.pop("delta_anchor", None)
            if anchor is not None:
                self._delta_anchors[name] = anchor
            eps = cfg.pop("scale_anchor_eps", None)
            if eps is not None:
                assert anchor is not None, f"{name}: scale_anchor_eps needs delta_anchor"
                self._scale_eps[name] = float(eps)
            self.losses[name] = QuantileLoss(**cfg)
        self._output_slices: dict[str, tuple[int, int]] = {}
        offset = 0
        for name in self.parameter_order:
            n = self.losses[name].num_outputs
            self._output_slices[name] = (offset, offset + n)
            offset += n
        self._total_outputs = offset

    @property
    def total_outputs(self) -> int:
        """Number of raw outputs the model must produce."""
        return self._total_outputs

    def _raw(self, pred: Tensor, name: str) -> Tensor:
        start, end = self._output_slices[name]
        return pred[..., start:end]

    def forward(self, pred: Tensor, targets: dict[str, Tensor]) -> dict[str, Tensor]:
        """Per-parameter losses plus their sum under ``"total"``."""
        losses: dict[str, Tensor] = {}
        for name in self.parameter_order:
            t = targets[name]
            if name in self._delta_anchors:
                anchor = targets[self._delta_anchors[name]]
                t = t - anchor
                if name == "phi":
                    t = _wrap(t)
                if name in self._scale_eps:
                    t = t / (anchor.abs() + self._scale_eps[name])
            losses[name] = self.losses[name](self._raw(pred, name), t)
        losses["total"] = sum(losses[n] for n in self.parameter_order)
        return losses

    def predict_physical(self, pred: Tensor, anchors: dict[str, Tensor]) -> dict[str, Tensor]:
        """Physical point predictions: median + anchor (``anchors`` holds ``seed_<param>``)."""
        preds: dict[str, Tensor] = {}
        for name in self.parameter_order:
            p = self.losses[name].predict(self._raw(pred, name))
            if name in self._delta_anchors:
                a = anchors[self._delta_anchors[name]]
                if name in self._scale_eps:
                    p = p * (a.abs() + self._scale_eps[name])
                p = p + a
                if name == "phi":
                    p = _wrap(p)
            preds[name] = p
        return preds

    def predict_quantiles(self, pred: Tensor) -> dict[str, Tensor]:
        """Full quantile ladders, in the anchored (residual) units of each head."""
        return {n: self.losses[n].predict_quantiles(self._raw(pred, n)) for n in self.parameter_order}
