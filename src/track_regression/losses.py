"""Config-driven loss functions for track parameter regression.

Every parameter's loss behaviour is determined *entirely* by the YAML config.
The :class:`TrackParameterLoss` module reads the ``losses`` section of the
config and instantiates the appropriate per-parameter loss object.

Supported loss types
--------------------
``smooth_l1``
    Direct Smooth-L1 on (optionally pre-normalised) targets.
    Smooth-L1 in quantile-spline-normalised ``[0, 1]`` space.
``quantile``
    Pinball (quantile) loss directly on physical / normalised targets.
``quantile_eta``
    Pinball loss in pseudorapidity space (for θ).
    Pinball loss in spline-normalised space.
``circular``
    ``SmoothL1(sin) + SmoothL1(cos)`` for angular parameters (phi).
``gaussian``
    Gaussian NLL on normalised targets, outputs ``(mu, raw_var)``.
    Set ``variance_param: softplus`` for bounded-gradient parameterisation.
``gaussian_eta``
    Gaussian NLL in η-space (for θ), outputs ``(mu, raw_var)``.
``cosine_phi``
    ``1 - cos(phi_pred - phi_true)`` via (sin, cos) outputs for stability.
``mixture_density``
    Gaussian-mixture NLL with K components, outputs ``3K`` channels
    ``[π_k, μ_k, σ_k]``. Breaks the mode-collapse attractor on
    sharply-peaked targets (e.g. d0).

All individual components expose ``.forward(pred, target) → loss``
and ``.predict(raw_output) → physical_value``.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import yaml
from torch import Tensor, nn


# Directory containing this file — used to resolve relative spline config paths
_EXPERIMENT_DIR = Path(__file__).resolve().parent


def _resolve_spline_path(spline_config: str | Path) -> Path:
    """Resolve a spline config path relative to the experiment directory."""
    p = Path(spline_config)
    if p.is_absolute():
        return p
    # Try relative to experiment dir first (handles 'config/splines/...')
    resolved = _EXPERIMENT_DIR / p
    if resolved.exists():
        return resolved
    # Fall back to CWD-relative (original behaviour)
    return p


# ============================================================================
# Shared normalisation helpers
# ============================================================================


def _linear_normalise(x: Tensor, norm_min: Tensor, norm_max: Tensor) -> Tensor:
    """Map physical value to [-1, 1]."""
    return 2.0 * (x - norm_min) / (norm_max - norm_min) - 1.0


def _linear_denormalise(u: Tensor, norm_min: Tensor, norm_max: Tensor) -> Tensor:
    """Map [-1, 1] back to physical value."""
    return (u + 1.0) / 2.0 * (norm_max - norm_min) + norm_min


def _validate_quantiles(quantiles: list[float]) -> None:
    """Validate quantile levels are inside (0, 1) and strictly increasing."""
    if len(quantiles) == 0:
        raise ValueError("quantiles must contain at least one value")
    if any((q <= 0.0 or q >= 1.0) for q in quantiles):
        raise ValueError(f"quantiles must be in (0, 1), got: {quantiles}")
    if any(q2 <= q1 for q1, q2 in zip(quantiles[:-1], quantiles[1:], strict=False)):
        raise ValueError(f"quantiles must be strictly increasing, got: {quantiles}")


def _raw_crossing_stats(raw: Tensor) -> dict[str, Tensor]:
    """Compute crossing stats from raw quantile channels.

    NOTE: These statistics are computed on the **raw unconstrained** model
    outputs (base + delta channels), NOT on the ordered quantile predictions.
    The actual quantile predictions never cross thanks to the softplus + cumsum
    construction.  A high crossing rate here simply means the raw network
    outputs do not naturally maintain ordering — which is expected and harmless.

    Crossing is measured on adjacent raw channels as
    ``max(raw_i - raw_{i+1}, 0)``.
    """
    if raw.numel() == 0 or raw.shape[-1] < 2:
        z = raw.new_tensor(0.0)
        return {"rate": z, "mean_gap": z, "max_gap": z}

    gaps = (raw[..., :-1] - raw[..., 1:]).clamp_min(0.0)
    # Fraction of individual adjacent pairs that violate ordering
    rate = (gaps > 0).to(dtype=raw.dtype).mean()
    return {
        "rate": rate,
        "mean_gap": gaps.mean(),
        "max_gap": gaps.max(),
    }


def _quantile_calibration(
    ordered_quantiles: Tensor,
    target: Tensor,
    quantile_levels: Tensor,
) -> dict[str, Tensor]:
    """Compute quantile calibration: empirical coverage vs nominal levels.

    For a well-calibrated model, the fraction of targets below the τ-th
    predicted quantile should equal τ.

    Parameters
    ----------
    ordered_quantiles : Tensor
        Ordered quantile predictions ``(N, Q)`` in physical space.
    target : Tensor
        Ground truth values ``(N,)``.
    quantile_levels : Tensor
        Nominal quantile levels ``(Q,)``, e.g. [0.05, 0.1, 0.25, 0.5, ...].

    Returns
    -------
    dict[str, Tensor]
        ``calibration_error``: mean |empirical_coverage - nominal| across quantiles.
    """
    if target.numel() == 0 or ordered_quantiles.numel() == 0:
        z = target.new_tensor(0.0)
        return {"calibration_error": z}

    # (N, 1) < (N, Q) → (N, Q)
    below = (target.unsqueeze(-1) < ordered_quantiles).float()
    empirical_coverage = below.mean(dim=0)  # (Q,)
    calibration_error = (empirical_coverage - quantile_levels).abs().mean()
    return {"calibration_error": calibration_error}


# ============================================================================
# Per-parameter loss components
# ============================================================================


class SmoothL1Loss(nn.Module):
    """Direct Smooth-L1 on normalised targets.

    The model predicts a single scalar per parameter.  Targets are linearly
    normalised to ``[-1, 1]`` using the configured ``norm_min`` / ``norm_max``
    range before computing the loss.

    Parameters
    ----------
    norm_min : float
        Lower bound of the physical range for linear normalisation.
    norm_max : float
        Upper bound of the physical range for linear normalisation.
    weight : float
        Multiplicative weight applied to this parameter's loss.
    beta : float
        Smooth-L1 transition point.
    """

    num_outputs: int = 1

    def __init__(
        self,
        norm_min: float = -1.0,
        norm_max: float = 1.0,
        weight: float = 1.0,
        beta: float = 1.0,
    ):
        super().__init__()
        self.weight = weight
        self.beta = beta
        self.register_buffer("norm_min", torch.tensor(norm_min, dtype=torch.float32))
        self.register_buffer("norm_max", torch.tensor(norm_max, dtype=torch.float32))

    def forward(self, pred: Tensor, target: Tensor, sample_weights: Tensor | None = None) -> Tensor:
        """Compute loss.  Both tensors have shape ``(N,)`` or ``(N, 1)``."""
        t_norm = _linear_normalise(target, self.norm_min, self.norm_max)
        p = pred.squeeze(-1) if pred.dim() > target.dim() else pred
        per_sample = F.smooth_l1_loss(p, t_norm, beta=self.beta, reduction="none")
        if sample_weights is not None:
            loss = (sample_weights * per_sample).sum() / sample_weights.sum()
        else:
            loss = per_sample.mean()
        return self.weight * loss

    def predict(self, raw: Tensor) -> Tensor:
        """Convert raw model output to physical value."""
        return _linear_denormalise(raw.squeeze(-1), self.norm_min, self.norm_max)

class QuantileLoss(nn.Module):
    """Pinball (quantile / check) loss on normalised physical values.

    The model outputs one value per quantile.  The median (τ=0.5) serves
    as the point prediction.

    Parameters
    ----------
    quantiles : list[float]
        Quantile levels, e.g. ``[0.1, 0.25, 0.5, 0.75, 0.9]``.
    norm_min : float
        Lower bound for linear normalisation.
    norm_max : float
        Upper bound for linear normalisation.
    weight : float
        Loss weight.
    """

    def __init__(
        self,
        quantiles: list[float] | None = None,
        norm_min: float = -1.0,
        norm_max: float = 1.0,
        weight: float = 1.0,
        monotone_eps: float = 1.0e-6,
    ):
        super().__init__()
        if quantiles is None:
            quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
        _validate_quantiles(quantiles)
        self.weight = weight
        self.monotone_eps = monotone_eps
        self.register_buffer("quantiles", torch.tensor(quantiles, dtype=torch.float32))
        self.register_buffer("norm_min", torch.tensor(norm_min, dtype=torch.float32))
        self.register_buffer("norm_max", torch.tensor(norm_max, dtype=torch.float32))

    @property
    def num_outputs(self) -> int:
        return len(self.quantiles)

    def _ordered_from_raw(self, raw: Tensor) -> Tensor:
        """Map raw channels to strictly ordered quantile values."""
        if raw.shape[-1] < 2:
            return raw
        base = raw[..., :1]
        deltas = F.softplus(raw[..., 1:]) + self.monotone_eps
        return torch.cat([base, base + torch.cumsum(deltas, dim=-1)], dim=-1)

    def forward(self, pred: Tensor, target: Tensor, sample_weights: Tensor | None = None) -> Tensor:
        """pred: (N, num_quantiles),  target: (N,)."""
        t_norm = _linear_normalise(target, self.norm_min, self.norm_max).unsqueeze(-1)  # (N, 1)
        p = self._ordered_from_raw(pred)
        tau = self.quantiles.unsqueeze(0)  # (1, Q)
        diff = t_norm - p  # (N, Q)
        per_sample = torch.max(tau * diff, (tau - 1) * diff).mean(dim=-1)  # (N,)
        if sample_weights is not None:
            loss = (sample_weights * per_sample).sum() / sample_weights.sum()
        else:
            loss = per_sample.mean()
        return self.weight * loss

    def predict(self, raw: Tensor) -> Tensor:
        """Return median quantile mapped back to physical space."""
        ordered = self._ordered_from_raw(raw)
        median_idx = (self.quantiles - 0.5).abs().argmin()
        return _linear_denormalise(ordered[..., median_idx], self.norm_min, self.norm_max)

    def predict_quantiles(self, raw: Tensor) -> Tensor:
        """Return all ordered quantile predictions in physical space."""
        ordered = self._ordered_from_raw(raw)
        return _linear_denormalise(ordered, self.norm_min, self.norm_max)

    def raw_crossing_metrics(self, raw: Tensor) -> dict[str, Tensor]:
        """Return crossing metrics computed on the raw (unconstrained) channels."""
        return _raw_crossing_stats(raw)

    def calibration_metrics(self, raw: Tensor, target: Tensor) -> dict[str, Tensor]:
        """Return quantile calibration metrics on physical-space predictions."""
        ordered = self._ordered_from_raw(raw)
        ordered_phys = _linear_denormalise(ordered, self.norm_min, self.norm_max)
        return _quantile_calibration(ordered_phys, target, self.quantiles)

def _theta_to_eta(theta: Tensor) -> Tensor:
    """Convert polar angle θ ∈ (0, π) to pseudorapidity η = -ln(tan(θ/2))."""
    half = theta.clamp(1e-7, torch.pi - 1e-7) * 0.5
    return -torch.log(torch.tan(half))


def _eta_to_theta(eta: Tensor) -> Tensor:
    """Convert pseudorapidity η back to polar angle θ = 2·arctan(exp(-η))."""
    return 2.0 * torch.atan(torch.exp(-eta))


class EtaQuantileLoss(nn.Module):
    """Quantile loss that operates in pseudorapidity (η) space.

    Targets arrive as θ ∈ (0, π), are converted to η = -ln(tan(θ/2)),
    normalised to [-1, 1] using ``norm_min``/``norm_max`` (in η units),
    and the pinball loss is computed in that space.

    **Predictions are always returned in θ space** so that downstream
    metrics (MAE, precision) remain comparable to the θ-native loss.

    Parameters
    ----------
    quantiles : list[float]
        Quantile levels, e.g. ``[0.1, 0.25, 0.5, 0.75, 0.9]``.
    norm_min : float
        Lower bound of η range for linear normalisation (e.g. -5.0).
    norm_max : float
        Upper bound of η range for linear normalisation (e.g. 5.0).
    weight : float
        Loss weight.
    monotone_eps : float
        Minimum gap between adjacent quantile predictions.
    """

    def __init__(
        self,
        quantiles: list[float] | None = None,
        norm_min: float = -5.0,
        norm_max: float = 5.0,
        weight: float = 1.0,
        monotone_eps: float = 1.0e-6,
    ):
        super().__init__()
        if quantiles is None:
            quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
        _validate_quantiles(quantiles)
        self.weight = weight
        self.monotone_eps = monotone_eps
        self.register_buffer("quantiles", torch.tensor(quantiles, dtype=torch.float32))
        self.register_buffer("norm_min", torch.tensor(norm_min, dtype=torch.float32))
        self.register_buffer("norm_max", torch.tensor(norm_max, dtype=torch.float32))

    @property
    def num_outputs(self) -> int:
        return len(self.quantiles)

    def _ordered_from_raw(self, raw: Tensor) -> Tensor:
        """Map raw channels to strictly ordered quantile values."""
        if raw.shape[-1] < 2:
            return raw
        base = raw[..., :1]
        deltas = F.softplus(raw[..., 1:]) + self.monotone_eps
        return torch.cat([base, base + torch.cumsum(deltas, dim=-1)], dim=-1)

    def forward(self, pred: Tensor, target: Tensor, sample_weights: Tensor | None = None) -> Tensor:
        """pred: (N, num_quantiles),  target: (N,) in θ space."""
        eta = _theta_to_eta(target)
        t_norm = _linear_normalise(eta, self.norm_min, self.norm_max).unsqueeze(-1)  # (N, 1)
        p = self._ordered_from_raw(pred)
        tau = self.quantiles.unsqueeze(0)  # (1, Q)
        diff = t_norm - p  # (N, Q)
        per_sample = torch.max(tau * diff, (tau - 1) * diff).mean(dim=-1)  # (N,)
        if sample_weights is not None:
            loss = (sample_weights * per_sample).sum() / sample_weights.sum()
        else:
            loss = per_sample.mean()
        return self.weight * loss

    def predict(self, raw: Tensor) -> Tensor:
        """Return median quantile mapped back to θ space."""
        ordered = self._ordered_from_raw(raw)
        median_idx = (self.quantiles - 0.5).abs().argmin()
        eta_pred = _linear_denormalise(ordered[..., median_idx], self.norm_min, self.norm_max)
        return _eta_to_theta(eta_pred)

    def predict_quantiles(self, raw: Tensor) -> Tensor:
        """Return all ordered quantile predictions in θ space."""
        ordered = self._ordered_from_raw(raw)
        eta_preds = _linear_denormalise(ordered, self.norm_min, self.norm_max)
        return _eta_to_theta(eta_preds)

    def raw_crossing_metrics(self, raw: Tensor) -> dict[str, Tensor]:
        """Return crossing metrics computed on the raw (unconstrained) channels."""
        return _raw_crossing_stats(raw)

    def calibration_metrics(self, raw: Tensor, target: Tensor) -> dict[str, Tensor]:
        """Return quantile calibration metrics in θ space."""
        ordered = self._ordered_from_raw(raw)
        eta_preds = _linear_denormalise(ordered, self.norm_min, self.norm_max)
        theta_preds = _eta_to_theta(eta_preds)
        return _quantile_calibration(theta_preds, target, self.quantiles)


class CircularPhiLoss(nn.Module):
    """Smooth-L1 on sin/cos components for circular angular regression.

    The model outputs two values ``(sin_pred, cos_pred)``.  Loss is::

        SmoothL1(sin_pred - sin(phi_true)) + SmoothL1(cos_pred - cos(phi_true))

    Recovery: ``phi = atan2(sin_pred, cos_pred)``.

    Parameters
    ----------
    weight : float
        Loss weight.
    beta : float
        Smooth-L1 transition point.
    reduction : str
        ``"mean"`` (default) returns scalar; ``"none"`` returns the
        per-sample ``weight * loss_i`` tensor with ``sample_weights``
        multiplied in but not normalised.  Used by batch-trimming.
    """

    num_outputs: int = 2

    def __init__(self, weight: float = 1.0, beta: float = 1.0, reduction: str = "mean"):
        super().__init__()
        self.weight = weight
        self.beta = beta
        if reduction not in ("mean", "none"):
            raise ValueError(f"reduction must be 'mean' or 'none', got {reduction!r}")
        self.reduction = reduction

    def forward(self, pred: Tensor, target: Tensor, sample_weights: Tensor | None = None) -> Tensor:
        """pred: (N, 2) with [sin, cos];  target: (N,) with phi in radians."""
        sin_true = torch.sin(target)
        cos_true = torch.cos(target)
        sin_pred = pred[..., 0]
        cos_pred = pred[..., 1]
        per_sample = (
            F.smooth_l1_loss(sin_pred, sin_true, beta=self.beta, reduction="none")
            + F.smooth_l1_loss(cos_pred, cos_true, beta=self.beta, reduction="none")
        )

        if self.reduction == "none":
            if sample_weights is not None:
                per_sample = sample_weights * per_sample
            return self.weight * per_sample

        if sample_weights is not None:
            loss = (sample_weights * per_sample).sum() / sample_weights.sum()
        else:
            loss = per_sample.mean()
        return self.weight * loss

    def predict(self, raw: Tensor) -> Tensor:
        """Recover phi from (sin, cos) outputs."""
        return torch.atan2(raw[..., 0], raw[..., 1])


# ============================================================================
# Gaussian NLL losses (for the "Gaussian" loss family — sharpens core
# resolution at the cost of the quantile loss's tail focus)
# ============================================================================


class GaussianParameterLoss(nn.Module):
    """Gaussian negative log-likelihood on linearly normalised targets.

    The model outputs ``(mu, log_var)`` per parameter.  Loss is the standard
    Gaussian NLL with the constant term dropped (irrelevant for gradients)::

        per_sample = 0.5 * (exp(-log_var) * (target_norm - mu)**2 + log_var)

    Using ``log_var`` instead of ``var`` guarantees positive variance without
    clamping and prevents divide-by-zero / NaN gradient explosions.

    Targets are linearly normalised to ``[-1, 1]`` using ``norm_min`` /
    ``norm_max`` before the NLL is computed; predictions are denormalised
    back to physical space.  Analytic quantile predictions are available via
    ``predict_quantiles(raw, quantiles)``.

    Parameters
    ----------
    norm_min : float
        Lower bound of the physical range for linear normalisation.
    norm_max : float
        Upper bound of the physical range for linear normalisation.
    weight : float
        Multiplicative weight applied to this parameter's loss.
    log_var_clamp : float
        Symmetric clamp applied to ``log_var`` inside the NLL.  Protects
        against ``exp(-log_var)`` overflow under sharp loss-landscape
        perturbations (e.g. SAM ascent step) which can otherwise push
        ``log_var`` arbitrarily negative → NaN.  Default ``5.0`` limits
        σ_min to exp(-2.5) ≈ 0.08 in normalised space, preventing the
        catastrophic exp(8) ≈ 3000 multiplier that caused gradient spikes
        with the previous default of 8.0.
    log_var_init : float
        Fixed bias added to the raw ``log_var`` output before clamping.
        Shifts the effective starting point of the variance prediction so
        that a zero-initialised output head produces ``exp(-(0 +
        log_var_init))`` as the initial precision.  A positive value
        (e.g. ``1.0``) starts the model with lower precision (higher
        variance), which prevents the ``exp(-log_var) * residual²``
        gradient explosion that occurs when ``log_var`` is randomly
        negative at initialisation.  Default ``0.0`` preserves the
        original behaviour.
    reduction : str
        ``"mean"`` (default): return scalar, current behaviour.
        ``"none"``: return per-sample ``weight * per_sample`` with shape
        ``(N,)``, with ``sample_weights`` multiplied in but **not
        normalised**.  Used by batch-trimming to rank samples.
    """

    num_outputs: int = 2

    def __init__(
        self,
        norm_min: float = -1.0,
        norm_max: float = 1.0,
        weight: float = 1.0,
        log_var_clamp: float = 5.0,
        log_var_init: float = 0.0,
        variance_param: str = "log_var",
        var_init: float = 0.0,
        var_eps: float = 1e-6,
        reduction: str = "mean",
        beta_nll: float = 0.0,
    ):
        super().__init__()
        if variance_param not in ("log_var", "softplus"):
            raise ValueError(f"variance_param must be 'log_var' or 'softplus', got {variance_param!r}")
        self.weight = weight
        self.variance_param = variance_param
        self.log_var_clamp = float(log_var_clamp)
        self.beta_nll = float(beta_nll)
        if reduction not in ("mean", "none"):
            raise ValueError(f"reduction must be 'mean' or 'none', got {reduction!r}")
        self.reduction = reduction
        self.register_buffer("norm_min", torch.tensor(norm_min, dtype=torch.float32))
        self.register_buffer("norm_max", torch.tensor(norm_max, dtype=torch.float32))
        # Legacy log_var buffers (used when variance_param == "log_var")
        self.register_buffer("_log_var_init", torch.tensor(float(log_var_init), dtype=torch.float32))
        # Softplus buffers (used when variance_param == "softplus")
        self.register_buffer("_var_init", torch.tensor(float(var_init), dtype=torch.float32))
        self.register_buffer("_var_eps", torch.tensor(float(var_eps), dtype=torch.float32))

    def _nll(self, mu: Tensor, raw_var: Tensor, target_norm: Tensor) -> Tensor:
        """Full Gaussian NLL including the 0.5·log(2π) constant.

        Including the constant keeps the loss non-negative when the model
        is well-calibrated (σ ≈ residual std), which prevents confusing
        negative loss values on the dashboard and makes the loss landscape
        smoother near the optimum.

        When ``variance_param == "softplus"``, variance is parameterised as
        ``softplus(raw + bias) + eps`` instead of ``exp(log_var)``.  The
        softplus gradient is ``sigmoid(x) ∈ (0, 1)`` — always bounded, no
        clamping needed, no dead zones at saturation boundaries.

        When ``beta_nll > 0``, applies the β-NLL weighting from Seitzer et
        al. (ICLR 2022): each sample's NLL is multiplied by σ^(2β) with
        stop-gradient, preventing the model from inflating predicted
        variance to reduce the loss on hard examples.
        """
        residual_sq = (target_norm - mu) ** 2

        if self.variance_param == "softplus":
            var = F.softplus(raw_var + self._var_init) + self._var_eps
            nll_core = 0.5 * (residual_sq / var + torch.log(var))
        else:
            # Legacy exp(-log_var) path
            log_var = raw_var + self._log_var_init
            log_var = log_var.clamp(-self.log_var_clamp, self.log_var_clamp)
            var = torch.exp(log_var)
            nll_core = 0.5 * (residual_sq / var + log_var)

        # β-NLL (Seitzer et al., ICLR 2022): weight the data-dependent NLL by
        # σ^(2β) = var^β with stop-gradient to prevent variance inflation on
        # hard examples.  β=0.5 gives scale-invariant gradients w.r.t. μ.
        # The 0.5·log(2π) normalisation constant is added *outside* the
        # β-weighting — it has zero gradient so the math is unaffected, but
        # pulling it out keeps the logged loss interpretable (a clean
        # additive offset rather than a σ²ᵝ-scaled one).
        if self.beta_nll > 0.0:
            nll_core = var.detach().pow(self.beta_nll) * nll_core

        return nll_core + 0.5 * math.log(2.0 * math.pi)

    def forward(self, pred: Tensor, target: Tensor, sample_weights: Tensor | None = None) -> Tensor:
        """pred: (N, 2) with [mu, raw_var];  target: (N,) in physical units."""
        mu = pred[..., 0]
        raw_var = pred[..., 1]
        t_norm = _linear_normalise(target, self.norm_min, self.norm_max)
        per_sample = self._nll(mu, raw_var, t_norm)

        if self.reduction == "none":
            # Per-sample form: apply sample_weights as a multiplier (not a
            # normaliser) so downstream code can sum across parameters.
            if sample_weights is not None:
                per_sample = sample_weights * per_sample
            return self.weight * per_sample

        if sample_weights is not None:
            loss = (sample_weights * per_sample).sum() / sample_weights.sum()
        else:
            loss = per_sample.mean()
        return self.weight * loss

    def predict(self, raw: Tensor) -> Tensor:
        """Return the mean (``mu``) denormalised to physical units."""
        return _linear_denormalise(raw[..., 0], self.norm_min, self.norm_max)

    def _sigma_from_raw(self, raw_var: Tensor) -> Tensor:
        """Convert the raw variance channel to σ (std dev) in normalised space."""
        if self.variance_param == "softplus":
            var = F.softplus(raw_var + self._var_init) + self._var_eps
            return torch.sqrt(var)
        return torch.exp(0.5 * (raw_var + self._log_var_init))

    def predict_quantiles(self, raw: Tensor, quantiles: list[float] | Tensor | None = None) -> Tensor:
        """Analytic quantiles from (mu, raw_var).

        For a Gaussian, ``Q(τ) = μ + σ · Φ⁻¹(τ)``.  The inverse CDF is
        computed via ``Φ⁻¹(τ) = √2 · erfinv(2τ - 1)``.  Both ``mu`` and
        ``sigma`` are converted to physical units before the quantile is
        formed, so the return value is in physical space.

        Parameters
        ----------
        raw : Tensor
            Model output of shape ``(N, 2)`` with ``[mu, raw_var]``.
        quantiles : list[float] | Tensor | None
            Quantile levels to sample.  Defaults to the standard
            ``[0.05, 0.25, 0.5, 0.75, 0.95]`` five-point summary.
        """
        if quantiles is None:
            quantiles = [0.05, 0.25, 0.5, 0.75, 0.95]
        if not isinstance(quantiles, Tensor):
            quantiles = torch.tensor(list(quantiles), dtype=raw.dtype, device=raw.device)

        mu_phys = _linear_denormalise(raw[..., 0], self.norm_min, self.norm_max)
        # σ is in normalised space; the linear de-normalisation scales it by
        # (norm_max - norm_min) / 2 (same scale factor applied to every coord).
        half_range = 0.5 * (self.norm_max - self.norm_min)
        sigma_phys = self._sigma_from_raw(raw[..., 1]) * half_range
        z = math.sqrt(2.0) * torch.erfinv(2.0 * quantiles - 1.0)  # Φ⁻¹(τ)
        return mu_phys.unsqueeze(-1) + sigma_phys.unsqueeze(-1) * z  # (N, Q)


class GaussianEtaLoss(GaussianParameterLoss):
    """Gaussian NLL in pseudorapidity (η) space for the θ parameter.

    Mirrors :class:`EtaQuantileLoss`: targets arrive as θ ∈ (0, π), are
    converted to η = -ln(tan(θ/2)), normalised to [-1, 1] in η units, and
    the Gaussian NLL is computed in that space.  **Predictions are returned
    in θ space** (via the inverse η→θ map) so downstream metrics remain
    comparable to the θ-native quantile loss.
    """

    num_outputs: int = 2

    def forward(self, pred: Tensor, target: Tensor, sample_weights: Tensor | None = None) -> Tensor:
        # Convert θ → η once and delegate to the parent's NLL on the η-space target.
        return super().forward(pred, _theta_to_eta(target), sample_weights)

    def predict(self, raw: Tensor) -> Tensor:
        """Return the mean η denormalised and converted back to θ."""
        eta_pred = _linear_denormalise(raw[..., 0], self.norm_min, self.norm_max)
        return _eta_to_theta(eta_pred)

    def predict_quantiles(self, raw: Tensor, quantiles: list[float] | Tensor | None = None) -> Tensor:
        """Analytic quantiles in θ space.

        Compute quantiles in η space (where the Gaussian lives), then apply
        the monotonic η→θ map.  Note: η→θ is monotonically decreasing, so the
        resulting θ quantiles are in *reverse* order.  We sort descending to
        keep the returned tensor monotonically increasing in θ.
        """
        if quantiles is None:
            quantiles = [0.05, 0.25, 0.5, 0.75, 0.95]
        if not isinstance(quantiles, Tensor):
            quantiles = torch.tensor(list(quantiles), dtype=raw.dtype, device=raw.device)

        mu_eta = raw[..., 0]  # in normalised η-space
        sigma_norm = self._sigma_from_raw(raw[..., 1])
        z = math.sqrt(2.0) * torch.erfinv(2.0 * quantiles - 1.0)
        # quantiles in normalised η-space: μ + σ · z
        eta_norm_q = mu_eta.unsqueeze(-1) + sigma_norm.unsqueeze(-1) * z  # (N, Q)
        eta_phys_q = _linear_denormalise(eta_norm_q, self.norm_min, self.norm_max)
        theta_q = _eta_to_theta(eta_phys_q)  # (N, Q), decreasing in q
        # Sort to ascending θ order so downstream code sees monotonic quantiles.
        return torch.sort(theta_q, dim=-1).values


class MixtureDensityLoss(nn.Module):
    """Gaussian-mixture density network loss for heavy-tailed scalar regression.

    The model outputs ``3 * n_components`` channels per track:
        ``[π₁..π_K, μ₁..μ_K, σ₁..σ_K]`` (raw).
    After activations:
        π = softmax(π_raw),  μ = μ_raw + init_means,  σ = softplus(σ_raw + σ_init) + σ_floor.
    The loss is the per-sample negative log-likelihood of the mixture,
    ``−log Σₖ π_k · N(t | μ_k, σ_k²)``, computed stably via logsumexp.

    Designed to fix the mode-collapse attractor seen when the target
    distribution is sharply peaked (e.g. d0, 95 % within 30 μm): a
    mixture lets the network hedge uncertainty across well-separated
    components instead of collapsing μ toward the mode.

    Parameters
    ----------
    n_components : int
        Number of Gaussian mixture components (K). Typical: 2–3.
    norm_min, norm_max : float
        Linear normalisation bounds applied to targets and predictions
        (same convention as every other loss in this file).
    weight : float
        Multiplicative weight applied to the loss.
    init_means : list[float]
        Initial μ_k values **in normalised [-1, 1] space**. Must have
        length ``n_components``. Keep them spread out (e.g.
        ``[-0.1, 0.0, 0.1]``) so components don't stack at init.
    sigma_init : float
        Bias added to the raw σ channel before softplus.  σ starts at
        ``softplus(sigma_init)`` (plus floor).  Default −2.0 → σ ≈ 0.13
        in normalised space, roughly the effective residual scale
        during early training.
    sigma_floor : float
        Minimum σ (numerical safety floor, in normalised units).
    tail_weight_tau : float
        If > 0, apply a continuous per-track weight
        ``w = max(1, |target| / tail_weight_tau)`` (τ in physical
        units) to the per-sample loss.  Counters the
        dominant-prompt-track gradient imbalance.  Set to 0 to disable.
    reduction : str
        ``"mean"`` (default) or ``"none"``.
    """

    def __init__(
        self,
        n_components: int = 3,
        norm_min: float = -1.0,
        norm_max: float = 1.0,
        weight: float = 1.0,
        init_means: list[float] | None = None,
        sigma_init: float = -2.0,
        sigma_floor: float = 1.0e-4,
        tail_weight_tau: float = 0.0,
        reduction: str = "mean",
    ):
        super().__init__()
        if n_components < 1:
            raise ValueError(f"n_components must be >= 1, got {n_components}")
        if init_means is None:
            # Default: evenly spread over a small subrange of normalised space
            span = 0.2
            if n_components == 1:
                init_means = [0.0]
            else:
                init_means = [
                    -span + 2 * span * k / (n_components - 1)
                    for k in range(n_components)
                ]
        if len(init_means) != n_components:
            raise ValueError(
                f"init_means length ({len(init_means)}) != n_components ({n_components})"
            )
        if reduction not in ("mean", "none"):
            raise ValueError(f"reduction must be 'mean' or 'none', got {reduction!r}")

        self.n_components = n_components
        self.weight = weight
        self.tail_weight_tau = float(tail_weight_tau)
        self.reduction = reduction

        self.register_buffer("norm_min", torch.tensor(norm_min, dtype=torch.float32))
        self.register_buffer("norm_max", torch.tensor(norm_max, dtype=torch.float32))
        self.register_buffer(
            "init_means", torch.tensor(init_means, dtype=torch.float32)
        )
        self.register_buffer(
            "sigma_init", torch.tensor(float(sigma_init), dtype=torch.float32)
        )
        self.register_buffer(
            "sigma_floor", torch.tensor(float(sigma_floor), dtype=torch.float32)
        )

    @property
    def num_outputs(self) -> int:
        return 3 * self.n_components

    def _components(self, raw: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Split raw output into (log_pi, mu, sigma) in normalised space."""
        K = self.n_components
        pi_logits = raw[..., :K]
        mu_raw = raw[..., K:2 * K]
        sigma_raw = raw[..., 2 * K:3 * K]

        log_pi = F.log_softmax(pi_logits, dim=-1)
        mu = mu_raw + self.init_means
        sigma = F.softplus(sigma_raw + self.sigma_init) + self.sigma_floor
        return log_pi, mu, sigma

    def forward(self, pred: Tensor, target: Tensor, sample_weights: Tensor | None = None) -> Tensor:
        """pred: (N, 3K), target: (N,) in physical units."""
        log_pi, mu, sigma = self._components(pred)
        t_norm = _linear_normalise(target, self.norm_min, self.norm_max).unsqueeze(-1)

        log_gauss = (
            -0.5 * ((t_norm - mu) / sigma).pow(2)
            - torch.log(sigma)
            - 0.5 * math.log(2.0 * math.pi)
        )  # (N, K)
        log_mix = torch.logsumexp(log_pi + log_gauss, dim=-1)  # (N,)
        per_sample = -log_mix

        # Continuous tail upweighting by |target| (in physical units)
        if self.tail_weight_tau > 0.0:
            tail_w = torch.clamp(target.abs() / self.tail_weight_tau, min=1.0)
            per_sample = tail_w * per_sample

        if self.reduction == "none":
            if sample_weights is not None:
                per_sample = sample_weights * per_sample
            return self.weight * per_sample

        if sample_weights is not None:
            loss = (sample_weights * per_sample).sum() / sample_weights.sum()
        else:
            loss = per_sample.mean()
        return self.weight * loss

    def predict(self, raw: Tensor) -> Tensor:
        """MAP point prediction: the μ of the highest-π component, denormalised."""
        log_pi, mu, _sigma = self._components(raw)
        k_max = log_pi.argmax(dim=-1, keepdim=True)              # (N, 1)
        mu_map = mu.gather(-1, k_max).squeeze(-1)                # (N,)
        return _linear_denormalise(mu_map, self.norm_min, self.norm_max)

    def predict_components(self, raw: Tensor) -> dict[str, Tensor]:
        """Expose (pi, mu_phys, sigma_phys) for downstream diagnostics."""
        log_pi, mu, sigma = self._components(raw)
        pi = torch.exp(log_pi)
        mu_phys = _linear_denormalise(mu, self.norm_min, self.norm_max)
        half_range = 0.5 * (self.norm_max - self.norm_min)
        sigma_phys = sigma * half_range
        return {"pi": pi, "mu": mu_phys, "sigma": sigma_phys}


class BinnedDFLQuantileOffsetLoss(nn.Module):
    """YOLO/GFL-style binned classification + within-bin quantile offset.

    The model outputs ``K + Q`` channels per parameter:

    - ``K`` classification logits over discrete bins (softmax, trained with
      Distribution Focal Loss / arXiv:2006.04388).
    - ``Q`` within-bin offset channels in bin-width units, trained with
      quantile (pinball) loss, QFL-coupled (arXiv:2006.04388 §3) by the
      classification soft-probabilities so train and inference match.

    Three binning modes:

    - ``binning="linear"``: uniform bins over ``[range_min, range_max]``
      with optional 2 overflow bins (one each side) catching tracks
      outside the core range.  Used for d0 where CDF-warping would
      amplify the mode-collapse attractor.
    - ``binning="cdf"``: REMOVED in this release
      (``spline_config``).  Uniform in probability, bin count sets the
      resolution directly.
    - ``binning="cdf_eta"``: target physical θ is first converted to η
      via ``η = -ln(tan(θ/2))``, then CDF-warped.  Predictions return
      physical θ via the inverse.

    Point prediction = softmax-weighted expectation over bin centers +
    median-quantile offset (Q/2 index), scaled by bin width.

    Parameters
    ----------
    n_bins : int
        Total number of bins K (including overflow in linear mode).
    binning : str
        ``"linear"`` | ``"cdf"`` | ``"cdf_eta"``.
    range_min, range_max : float
        Linear mode: physical range covered by the ``K_core`` core bins
        (``K_core = K - 2`` when overflow is on, else ``K``).
    n_overflow : int
        Linear mode only: 0 (no overflow) or 2 (one overflow bin each
        side).
    overflow_pad : float | None
        Linear mode, ``n_overflow=2`` only: physical width of each
        overflow bin (defaults to the core bin width).
    spline_config : str | dict | None
        CDF modes only: path or inline dict for the PCHIP CDF spline.
    quantiles : list[float]
        Quantile levels for the within-bin offset head (default matches
        the validated 7-quantile pinball recipe).
    weight : float
        Per-parameter loss weight (matches the other sub-losses).
    lambda_offset : float
        Multiplier on the offset loss relative to the DFL classification
        loss.  Default 1.0 (equal weight).
    coupling : str
        ``"soft_qfl"`` (default): offset loss is softmax-weighted across
        all bins (Generalized Focal Loss).  ``"none"``: DFL only (cls
        head alone, no offset head trained).
    tail_weight_tau : float
        If > 0, apply a continuous per-track weight
        ``w = max(1, |target| / tail_weight_tau)`` (τ in physical units
        of the target) to the combined per-sample loss.  Restores
        gradient balance for heavy-tail targets: the ~5% of tail tracks
        were previously contributing ~5% of gradient, which lets the
        cls softmax drift toward the mode under the 95%/5% class
        imbalance (the Bayesian soft-collapse mechanism).  Matches
        :class:`MixtureDensityLoss`'s ``tail_weight_tau`` semantics
        exactly so YOLO/MDN ablations stay apples-to-apples.  For d0
        (95% within ±30 μm) set ``0.03``; for z0 set ``1.0``; leave at
        ``0`` (disabled) for φ / θ / q/p unless there's a specific
        tail-mass argument.
    class_balance_path : str | None
        If given, path to a (n_bins,) torch tensor of per-bin truth
        counts produced by ``scripts/compute_bin_histograms.py``.
        Per-bin CE weights are computed as
        ``w_bin ∝ 1 / hist[bin]^class_balance_power``, mean-normalised
        to 1, then clipped to ``class_balance_clip``.  Weights
        multiply each bin's CE log-probability contribution
        element-wise — i.e. implements standard DIR/LDS-style
        rebalancing (Yang et al., ICML 2021).  Stacks additively with
        ``tail_weight_tau``.  Mostly relevant for linear-binned
        parameters (d0); CDF-binned parameters already have ~uniform
        bin mass, so the rebalance is near-no-op for them.
    class_balance_power : float
        Exponent in ``w_bin ∝ 1 / hist^power``.  Default 0.5 = "sqrt
        rebalance" — the standard DIR/LDS choice.  0 disables;
        1.0 = full inverse frequency (more aggressive).
    class_balance_clip : tuple[float, float]
        Min/max clip on the normalised weights to prevent a
        near-empty bin from dominating the loss.  Default
        ``(0.1, 10.0)`` = 100× dynamic range.
    offset_memory_efficient : bool
        If True, compute the QFL-coupled offset loss by looping over the
        ``Q`` quantile channels instead of materialising the
        ``(N, K, Q)`` pinball tensor in one shot.  Cuts peak offset-loss
        memory by a factor ≈ Q (=7 for the default quantiles), at the
        cost of ``Q`` sequential GPU ops (≈10-15% forward slowdown on
        H100).  Recommended when K per parameter is large (>256) and/or
        batch size is pushed (large-batch DDP).  Mathematically
        identical to the vectorised branch.
    reduction : str
        ``"mean"`` (default) or ``"none"`` (per-sample, for batch-trim).
    monotone_eps : float
        Min gap between adjacent within-bin quantile predictions.
    """

    BINNING_MODES = ("linear", "cdf", "cdf_eta")

    def __init__(
        self,
        n_bins: int = 64,
        binning: str = "cdf",
        range_min: float = -1.0,
        range_max: float = 1.0,
        n_overflow: int = 0,
        overflow_pad: float | None = None,
        spline_config: str | dict[str, Any] | None = None,
        quantiles: list[float] | None = None,
        weight: float = 1.0,
        lambda_offset: float = 1.0,
        coupling: str = "soft_qfl",
        tail_weight_tau: float = 0.0,
        class_balance_path: str | None = None,
        class_balance_power: float = 0.5,
        class_balance_clip: tuple[float, float] = (0.1, 10.0),
        offset_memory_efficient: bool = False,
        reduction: str = "mean",
        monotone_eps: float = 1.0e-6,
        # When True, ``predict()`` returns the pure classification-posterior
        # expectation (softmax-weighted bin expectation, no within-bin
        # regression offset).  Training is unaffected.  Used for the d0-only
        # frozen-encoder fine-tune (Run E) where we want to read out the
        # classifier cleanly without the collapse-prone quantile offset.
        classification_only_predict: bool = False,
    ):
        super().__init__()
        if binning not in self.BINNING_MODES:
            raise ValueError(f"binning must be one of {self.BINNING_MODES}, got {binning!r}")
        if coupling not in ("soft_qfl", "none"):
            raise ValueError(f"coupling must be 'soft_qfl' or 'none', got {coupling!r}")
        if reduction not in ("mean", "none"):
            raise ValueError(f"reduction must be 'mean' or 'none', got {reduction!r}")
        if quantiles is None:
            quantiles = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
        _validate_quantiles(quantiles)
        if n_bins < 4:
            raise ValueError(f"n_bins must be >= 4, got {n_bins}")
        if binning == "linear":
            if n_overflow not in (0, 2):
                raise ValueError(f"n_overflow must be 0 or 2, got {n_overflow}")
            if range_max <= range_min:
                raise ValueError(f"range_max ({range_max}) must exceed range_min ({range_min})")
        else:
            if spline_config is None:
                raise ValueError(f"spline_config is required for binning={binning!r}")

        self.n_bins = int(n_bins)
        self.binning = binning
        self.n_overflow = int(n_overflow) if binning == "linear" else 0
        self.weight = float(weight)
        self.lambda_offset = float(lambda_offset)
        self.coupling = coupling
        self.tail_weight_tau = float(tail_weight_tau)
        self.offset_memory_efficient = bool(offset_memory_efficient)
        self.reduction = reduction
        self.monotone_eps = float(monotone_eps)
        self.classification_only_predict = bool(classification_only_predict)

        self.register_buffer("quantiles", torch.tensor(quantiles, dtype=torch.float32))

        # Precompute bin-center buffers.  We store:
        #   u_centers      — normalised bin centers in [0, 1] index space,
        #                    i.e. (k + 0.5) / K.  Used for DFL soft-target
        #                    computation and (in CDF modes) for the
        #                    softmax-weighted u expectation at predict time.
        #   phys_centers   — physical bin centers (linear mode only).
        #   bin_width_core — physical width of a core bin (linear mode only).
        #   range_{min,max} — physical linear-mode range.
        u_centers = (torch.arange(self.n_bins, dtype=torch.float32) + 0.5) / self.n_bins
        self.register_buffer("u_centers", u_centers)

        if binning == "linear":
            self.register_buffer("range_min", torch.tensor(float(range_min), dtype=torch.float32))
            self.register_buffer("range_max", torch.tensor(float(range_max), dtype=torch.float32))
            if self.n_overflow == 0:
                k_core = self.n_bins
                bin_width_core = (range_max - range_min) / k_core
                centers = range_min + (torch.arange(k_core, dtype=torch.float32) + 0.5) * bin_width_core
            else:  # n_overflow == 2
                k_core = self.n_bins - 2
                if k_core < 2:
                    raise ValueError("n_bins - 2 core bins must be >= 2 with overflow")
                bin_width_core = (range_max - range_min) / k_core
                pad = bin_width_core if overflow_pad is None else float(overflow_pad)
                centers = torch.zeros(self.n_bins, dtype=torch.float32)
                centers[0] = range_min - pad / 2.0       # low overflow
                centers[-1] = range_max + pad / 2.0      # high overflow
                for k in range(k_core):
                    centers[k + 1] = range_min + (k + 0.5) * bin_width_core
            self.register_buffer("phys_centers", centers)
            self.register_buffer("bin_width_core",
                                 torch.tensor(float(bin_width_core), dtype=torch.float32))
            self.spline = None
            self._k_core = k_core
        else:
            raise NotImplementedError(
                "BinnedDFLQuantileOffsetLoss with binning='cdf' has been removed "
                "from this release; use binning='range_split' instead."
            )

        # Per-bin class-rebalance weights.  If no histogram file is given we
        # store a uniform (K,) ones buffer so the forward path can multiply
        # unconditionally — zero-cost no-op when rebalancing is disabled.
        if class_balance_path is not None:
            hist_path = _resolve_spline_path(class_balance_path)
            if not hist_path.exists():
                raise FileNotFoundError(f"class_balance_path not found: {hist_path}")
            hist = torch.load(hist_path, map_location="cpu", weights_only=True)
            if not isinstance(hist, torch.Tensor):
                raise TypeError(
                    f"Expected a torch.Tensor at {hist_path}, got {type(hist).__name__}"
                )
            if hist.shape != (self.n_bins,):
                raise ValueError(
                    f"Histogram shape {tuple(hist.shape)} does not match n_bins={self.n_bins}"
                )
            if not (0.0 <= class_balance_clip[0] <= class_balance_clip[1]):
                raise ValueError(
                    f"class_balance_clip must be (low, high) with 0 <= low <= high, "
                    f"got {class_balance_clip}"
                )
            # Floor at 1 so empty bins become minimum-upweight, not inf.
            hist_f = hist.to(torch.float32).clamp_min(1.0)
            raw_w = 1.0 / hist_f.pow(float(class_balance_power))
            # Mean-normalise so the overall CE magnitude stays stable.
            raw_w = raw_w / raw_w.mean().clamp_min(1e-8)
            # Clip to bound dynamic range, then re-normalise.
            lo, hi = class_balance_clip
            clipped_w = raw_w.clamp(min=float(lo), max=float(hi))
            clipped_w = clipped_w / clipped_w.mean().clamp_min(1e-8)
            self.register_buffer("bin_class_weights", clipped_w)
        else:
            self.register_buffer(
                "bin_class_weights",
                torch.ones(self.n_bins, dtype=torch.float32),
            )

    # ---------------------------------------------------------------------
    # Properties / helpers
    # ---------------------------------------------------------------------

    @property
    def num_outputs(self) -> int:
        return self.n_bins + int(self.quantiles.numel())

    @property
    def n_quantiles(self) -> int:
        return int(self.quantiles.numel())

    def _target_to_u(self, target: Tensor) -> Tensor:
        """Map physical target to normalised bin-index position ``u ∈ [0, 1]``.

        ``u`` is defined so that bin ``k`` has center ``u_center_k = (k + 0.5) / K``,
        spanning ``u ∈ [k/K, (k+1)/K]``.  Values outside ``[0, 1]`` are
        clamped — linear+overflow mode routes out-of-core tracks to the
        center of the appropriate overflow bin.
        """
        if self.binning == "cdf":
            return self.spline.forward(target).clamp(0.0, 1.0)
        if self.binning == "cdf_eta":
            return self.spline.forward(_theta_to_eta(target)).clamp(0.0, 1.0)
        # Linear
        if self.n_overflow == 0:
            u = (target - self.range_min) / (self.range_max - self.range_min)
            return u.clamp(0.0, 1.0)
        # Linear + 2 overflow: core → u in [1/K, (K-1)/K]; out-of-range → overflow centers.
        k_core = self._k_core
        core_u = (target - self.range_min) / (self.range_max - self.range_min)  # [0, 1] for in-range
        u_mapped = (core_u * k_core + 1.0) / self.n_bins
        u_low = 0.5 / self.n_bins                                         # center of bin 0
        u_high = (self.n_bins - 0.5) / self.n_bins                        # center of bin K-1
        u_mapped = torch.where(target < self.range_min, torch.full_like(u_mapped, float(u_low)), u_mapped)
        u_mapped = torch.where(target > self.range_max, torch.full_like(u_mapped, float(u_high)), u_mapped)
        return u_mapped

    def _u_to_physical(self, u: Tensor) -> Tensor:
        """Inverse of _target_to_u.  Only used for the CDF modes at predict time."""
        u = u.clamp(0.0, 1.0)
        if self.binning == "cdf":
            return self.spline.inverse(u)
        if self.binning == "cdf_eta":
            return _eta_to_theta(self.spline.inverse(u))
        raise RuntimeError("Use phys_centers-based prediction for linear mode.")

    def _soft_target(self, u: Tensor) -> Tensor:
        """DFL triangular soft target over the K bins (N, K)."""
        # Continuous bin index: center of bin k is at k in this coord, so truth
        # at u_truth sits at position u_truth * K - 0.5 along integer bin axis.
        continuous = u * self.n_bins - 0.5
        # Clamp so that (bin_lo, bin_hi) stay in [0, K-1].
        continuous_c = continuous.clamp(0.0, float(self.n_bins - 1))
        bin_lo = continuous_c.floor().clamp(0, self.n_bins - 2).long()
        bin_hi = bin_lo + 1
        frac = (continuous_c - bin_lo.to(continuous_c.dtype)).clamp(0.0, 1.0)

        N = u.shape[0]
        soft = torch.zeros(N, self.n_bins, device=u.device, dtype=u.dtype)
        soft.scatter_add_(1, bin_lo.unsqueeze(-1), (1.0 - frac).unsqueeze(-1))
        soft.scatter_add_(1, bin_hi.unsqueeze(-1), frac.unsqueeze(-1))
        return soft

    def _ordered_offset(self, raw: Tensor) -> Tensor:
        """Ordered within-bin quantile-offset outputs (N, Q) via softplus+cumsum.

        Matches :class:`QuantileLoss` monotone construction so the offset
        head's quantile outputs are non-crossing by design.
        """
        if raw.shape[-1] < 2:
            return raw
        base = raw[..., :1]
        deltas = F.softplus(raw[..., 1:]) + self.monotone_eps
        return torch.cat([base, base + torch.cumsum(deltas, dim=-1)], dim=-1)

    # ---------------------------------------------------------------------
    # forward / predict
    # ---------------------------------------------------------------------

    def forward(self, pred: Tensor, target: Tensor, sample_weights: Tensor | None = None) -> Tensor:
        """Compute the combined DFL + QFL-coupled quantile-offset loss.

        pred shape: ``(N, K + Q)``.  target shape: ``(N,)`` in physical units.
        """
        K = self.n_bins
        logits = pred[..., :K]
        offset_raw = pred[..., K:]

        u = self._target_to_u(target)                 # (N,)
        soft_target = self._soft_target(u)            # (N, K)
        log_probs = F.log_softmax(logits, dim=-1)     # (N, K)
        # Per-bin class-rebalance weights (default uniform → no-op).  Weight
        # multiplies each bin's CE log-prob contribution, matching the
        # torch.nn.CrossEntropyLoss(class_weights=…) convention generalised
        # to DFL soft targets.
        l_cls = -(soft_target * self.bin_class_weights * log_probs).sum(dim=-1)  # (N,)

        if self.coupling == "soft_qfl":
            probs = F.softmax(logits, dim=-1)                             # (N, K)
            ordered = self._ordered_offset(offset_raw)                    # (N, Q)
            bin_idx = torch.arange(K, device=u.device, dtype=u.dtype)
            # Truth offset relative to each bin i's center, in bin-width units:
            #   off_to_i = (u*K - 0.5) - i
            off_truth = u.unsqueeze(-1) * K - 0.5 - bin_idx               # (N, K)

            if self.offset_memory_efficient:
                # Loop over quantiles to avoid materialising the (N, K, Q)
                # tensor.  Peak per-step memory drops from O(N·K·Q) to
                # O(N·K) at the cost of Q sequential matmul iterations.
                # Needed at large K (e.g. 1024 on theta) to keep per-rank
                # BS scaling intact.  Mathematically identical to the
                # vectorised branch (same reduction operator, same mean
                # over quantiles, same autograd).
                Q = int(self.quantiles.numel())
                pinball_sum = torch.zeros_like(off_truth)                 # (N, K)
                for q_idx in range(Q):
                    tau_q = self.quantiles[q_idx]
                    diff_q = off_truth - ordered[:, q_idx:q_idx + 1]      # (N, K)
                    pinball_sum = pinball_sum + torch.maximum(
                        tau_q * diff_q, (tau_q - 1.0) * diff_q
                    )
                pinball = pinball_sum / Q                                 # (N, K)
            else:
                # Pinball loss per (bin, quantile), mean over quantiles:
                #   diff_{n,i,q} = off_truth[n,i] - ordered[n,q]
                diff = off_truth.unsqueeze(-1) - ordered.unsqueeze(-2)    # (N, K, Q)
                tau = self.quantiles.view(1, 1, -1)
                pinball = torch.maximum(tau * diff, (tau - 1.0) * diff).mean(dim=-1)  # (N, K)

            l_off = (probs * pinball).sum(dim=-1)                         # (N,)
        else:
            l_off = torch.zeros_like(l_cls)

        per_sample = l_cls + self.lambda_offset * l_off

        # Continuous tail upweighting by |target| (in physical units).
        # Mirrors MixtureDensityLoss.forward so YOLO vs MDN d0-collapse
        # ablations share the same gradient-rebalancing mechanism.  For
        # cdf_eta binning ``target`` is physical θ; if you want the weight
        # to track η-space tails instead, disable here and attach a custom
        # sample_weights via the outer TrackParameterLoss.
        if self.tail_weight_tau > 0.0:
            tail_w = torch.clamp(target.abs() / self.tail_weight_tau, min=1.0)
            per_sample = tail_w * per_sample

        if self.reduction == "none":
            if sample_weights is not None:
                per_sample = sample_weights * per_sample
            return self.weight * per_sample

        if sample_weights is not None:
            loss = (sample_weights * per_sample).sum() / sample_weights.sum().clamp(min=1e-8)
        else:
            loss = per_sample.mean()
        return self.weight * loss

    def predict(self, raw: Tensor) -> Tensor:
        """Point prediction.

        Default: softmax-weighted bin expectation + median offset (i.e.
        classification + regression-offset composite).  When
        ``classification_only_predict=True`` at construction time, returns
        the pure classification-posterior expectation (no offset) — used
        for reading out a DFL head as a plain classifier.

        ``raw`` shape: ``(N, K + Q)``.  Returns physical ``(N,)``.
        """
        K = self.n_bins
        logits = raw[..., :K]
        probs = F.softmax(logits, dim=-1)                                # (N, K)

        if self.classification_only_predict:
            if self.binning == "linear":
                return (probs * self.phys_centers).sum(dim=-1)
            u_expected = (probs * self.u_centers).sum(dim=-1)            # (N,)
            return self._u_to_physical(u_expected.clamp(0.0, 1.0))

        offset_raw = raw[..., K:]
        ordered = self._ordered_offset(offset_raw)                       # (N, Q)
        median_idx = (self.quantiles - 0.5).abs().argmin()
        median_offset = ordered[..., median_idx]                         # (N,)

        if self.binning == "linear":
            phys_expected = (probs * self.phys_centers).sum(dim=-1)
            return phys_expected + median_offset * self.bin_width_core
        # CDF / CDF-eta
        u_expected = (probs * self.u_centers).sum(dim=-1)                # (N,)
        u_pred = (u_expected + median_offset / K).clamp(0.0, 1.0)
        return self._u_to_physical(u_pred)

    def predict_quantiles(self, raw: Tensor) -> Tensor:
        """Return per-quantile point predictions by composing bin expectation
        with each within-bin offset quantile.

        Shape: ``(N, Q)``.  Used for calibration diagnostics + residual plots.
        """
        K = self.n_bins
        logits = raw[..., :K]
        offset_raw = raw[..., K:]
        probs = F.softmax(logits, dim=-1)
        ordered = self._ordered_offset(offset_raw)                        # (N, Q)

        if self.binning == "linear":
            phys_expected = (probs * self.phys_centers).sum(dim=-1, keepdim=True)  # (N, 1)
            return phys_expected + ordered * self.bin_width_core          # (N, Q)

        u_expected = (probs * self.u_centers).sum(dim=-1, keepdim=True)   # (N, 1)
        u_q = (u_expected + ordered / K).clamp(0.0, 1.0)                  # (N, Q)
        if self.binning == "cdf":
            return self.spline.inverse(u_q)
        return _eta_to_theta(self.spline.inverse(u_q))

    def predict_components(self, raw: Tensor) -> dict[str, Tensor]:
        """Expose classification posterior + offsets for diagnostics.

        Returns a dict with:
          - ``probs``        (N, K): softmax classification posterior
          - ``bin_centers``  (K,): physical bin centers
          - ``offset_q``     (N, Q): ordered offset in *physical* units
          - ``cls_entropy``  (N,): per-sample Shannon entropy of the
                              posterior (debug signal for collapse —
                              low entropy == confident)
        """
        K = self.n_bins
        logits = raw[..., :K]
        offset_raw = raw[..., K:]
        probs = F.softmax(logits, dim=-1)
        ordered = self._ordered_offset(offset_raw)

        if self.binning == "linear":
            centers_phys = self.phys_centers.clone()
            offset_phys = ordered * self.bin_width_core
        else:
            if self.binning == "cdf":
                centers_phys = self.spline.inverse(self.u_centers)
            else:
                centers_phys = _eta_to_theta(self.spline.inverse(self.u_centers))
            # Typical physical bin width ≈ (range / K) in u-space → depends on local spline slope.
            # For diagnostics we just report offsets in bin-width units (raw, unitless).
            offset_phys = ordered

        entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=-1)
        return {
            "probs": probs,
            "bin_centers": centers_phys,
            "offset_q": offset_phys,
            "cls_entropy": entropy,
        }


class CosinePhiLoss(nn.Module):
    """Angular loss ``1 - cos(phi_pred - phi_true)`` via (sin, cos) outputs.

    The model outputs ``(sin_raw, cos_raw)``.  These are unit-normalised
    inside the forward (divided by ``sqrt(sin² + cos²)``) to prevent the
    unbounded-drift failure mode of direct-φ prediction.  The loss is::

        1 - cos(phi_pred - phi_true)
            = 1 - (sin_p · sin(phi_true) + cos_p · cos(phi_true))

    which is naturally wraparound-safe.  Recovery: ``φ = atan2(sin, cos)``.
    Uses ``num_outputs=2`` to match :class:`CircularPhiLoss` so the rest of
    the metric / slice machinery does not need to branch on loss type.

    Parameters
    ----------
    weight : float
        Loss weight.
    reduction : str
        ``"mean"`` (default) returns scalar; ``"none"`` returns the
        per-sample ``weight * (1 - cos(Δφ))`` tensor with ``sample_weights``
        multiplied in but not normalised.  Used by batch-trimming.
    """

    num_outputs: int = 2

    def __init__(self, weight: float = 1.0, reduction: str = "mean"):
        super().__init__()
        self.weight = weight
        if reduction not in ("mean", "none"):
            raise ValueError(f"reduction must be 'mean' or 'none', got {reduction!r}")
        self.reduction = reduction

    def forward(self, pred: Tensor, target: Tensor, sample_weights: Tensor | None = None) -> Tensor:
        """pred: (N, 2) with [sin, cos];  target: (N,) with phi in radians."""
        sin_p = pred[..., 0]
        cos_p = pred[..., 1]
        # Unit-normalise to the circle; small epsilon guards against zero norm.
        norm = torch.sqrt(sin_p * sin_p + cos_p * cos_p + 1e-8)
        sin_p = sin_p / norm
        cos_p = cos_p / norm
        # 1 - cos(pred - target) = 1 - (sin_p sin_t + cos_p cos_t)
        per_sample = 1.0 - (sin_p * torch.sin(target) + cos_p * torch.cos(target))

        if self.reduction == "none":
            if sample_weights is not None:
                per_sample = sample_weights * per_sample
            return self.weight * per_sample

        if sample_weights is not None:
            loss = (sample_weights * per_sample).sum() / sample_weights.sum()
        else:
            loss = per_sample.mean()
        return self.weight * loss

    def predict(self, raw: Tensor) -> Tensor:
        """Recover phi from (sin, cos) outputs."""
        return torch.atan2(raw[..., 0], raw[..., 1])


# ============================================================================
# Registry
# ============================================================================

LOSS_REGISTRY: dict[str, type] = {
    "smooth_l1": SmoothL1Loss,
    "quantile": QuantileLoss,
    "quantile_eta": EtaQuantileLoss,
    "circular": CircularPhiLoss,
    "gaussian": GaussianParameterLoss,
    "gaussian_eta": GaussianEtaLoss,
    "cosine_phi": CosinePhiLoss,
    "mixture_density": MixtureDensityLoss,
    "binned_dfl_quantile": BinnedDFLQuantileOffsetLoss,
    # "range_split_classification" registered below after the class is defined
}


# ============================================================================
# Range-split classification loss (router + inner-range cls + outer-range cls)
# ============================================================================


class RangeSplitClassificationLoss(nn.Module):
    """Three-head split classifier for heavy-tailed d0.

    Output layout per sample: ``(N, K_inner + K_outer + 2)``
        [ inner_logits (K_inner) | outer_logits (K_outer) | router_logits (2) ]

    Routing label (binary): ``|target| > split_mm``.  0 → inner (core);
    1 → outer (tail).

    Training losses (sum, all weighted):
      - router CE on every sample
      - inner head CE on samples with is_tail==0 only (mask 0 elsewhere)
      - outer head CE on samples with is_tail==1 only

    Inference ``predict(raw)``: argmax router → use that head's
    bin-expectation in physical units.

    Both inner and outer heads use **linear** binning — uniform in
    physical d0 over their respective ranges.  Inner: ``[-inner_half_mm,
    +inner_half_mm]`` with ``K_inner`` uniform bins.  Outer: two uniform
    grids, ``[-outer_half_mm, -inner_half_mm]`` and
    ``[+inner_half_mm, +outer_half_mm]``, each with ``K_outer // 2`` bins
    (so ``K_outer`` must be even).  Targets exactly on the boundary go to
    the outer head (strict inequality on is_tail).

    Parameters
    ----------
    k_inner, k_outer : int
        Number of linear bins for the inner and outer classifiers.
    inner_half_mm : float
        Half-width of the inner (core) range.
    outer_half_mm : float
        Half-width of the outer (tail) range.  Samples outside this get
        clamped to the edge bin during training.
    split_mm : float | None
        |target| threshold for router label.  Defaults to inner_half_mm.
    weight_router, weight_inner, weight_outer : float
        Per-head loss weights.
    inner_route_threshold : float
        Inference-time asymmetric routing.  Only commit to the INNER head
        when ``softmax(router)[inner] > inner_route_threshold``.  Anything
        below that defaults to the OUTER head.  ``0.5`` recovers plain
        argmax routing; raise (e.g. ``0.8``) to bias unconfident routes
        toward the wider outer head — makes the failure mode "clipped at
        the 30 um boundary" instead of "crushed to 0".  Can be tuned post-
        hoc on an existing checkpoint (it is only used in ``predict()``).
    reduction : str
        ``"mean"`` or ``"none"``.
    """

    def __init__(
        self,
        k_inner: int = 24,
        k_outer: int = 420,
        inner_half_mm: float = 0.030,
        outer_half_mm: float = 2.5,
        split_mm: float | None = None,
        weight_router: float = 1.0,
        weight_inner: float = 1.0,
        weight_outer: float = 1.0,
        weight: float = 1.0,  # outer scalar loss_module weight
        inner_route_threshold: float = 0.5,
        reduction: str = "mean",
    ):
        super().__init__()
        if k_outer % 2 != 0:
            raise ValueError(f"k_outer must be even (split into two equal sides); got {k_outer}")
        self.k_inner = int(k_inner)
        self.k_outer = int(k_outer)
        self.k_outer_side = self.k_outer // 2
        self.inner_half_mm = float(inner_half_mm)
        self.outer_half_mm = float(outer_half_mm)
        self.split_mm = float(split_mm) if split_mm is not None else self.inner_half_mm
        self.weight_router = float(weight_router)
        self.weight_inner = float(weight_inner)
        self.weight_outer = float(weight_outer)
        self.weight = float(weight)
        if not 0.0 <= float(inner_route_threshold) <= 1.0:
            raise ValueError(
                f"inner_route_threshold must be in [0,1]; got {inner_route_threshold}"
            )
        self.inner_route_threshold = float(inner_route_threshold)
        self.reduction = reduction

        # Precompute bin centers (physical mm) for each head.  Used in predict().
        inner_edges = torch.linspace(-self.inner_half_mm, self.inner_half_mm, self.k_inner + 1)
        inner_centers = 0.5 * (inner_edges[:-1] + inner_edges[1:])
        self.register_buffer("inner_centers", inner_centers)

        left_edges = torch.linspace(-self.outer_half_mm, -self.inner_half_mm, self.k_outer_side + 1)
        right_edges = torch.linspace(self.inner_half_mm, self.outer_half_mm, self.k_outer_side + 1)
        left_centers = 0.5 * (left_edges[:-1] + left_edges[1:])
        right_centers = 0.5 * (right_edges[:-1] + right_edges[1:])
        outer_centers = torch.cat([left_centers, right_centers])  # shape (K_outer,)
        self.register_buffer("outer_centers", outer_centers)

    @property
    def num_outputs(self) -> int:
        return self.k_inner + self.k_outer + 2

    def _assign_inner_bin(self, target: Tensor) -> Tensor:
        # Uniform bin index in [0, K_inner-1] over [-inner_half, inner_half].
        u = (target + self.inner_half_mm) / (2 * self.inner_half_mm)  # 0..1
        idx = (u * self.k_inner).long().clamp(0, self.k_inner - 1)
        return idx

    def _assign_outer_bin(self, target: Tensor) -> Tensor:
        # Target is outside [-inner_half, inner_half].  Map to [0, K_outer-1]:
        # left side first (K_outer_side bins), then right side.
        is_left = target < 0
        # Left: u in [0,1] over [-outer_half, -inner_half]
        u_left = (target + self.outer_half_mm) / (self.outer_half_mm - self.inner_half_mm)
        idx_left = (u_left * self.k_outer_side).long().clamp(0, self.k_outer_side - 1)
        # Right: u in [0,1] over [inner_half, outer_half]
        u_right = (target - self.inner_half_mm) / (self.outer_half_mm - self.inner_half_mm)
        idx_right = (u_right * self.k_outer_side).long().clamp(0, self.k_outer_side - 1) + self.k_outer_side
        return torch.where(is_left, idx_left, idx_right)

    def forward(self, pred: Tensor, target: Tensor, sample_weights: Tensor | None = None) -> Tensor:
        K_i, K_o = self.k_inner, self.k_outer
        inner_logits = pred[..., :K_i]
        outer_logits = pred[..., K_i:K_i + K_o]
        router_logits = pred[..., K_i + K_o:K_i + K_o + 2]

        is_tail = (target.abs() > self.split_mm).long()  # (N,)

        # Router CE — always on
        l_router = F.cross_entropy(router_logits, is_tail, reduction="none")

        # Inner CE — masked to core tracks
        inner_bin = self._assign_inner_bin(target.clamp(-self.inner_half_mm, self.inner_half_mm))
        l_inner_all = F.cross_entropy(inner_logits, inner_bin, reduction="none")
        l_inner = l_inner_all * (is_tail == 0).float()

        # Outer CE — masked to tail tracks.  For core tracks pass a dummy bin
        # (index 0) through the loss to keep shape consistent; the is_tail
        # mask zeros it out.
        outer_bin = torch.where(
            is_tail == 1,
            self._assign_outer_bin(target),
            torch.zeros_like(is_tail),
        )
        l_outer_all = F.cross_entropy(outer_logits, outer_bin, reduction="none")
        l_outer = l_outer_all * (is_tail == 1).float()

        per_sample = (
            self.weight_router * l_router
            + self.weight_inner * l_inner
            + self.weight_outer * l_outer
        )

        if sample_weights is not None:
            per_sample = per_sample * sample_weights

        if self.reduction == "mean":
            return per_sample.mean() * self.weight
        return per_sample * self.weight

    def predict(self, raw: Tensor) -> Tensor:
        """Asymmetric-threshold router → bin-expectation of the selected head.

        Uses ``softmax(router)[inner] > inner_route_threshold`` as the
        gate.  Default ``0.5`` recovers plain argmax routing.  Raising it
        (e.g. 0.8) biases unconfident routes toward the OUTER head — the
        outer head asked about a core track returns ~30 um (clipped at
        its edge bin, bounded error), whereas the inner head asked about
        a tail track returns ~0 (crushed to the mode, unbounded error).
        On a d0 measurement the former is almost always the preferable
        failure mode.
        """
        K_i, K_o = self.k_inner, self.k_outer
        inner_logits = raw[..., :K_i]
        outer_logits = raw[..., K_i:K_i + K_o]
        router_logits = raw[..., K_i + K_o:K_i + K_o + 2]

        inner_probs = F.softmax(inner_logits, dim=-1)
        outer_probs = F.softmax(outer_logits, dim=-1)
        inner_pred = (inner_probs * self.inner_centers).sum(dim=-1)
        outer_pred = (outer_probs * self.outer_centers).sum(dim=-1)

        router_probs = F.softmax(router_logits, dim=-1)  # (N, 2)
        p_inner = router_probs[..., 0]
        use_inner = p_inner > self.inner_route_threshold  # bool (N,)
        return torch.where(use_inner, inner_pred, outer_pred)


LOSS_REGISTRY["range_split_classification"] = RangeSplitClassificationLoss


# ============================================================================
# Composite loss over all five track parameters
# ============================================================================


class TrackParameterLoss(nn.Module):
    """Config-driven composite loss for the five perigee track parameters.

    Instantiated entirely from a ``losses`` dict in the YAML config::

        losses:
          d0:
            type: smooth_l1
            weight: 1.0
            norm_min: -2.0
            norm_max: 2.0
          phi:
            type: circular
            weight: 1.0
          ...

    The ``parameters`` ordering determines the slice of the model's output
    that each sub-loss reads.

    Parameters
    ----------
    config : dict[str, dict]
        Per-parameter loss config keyed by parameter name.
        Each value is a dict with at least ``type`` plus any kwargs
        for that loss class.  Optionally includes ``delta_anchor`` —
        a key in the targets dict whose value is subtracted from the
        target before computing the loss (for delta parameterization).
    parameter_order : list[str]
        Canonical ordering of the five parameters.
    loss_aggregation : str
        How to combine per-parameter losses: ``"sum"`` (default) or
        ``"geometric_mean"``.  Geometric mean is scale-invariant and
        automatically balances tasks without manual weight tuning
        (Chennupati et al., MultiNet++, 2019).
    """

    def __init__(
        self,
        config: dict[str, dict[str, Any]],
        parameter_order: list[str] | None = None,
        qop_tail_weight: float = 0.0,
        qop_scale: float = 2.0,
        loss_aggregation: str = "sum",
    ):
        super().__init__()

        if parameter_order is None:
            parameter_order = ["d0", "z0", "phi", "theta", "qop"]
        if loss_aggregation not in ("sum", "geometric_mean"):
            raise ValueError(f"loss_aggregation must be 'sum' or 'geometric_mean', got {loss_aggregation!r}")

        self.parameter_order = parameter_order
        self.loss_aggregation = loss_aggregation
        self.losses = nn.ModuleDict()
        self._delta_anchors: dict[str, str] = {}

        for name in parameter_order:
            assert name in config, f"Missing loss config for parameter '{name}'"
            cfg = dict(config[name])  # shallow copy to pop from
            loss_type = cfg.pop("type")
            # Pop delta_anchor before passing remaining kwargs to the sub-loss
            delta_anchor = cfg.pop("delta_anchor", None)
            if delta_anchor is not None:
                self._delta_anchors[name] = delta_anchor
            assert loss_type in LOSS_REGISTRY, (
                f"Unknown loss type '{loss_type}' for parameter '{name}'. "
                f"Available: {list(LOSS_REGISTRY.keys())}"
            )
            self.losses[name] = LOSS_REGISTRY[loss_type](**cfg)

        # Compute output slice boundaries
        self._output_slices: dict[str, tuple[int, int]] = {}
        offset = 0
        for name in parameter_order:
            n = self.losses[name].num_outputs
            self._output_slices[name] = (offset, offset + n)
            offset += n

        self._total_outputs = offset
        self.qop_tail_weight = qop_tail_weight
        self.qop_scale = qop_scale

    @property
    def total_outputs(self) -> int:
        """Total number of raw regression outputs the model must produce."""
        return self._total_outputs

    def get_output_slice(self, name: str) -> tuple[int, int]:
        """Return (start, end) indices into the model's output vector."""
        return self._output_slices[name]

    def forward(
        self,
        pred: Tensor,
        targets: dict[str, Tensor],
        valid_mask: Tensor | None = None,
        trim_mask: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Compute per-parameter losses.

        Parameters
        ----------
        pred : Tensor
            Model output of shape ``(B, total_outputs)`` or
            ``(N, total_outputs)`` after masking.
        targets : dict[str, Tensor]
            Per-parameter target tensors, each ``(B,)`` or ``(N,)``.
        valid_mask : Tensor | None
            Boolean mask ``(B,)`` selecting valid tracks.  If given,
            both ``pred`` and ``targets`` are masked before computing losses.
        trim_mask : Tensor | None
            Optional per-sample weight mask of shape ``(N_valid,)``
            (i.e. same size as ``valid_mask.sum()`` if a valid_mask is
            given, else same size as ``pred.shape[0]``).  Used by
            batch-trimming: a float tensor of 0/1 values where 0 drops
            the sample from the loss and 1 keeps it with its full
            weight.  The trim mask is multiplied into ``sample_weights``
            so that the weighted-mean normalisation denominator is
            ``(sample_weights * trim_mask).sum()`` — i.e. a proper mean
            over retained samples, not a scaled mean over all samples.

        Returns
        -------
        dict[str, Tensor]
            Per-parameter scalar loss values, plus a ``"total"`` entry.
        """
        losses: dict[str, Tensor] = {}
        device = pred.device

        # Per-track |qop| weighting — opt-in, off by default (qop_tail_weight=0)
        # Tracks with high |qop| (large curvature, significant scattering) get
        # proportionally more gradient signal.  Weights are normalised so the
        # mean weight is 1, preserving the overall loss scale.
        sample_weights: Tensor | None = None
        if self.qop_tail_weight > 0.0 and "qop" in targets:
            qop_vals = targets["qop"]
            if valid_mask is not None:
                qop_vals = qop_vals[valid_mask]
            w = 1.0 + self.qop_tail_weight * (qop_vals.abs() / self.qop_scale).clamp(max=1.0)
            sample_weights = w / w.mean()

        # Fold the trim_mask into sample_weights (creating one if needed).
        # The sub-losses use ``sum(sw * per) / sum(sw)`` so a hard 0/1 mask
        # acts as a dropout filter on the batch.
        if trim_mask is not None:
            if sample_weights is None:
                sample_weights = trim_mask
            else:
                sample_weights = sample_weights * trim_mask

        loss_values: list[Tensor] = []

        for name in self.parameter_order:
            start, end = self._output_slices[name]
            p = pred[..., start:end] if end - start > 1 else pred[..., start]
            t = targets[name]

            # Delta anchor: subtract anchor value from target so the model
            # predicts the residual (e.g. phi - innermost_hit_phi).
            anchor: Tensor | None = None
            if name in self._delta_anchors:
                anchor_key = self._delta_anchors[name]
                anchor = targets.get(anchor_key)

            if valid_mask is not None:
                p = p[valid_mask]
                t = t[valid_mask]
                if anchor is not None:
                    anchor = anchor[valid_mask]

            if anchor is not None:
                t = t - anchor
                # Wrap phi-like deltas to [-π, π]
                if name == "phi":
                    t = torch.remainder(t + math.pi, 2.0 * math.pi) - math.pi

            if t.numel() == 0:
                losses[name] = torch.tensor(0.0, device=device)
                continue

            loss = self.losses[name](p, t, sample_weights=sample_weights)
            losses[name] = loss
            loss_values.append(loss)

        # Aggregate per-parameter losses into total
        if self.loss_aggregation == "geometric_mean" and loss_values:
            log_losses = torch.stack([torch.log(lv.clamp(min=1e-8)) for lv in loss_values])
            losses["total"] = torch.exp(log_losses.mean())
        else:
            total = torch.tensor(0.0, device=device)
            for lv in loss_values:
                total = total + lv
            losses["total"] = total

        return losses

    def per_sample_total(
        self,
        pred: Tensor,
        targets: dict[str, Tensor],
        valid_mask: Tensor | None = None,
    ) -> Tensor:
        """Return per-sample weighted sum of the five parameter losses.

        Shape: ``(N_valid,)`` — one scalar per *kept* track, which is the
        quantity a batch-trimming rule should rank on.  The per-parameter
        loss weights from the YAML config are applied (so this matches the
        objective the optimizer sees), but the ``qop_tail_weight`` meta
        sample weights are **deliberately NOT** applied — those are an
        orthogonal curvature re-weighting and would bias the trimmer
        toward keeping high-|qop| tracks, which is the opposite of what
        we want.

        Each sub-loss is invoked in its ``reduction="none"`` path, which
        requires that every loss class used in the config supports it.
        The currently supported ones are ``gaussian``, ``gaussian_eta``,
        and ``cosine_phi``.  If a quantile loss is plugged in later a
        ``reduction`` argument will have to be added there too — we fail
        loudly with a clear error rather than silently returning a wrong
        aggregate.

        Parameters
        ----------
        pred : Tensor
            Raw model output ``(B, total_outputs)``.
        targets : dict[str, Tensor]
            Per-parameter targets, each ``(B,)``.
        valid_mask : Tensor | None
            Boolean mask selecting valid tracks.

        Returns
        -------
        Tensor
            Shape ``(N_valid,)`` — per-sample aggregate loss, detached
            from the autograd graph (safe to feed into ``kthvalue``).
        """
        device = pred.device

        # Determine N_valid for the zero-init accumulator.
        n_valid = (
            int(valid_mask.sum().item())
            if valid_mask is not None
            else int(pred.shape[0])
        )
        if n_valid == 0:
            return torch.zeros(0, device=device)

        aggregate = torch.zeros(n_valid, device=device)

        for name in self.parameter_order:
            sub_loss = self.losses[name]
            if not hasattr(sub_loss, "reduction"):
                raise RuntimeError(
                    f"per_sample_total: loss class {type(sub_loss).__name__} "
                    f"for parameter {name!r} does not support reduction='none'. "
                    "Batch trimming requires per-sample losses on every parameter."
                )

            start, end = self._output_slices[name]
            p = pred[..., start:end] if end - start > 1 else pred[..., start]
            t = targets[name]

            # Delta anchor — same transform as forward()
            anchor: Tensor | None = None
            if name in self._delta_anchors:
                anchor = targets.get(self._delta_anchors[name])

            if valid_mask is not None:
                p = p[valid_mask]
                t = t[valid_mask]
                if anchor is not None:
                    anchor = anchor[valid_mask]

            if anchor is not None:
                t = t - anchor
                if name == "phi":
                    t = torch.remainder(t + math.pi, 2.0 * math.pi) - math.pi

            if t.numel() == 0:
                continue

            prev = sub_loss.reduction
            sub_loss.reduction = "none"
            try:
                per_sample = sub_loss(p, t, sample_weights=None)
            finally:
                sub_loss.reduction = prev

            aggregate = aggregate + per_sample

        return aggregate.detach()

    def predict(
        self,
        pred: Tensor,
    ) -> dict[str, Tensor]:
        """Convert raw model outputs to physical predictions.

        Parameters
        ----------
        pred : Tensor
            Raw model output ``(B, total_outputs)`` or ``(N, total_outputs)``.

        Returns
        -------
        dict[str, Tensor]
            Per-parameter physical predictions, each ``(B,)`` or ``(N,)``.
        """
        preds: dict[str, Tensor] = {}
        for name in self.parameter_order:
            start, end = self._output_slices[name]
            raw = pred[..., start:end] if end - start > 1 else pred[..., start:start + 1]
            preds[name] = self.losses[name].predict(raw)
        return preds

    def predict_physical(
        self,
        pred: Tensor,
        targets: dict[str, Tensor] | None = None,
    ) -> dict[str, Tensor]:
        """Convert raw model outputs to physical predictions, adding back delta anchors.

        For parameters with ``delta_anchor`` configured, the prediction is
        in delta space.  This method adds back the anchor value to recover
        the full physical prediction.  Falls back to :meth:`predict` when
        no delta anchors are configured or ``targets`` is ``None``.

        Parameters
        ----------
        pred : Tensor
            Raw model output ``(B, total_outputs)``.
        targets : dict[str, Tensor] | None
            Target dict containing anchor values (e.g. ``innermost_phi``).
        """
        preds = self.predict(pred)
        if targets is not None and self._delta_anchors:
            for name, anchor_key in self._delta_anchors.items():
                if anchor_key in targets:
                    preds[name] = preds[name] + targets[anchor_key]
                    # Wrap phi to [-π, π]
                    if name == "phi":
                        preds[name] = torch.remainder(preds[name] + math.pi, 2.0 * math.pi) - math.pi
        return preds

    def predict_quantiles(
        self,
        pred: Tensor,
    ) -> dict[str, Tensor]:
        """Return full quantile predictions (for quantile-based losses only).

        For non-quantile parameters the single point prediction is returned.
        Gaussian losses (``GaussianParameterLoss`` / ``GaussianEtaLoss``)
        expose an analytic ``predict_quantiles`` from ``(mu, log_var)`` and are
        therefore included in the dispatch.
        """
        preds: dict[str, Tensor] = {}
        for name in self.parameter_order:
            start, end = self._output_slices[name]
            raw = pred[..., start:end]
            loss_fn = self.losses[name]
            if isinstance(
                loss_fn,
                (QuantileLoss, EtaQuantileLoss,
                 GaussianParameterLoss, BinnedDFLQuantileOffsetLoss),
            ):
                preds[name] = loss_fn.predict_quantiles(raw)
            else:
                preds[name] = loss_fn.predict(raw)
        return preds

    def quantile_crossing_metrics(
        self,
        pred: Tensor,
        valid_mask: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Return crossing diagnostics from raw quantile channels.

        Metrics are computed before monotone reconstruction to monitor how often
        the unconstrained model outputs violate quantile ordering.
        """
        metrics: dict[str, Tensor] = {}
        rates: list[Tensor] = []
        mean_gaps: list[Tensor] = []
        max_gaps: list[Tensor] = []

        for name in self.parameter_order:
            loss_fn = self.losses[name]
            if not isinstance(loss_fn, (QuantileLoss, EtaQuantileLoss)):
                continue

            start, end = self._output_slices[name]
            raw = pred[..., start:end]
            if valid_mask is not None:
                raw = raw[valid_mask]

            stats = loss_fn.raw_crossing_metrics(raw)
            metrics[f"{name}/raw_crossing_rate"] = stats["rate"]
            metrics[f"{name}/raw_crossing_mean_gap"] = stats["mean_gap"]
            metrics[f"{name}/raw_crossing_max_gap"] = stats["max_gap"]

            rates.append(stats["rate"])
            mean_gaps.append(stats["mean_gap"])
            max_gaps.append(stats["max_gap"])

        if rates:
            metrics["quantiles/raw_crossing_rate_mean"] = torch.stack(rates).mean()
            metrics["quantiles/raw_crossing_mean_gap_mean"] = torch.stack(mean_gaps).mean()
            metrics["quantiles/raw_crossing_max_gap_max"] = torch.stack(max_gaps).max()

        return metrics

    def quantile_calibration_metrics(
        self,
        pred: Tensor,
        targets: dict[str, Tensor],
        valid_mask: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Return quantile calibration diagnostics.

        For each quantile-based parameter, computes how well the empirical
        coverage matches the nominal quantile levels.
        """
        metrics: dict[str, Tensor] = {}
        cal_errors: list[Tensor] = []

        for name in self.parameter_order:
            loss_fn = self.losses[name]
            if not isinstance(loss_fn, (QuantileLoss, EtaQuantileLoss)):
                continue

            start, end = self._output_slices[name]
            raw = pred[..., start:end]
            t = targets[name]
            if valid_mask is not None:
                raw = raw[valid_mask]
                t = t[valid_mask]

            if t.numel() == 0:
                continue

            stats = loss_fn.calibration_metrics(raw, t)
            metrics[f"{name}/quantile_calibration_error"] = stats["calibration_error"]
            cal_errors.append(stats["calibration_error"])

        if cal_errors:
            metrics["quantiles/calibration_error_mean"] = torch.stack(cal_errors).mean()

        return metrics
