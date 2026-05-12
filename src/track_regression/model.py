"""Track parameter regression model using Bidirectional Mamba-2 or Transformer.

This module provides:

- :class:`TrackParameterRegressor` — the core ``nn.Module`` that embeds
  per-hit features (with Fourier encoding and min-max normalisation),
  encodes them with one of three backends (selected via ``pool``), and
  regresses the five perigee track parameters through a Dense head.

- :class:`TrackRegressionWrapper` — a ``LightningModule`` that wraps the
  regressor, configures the optimiser / scheduler, and handles the
  train / val / test loops with MAE, precision, IQR/σ and RMS metrics.

Pool / backbone selection
-------------------------
The ``pool`` argument determines how a per-track summary is obtained from
the encoder and which encoder API is used:

``ssm_state``
    :class:`BidirectionalMambaEncoder` — extracts the concatenated
    forward + backward final SSM hidden states (``2 * nheads * headdim *
    d_state``) from the last layer, projects each direction through its
    own ``state_head``, concatenates, and feeds ``output_head``.
``ssm_cls``
    :class:`BidirectionalMambaCLSEncoder` — appends learned CLS tokens to
    each scan direction, reads the final token's sequence output per
    direction, and concatenates to shape ``(B, 2 * dim)``.  Feeds
    ``output_head`` directly.
``register_token``
    Transformer :class:`EncoderWithCLS` (flash-attn2) with a single
    learned register token.  The register's final representation is
    read out at shape ``(B, dim)`` and feeds ``output_head`` directly.

Hybrid FP32/bf16 precision
--------------------------
The Lightning trainer runs in strict FP32 (``trainer.precision: 32-true``).
Mamba-2 CUDA kernels and flash-attn2 both require bf16/fp16, so the
``self.encoder(...)`` call is wrapped in a ``torch.amp.autocast`` block and
the encoder outputs are explicitly cast back to FP32.  Heads, loss, and
metrics therefore remain in full precision — the precision regime targeted
by this work.

Data flow
---------
1. Hits arrive as ``(B, N, D_in)`` with 12 features per hit:
   [x, y, z, r, phi_hit, theta_hit, s, volume_id, layer_id, surface_id, detector, eta_hit]
2. Min-max normalisation scales each feature to [0, 1].
3. Fourier encoding expands each feature into multi-scale sin/cos components.
4. :class:`hepattn.models.Dense` projects Fourier features to dimension ``dim``.
5. Encoder (one of the three above) produces sequence output and a pooled
   per-track summary tensor.
6. A regression head (Dense) maps the pooled summary to the target vector.
7. :class:`TrackParameterLoss` computes the config-driven composite loss.
"""

from __future__ import annotations

import math
from contextlib import nullcontext
from typing import Any, Literal

import torch
import torch.distributed as dist
from lion_pytorch import Lion
from lightning import LightningModule
from torch import Tensor, nn
from torch.optim import AdamW

from hepattn.models.dense import Dense
from hepattn.experiments.colliderml_regr.losses import MixtureDensityLoss, TrackParameterLoss
from hepattn.experiments.colliderml_regr.mamba_state import BidirectionalMambaEncoder
from hepattn.experiments.colliderml_regr.muon import MuonHybrid, split_params_for_muon


class _GradScale(torch.autograd.Function):
    """Identity in forward, scalar multiply in backward.

    Used to dampen the d0 branch's gradient contribution on the shared
    encoder's ``pooled`` output so the DFL cross-entropy does not dominate
    the shared trunk update (see ``TrackParameterRegressor.d0_grad_scale``).
    """

    @staticmethod
    def forward(ctx, x: Tensor, scale: float) -> Tensor:
        ctx.scale = float(scale)
        return x

    @staticmethod
    def backward(ctx, grad_out: Tensor):
        return grad_out * ctx.scale, None


# ============================================================================
# Fourier encoding
# ============================================================================


def fourier_encode(
    x: Tensor,
    fourier_scales: list[int] | None = None,
    fourier_base: int = 3,
) -> Tensor:
    """Encode input tensor with multi-scale Fourier features.

    For input of shape ``(*, D)``, produces ``(*, 2 * len(fourier_scales) * D)``
    by concatenating ``sin(x / base^n)`` and ``cos(x / base^n)`` for each scale.

    Parameters
    ----------
    x : Tensor
        Input features ``(*, D)``.
    fourier_scales : list[int]
        Exponent scales for the Fourier basis.
    fourier_base : int
        Base for the exponential frequency scaling.
    """
    if fourier_scales is None:
        fourier_scales = [-3, -2, -1, 0, 1, 2, 3]
    sin = [torch.sin(x / (fourier_base**n)) for n in fourier_scales]
    cos = [torch.cos(x / (fourier_base**n)) for n in fourier_scales]
    return torch.cat(sin + cos, dim=-1)



class TrackParameterRegressor(nn.Module):
    """Track parameter regressor supporting SSM, SSM-CLS and Transformer backbones.

    The ``pool`` argument selects which pooling strategy is used:

    - ``ssm_state`` (default): :class:`BidirectionalMambaEncoder` exposes
      two per-direction SSM hidden states.  Each is projected through its
      own ``state_head`` (``fwd_head`` / ``bwd_head``), concatenated, and
      fed through ``output_head``.
    - ``ssm_cls``: encoder appends learned CLS tokens to each scan
      direction and returns the concatenated pair of ``(cls_fwd, cls_bwd)``
      shape ``(B, 2 * dim)``.  Goes straight into ``output_head`` (no
      state_head).
    - ``register_token``: Transformer encoder with a learned register
      token; the register's final representation shape ``(B, dim)`` is
      fed straight into ``output_head`` (no state_head).

    Parameters
    ----------
    input_dim : int
        Number of raw per-hit features (before Fourier encoding).
    dim : int
        Internal model dimension (embedding size).
    encoder : nn.Module
        Pre-constructed encoder module.  Its ``forward`` is expected to
        return ``(sequence_output, pooled_summary)``.
    loss_module : TrackParameterLoss
        Pre-constructed composite loss module.
    pool : str
        One of ``"ssm_state"``, ``"ssm_cls"``, ``"register_token"``.

    state_head_output_dim : int
        Output dimensionality of each per-direction projection head.
        Only used when ``pool == "ssm_state"``.
    state_head_hidden_layers : int | list[int] | None
        Hidden layers inside each per-direction head (``None`` = linear).
    state_head_dropout : float
        Dropout in each per-direction head.
    state_head_activation : str
        Activation in each per-direction head.

    output_head_hidden_layers : int | list[int] | None
        Hidden layers in the final output head (``None`` = linear).
    output_head_dropout : float
        Dropout in the final output head.
    output_head_activation : str
        Activation in the final output head.

    input_net_hidden_layers : int | list[int] | None
        Hidden layer config for the input embedding network.
    input_net_dropout : float
        Dropout in the input embedding network.
    input_net_activation : str
        Activation function for the input embedding network.

    input_fields : list[str]
        Names of hit-level input features. Used for validation against
        ``input_dim``.
    fourier_scales : list[int] | None
        Fourier encoding scales.
    fourier_base : int
        Base for Fourier frequency scaling.
    norm_min / norm_max : list[float] | None
        Per-feature bounds for min-max normalisation.
    encoder_autocast_dtype : str
        Autocast dtype used for the encoder forward pass.  Defaults to
        ``"bfloat16"``, which is required by Mamba-2 Triton kernels and
        flash-attn2.  Set to ``"float32"`` to disable the autocast (only
        meaningful for the Transformer baseline on CPU).
    """

    @staticmethod
    def _resolve_activation(name: str) -> nn.Module | str:
        """Map an activation name string to a Module instance.

        ``'SwiGLU'`` is returned as a string so that :class:`Dense` can
        handle the gated linear unit bookkeeping internally.
        """
        _map: dict[str, nn.Module | str] = {
            "silu": nn.SiLU(),
            "gelu": nn.GELU(),
            "swiglu": "SwiGLU",
            "relu": nn.ReLU(),
            "mish": nn.Mish(),
        }
        key = name.lower()
        if key not in _map:
            raise ValueError(f"Unknown activation '{name}'. Choose from: {list(_map.keys())}")
        return _map[key]

    def __init__(
        self,
        input_dim: int,
        dim: int,
        encoder: nn.Module,
        loss_module: TrackParameterLoss,
        pool: Literal["ssm_state", "ssm_cls", "register_token"] = "ssm_state",
        # Per-direction state projection heads (only used when pool == "ssm_state")
        state_head_output_dim: int = 256,
        state_head_hidden_layers: int | list[int] | None = 0,
        state_head_dropout: float = 0.1,
        state_head_activation: str = "SiLU",
        # Final output head (after combining fwd + bwd projections)
        output_head_hidden_layers: int | list[int] | None = 0,
        output_head_dropout: float = 0.0,
        output_head_activation: str = "SiLU",
        # Input embedding network
        input_net_hidden_layers: int | list[int] | None = 0,
        input_net_dropout: float = 0.0,
        input_net_activation: str = "SiLU",
        # Data config
        input_fields: list[str] | None = None,
        sort_field: str = "s",
        fourier_scales: list[int] | None = None,
        fourier_base: int = 3,
        norm_min: list[float] | None = None,
        norm_max: list[float] | None = None,
        # Precision
        encoder_autocast_dtype: Literal["bfloat16", "float16", "float32"] = "bfloat16",
        # Output head initialisation scale — multiply the last Linear layer's
        # weights by this factor at init.  Keeps initial predictions near zero
        # regardless of hidden-state magnitude, which is critical for Gaussian
        # NLL (exp(-log_var) amplifies large initial outputs).  Default 1.0
        # is a no-op.
        output_head_init_scale: float = 1.0,
        # Decouple d0's projection + output head from the main regression
        # branch.  When True, d0 gets its own pool_head + output_head in
        # parallel to the regression heads, so DFL-style classification
        # gradients on d0 do not propagate through the shared pool_head.
        # The shared encoder still sees d0's gradient (by design — we want
        # d0 to shape the encoder features), but the shared projection to
        # the other four heads is untouched.  Requires d0 to be first in
        # loss_module.parameter_order.
        separate_d0_head: bool = False,
        # When separate_d0_head=True, multiply the gradient that flows from
        # the d0 branch back into the shared encoder's pooled output by this
        # scalar.  Forward pass is unaffected.  Set < 1.0 to dampen DFL's
        # dominance on the shared trunk (CLAUDE.md Open Issue #3: d0's
        # cross-entropy trunk gradient is empirically 24-150x the continuous
        # regression heads'); matches the "GradNorm-lite" fix option (b)
        # without the EMA bookkeeping.  Default 1.0 is a no-op.
        d0_grad_scale: float = 1.0,
        # Bypass the post-pool projection (`pool_head` and, when
        # ``separate_d0_head`` is also enabled, ``d0_pool_head``).  These
        # Dense projections were originally introduced to attenuate large
        # bf16 gradients flowing from the encoder into the regression heads
        # (see `_pool_head` history).  Under ``encoder_autocast_dtype: float32``
        # that justification is gone and the projection only adds capacity
        # loss between the SSM-CLS pooled summary and the output heads.  When
        # ``True``, both pool projections are replaced by ``nn.Identity()``
        # and the output heads consume the raw pooled vector ((B, 2*dim) for
        # ssm_cls, (B, dim) for register_token).  No effect for
        # pool='ssm_state' since that path already has no shared pool_head.
        disable_main_pool_head: bool = False,
    ):
        super().__init__()

        if pool not in ("ssm_state", "ssm_cls", "register_token"):
            raise ValueError(
                f"Unknown pool='{pool}'. Must be one of "
                "('ssm_state', 'ssm_cls', 'register_token')"
            )
        self.pool = pool

        self.input_dim = input_dim
        self.dim = dim
        self.input_fields = input_fields or []
        self.encoder_autocast_dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[encoder_autocast_dtype]

        # Fourier encoding config
        self.fourier_scales = fourier_scales if fourier_scales is not None else [-3, -2, -1, 0, 1, 2, 3]
        self.fourier_base = fourier_base
        fourier_dim = input_dim * 2 * len(self.fourier_scales)

        # Min-max normalisation buffers
        if norm_min is not None and norm_max is not None:
            self.register_buffer("norm_min", torch.tensor(norm_min, dtype=torch.float32))
            self.register_buffer("norm_max", torch.tensor(norm_max, dtype=torch.float32))
            self.use_norm = True
        else:
            self.use_norm = False

        # Validate input_fields matches input_dim if both are provided
        if self.input_fields and len(self.input_fields) != input_dim:
            raise ValueError(
                f"len(input_fields)={len(self.input_fields)} != input_dim={input_dim}"
            )

        # Hit embedding: fourier_dim → dim
        self.input_net = Dense(
            input_size=fourier_dim,
            output_size=dim,
            hidden_layers=input_net_hidden_layers,
            dropout=input_net_dropout,
            activation=self._resolve_activation(input_net_activation),
        )

        # Backbone encoder (SSM-state, SSM-CLS, or Transformer with register token)
        self.encoder = encoder

        # Loss module (holds sub-losses and knows output dimensionality)
        self.loss_module = loss_module

        # ---- Pool-dependent regression heads ----
        if pool == "ssm_state":
            # Two per-direction SSM states of dim (state_dim / 2), each
            # projected independently before being combined.
            if not hasattr(encoder, "state_dim"):
                raise ValueError(
                    "pool='ssm_state' requires the encoder to expose a "
                    "`state_dim` attribute (use BidirectionalMambaEncoder)."
                )
            self.per_dir_dim = encoder.state_dim // 2
            self.fwd_head = Dense(
                input_size=self.per_dir_dim,
                output_size=state_head_output_dim,
                hidden_layers=state_head_hidden_layers,
                dropout=state_head_dropout,
                activation=self._resolve_activation(state_head_activation),
            )
            self.bwd_head = Dense(
                input_size=self.per_dir_dim,
                output_size=state_head_output_dim,
                hidden_layers=state_head_hidden_layers,
                dropout=state_head_dropout,
                activation=self._resolve_activation(state_head_activation),
            )
            output_head_input_dim = 2 * state_head_output_dim
        elif pool == "ssm_cls":
            # Two learned CLS tokens concatenated → (B, 2 * dim).
            # Projection head provides a gradient bottleneck matching the
            # ssm_state path — without it, raw encoder-output gradients
            # enter the bf16 backward pass at ~7x higher magnitude.
            self.per_dir_dim = None
            self.fwd_head = None
            self.bwd_head = None
            if disable_main_pool_head:
                self.pool_head = nn.Identity()
                output_head_input_dim = 2 * dim
            else:
                self.pool_head = Dense(
                    input_size=2 * dim,
                    output_size=state_head_output_dim,
                    hidden_layers=state_head_hidden_layers,
                    dropout=state_head_dropout,
                    activation=self._resolve_activation(state_head_activation),
                )
                output_head_input_dim = state_head_output_dim
        elif pool == "register_token":
            # One or more learned register tokens.  The encoder optionally
            # exposes ``pool_dim`` = num_register_tokens * dim
            # (``EncoderWithCLS`` does so).  Fall back to ``dim`` for
            # encoders that don't expose it — preserves the original
            # 1-register behaviour for any legacy encoder.
            self.per_dir_dim = None
            self.fwd_head = None
            self.bwd_head = None
            pool_in_dim = int(getattr(encoder, "pool_dim", dim))
            if disable_main_pool_head:
                self.pool_head = nn.Identity()
                output_head_input_dim = pool_in_dim
            else:
                self.pool_head = Dense(
                    input_size=pool_in_dim,
                    output_size=state_head_output_dim,
                    hidden_layers=state_head_hidden_layers,
                    dropout=state_head_dropout,
                    activation=self._resolve_activation(state_head_activation),
                )
                output_head_input_dim = state_head_output_dim
        else:  # unreachable — guarded above
            raise ValueError(f"Unknown pool '{pool}'")

        # ---- Separate d0 branch (optional) --------------------------------
        # When enabled, d0's projection + final head live in a parallel
        # branch off the shared encoder so classification-style DFL
        # gradients never touch the shared pool_head / main output_head.
        self.separate_d0_head = bool(separate_d0_head)
        self.d0_grad_scale = float(d0_grad_scale)
        if self.separate_d0_head:
            if pool != "ssm_cls":
                raise NotImplementedError(
                    "separate_d0_head is currently implemented only for "
                    "pool='ssm_cls'.  Shape bookkeeping differs for the "
                    "other pools."
                )
            if loss_module.parameter_order[0] != "d0":
                raise ValueError(
                    "separate_d0_head requires 'd0' to be first in "
                    "loss_module.parameter_order; got "
                    f"{loss_module.parameter_order}"
                )
            d0_start, d0_end = loss_module.get_output_slice("d0")
            if d0_start != 0:
                raise ValueError(
                    "separate_d0_head requires d0 to occupy the leading "
                    f"output slice; got slice=({d0_start}, {d0_end})"
                )
            self._d0_output_dim = d0_end - d0_start
            self._reg_output_dim = loss_module.total_outputs - self._d0_output_dim
            # Parallel pool_head for d0 — same spec as the main pool_head.
            # When `disable_main_pool_head` is True, also bypass this one for
            # consistency: the d0 branch under fp32 training does not need
            # the bf16 gradient-attenuation projection either.
            if disable_main_pool_head:
                self.d0_pool_head = nn.Identity()
                d0_output_head_input_dim = 2 * dim
            else:
                self.d0_pool_head = Dense(
                    input_size=2 * dim,
                    output_size=state_head_output_dim,
                    hidden_layers=state_head_hidden_layers,
                    dropout=state_head_dropout,
                    activation=self._resolve_activation(state_head_activation),
                )
                d0_output_head_input_dim = state_head_output_dim
            # Parallel output head that emits only d0's slice.
            self.d0_output_head = Dense(
                input_size=d0_output_head_input_dim,
                output_size=self._d0_output_dim,
                hidden_layers=output_head_hidden_layers,
                dropout=output_head_dropout,
                activation=self._resolve_activation(output_head_activation),
            )
            main_output_size = self._reg_output_dim
        else:
            self._d0_output_dim = 0
            self._reg_output_dim = loss_module.total_outputs
            main_output_size = loss_module.total_outputs

        # Final output head — no final_activation (linear output for regression)
        self.output_head = Dense(
            input_size=output_head_input_dim,
            output_size=main_output_size,
            hidden_layers=output_head_hidden_layers,
            dropout=output_head_dropout,
            activation=self._resolve_activation(output_head_activation),
        )

        # Scale down the output head's last Linear layer so that initial
        # predictions are near zero regardless of hidden-state magnitude.
        # Critical for Gaussian NLL where large initial outputs push log_var
        # into saturated clamp regions with zero gradient.
        if output_head_init_scale != 1.0:
            for head in (
                self.output_head,
                getattr(self, "d0_output_head", None),
            ):
                if head is None:
                    continue
                last_layer = head.net[-1]
                assert isinstance(last_layer, nn.Linear), (
                    f"output_head_init_scale requires the last layer to be "
                    f"nn.Linear, got {type(last_layer).__name__}"
                )
                with torch.no_grad():
                    last_layer.weight.mul_(output_head_init_scale)
                    if last_layer.bias is not None:
                        last_layer.bias.zero_()

    def _normalise(self, x: Tensor) -> Tensor:
        """Min-max normalise features to [0, 1]."""
        if not self.use_norm:
            return x
        span = (self.norm_max - self.norm_min).clamp(min=1e-8)
        return (x - self.norm_min) / span

    def forward(
        self,
        inputs: dict[str, Tensor],
    ) -> dict[str, Tensor]:
        """Forward pass.

        Parameters
        ----------
        inputs : dict[str, Tensor]
            Must contain:
            - ``"hit_features"`` : ``(B, N, input_dim)``
            - ``"hit_s"`` : ``(B, N)`` distance-from-IP for ordering
            - ``"hit_valid"`` : ``(B, N)`` bool mask (for padded batches)
            Optionally:
            - ``"seq_idx"`` : ``(B, N)`` sequence indices for packed batches

        Returns
        -------
        dict[str, Tensor]
            ``"pred"`` — raw regression output ``(B, total_outputs)``
            ``"hidden_state"`` — pooled per-track summary ``(B, pool_dim)``
        """
        x = inputs["hit_features"]
        # Truth time is the encoder sort key (replaces s = sqrt(x²+y²+z²),
        # which underestimates on-helix arc length for forward tracks).
        # ``hit_s`` is still produced by the dataloader as input feature
        # column 6, but is no longer used for token ordering.
        sort_key = inputs["hit_time"]
        seq_idx = inputs.get("seq_idx")
        hit_valid = inputs.get("hit_valid")
        cu_seqlens = inputs.get("cu_seqlens")
        if cu_seqlens is not None and self.pool != "ssm_cls":
            raise ValueError(
                f"Packed batches (cu_seqlens given) are only supported with "
                f"pool='ssm_cls'; got pool='{self.pool}'."
            )

        # Min-max normalise
        x = self._normalise(x)

        # Fourier encode: (B, N, D) → (B, N, 2*n_scales*D)
        x = fourier_encode(x, self.fourier_scales, self.fourier_base)

        # Embed hits: (B, N, fourier_dim) → (B, N, dim)
        x = self.input_net(x)

        # Encoder forward pass in reduced precision.  Mamba-2 Triton kernels
        # and flash-attn2 both require bf16/fp16; everything outside this
        # context (heads, loss, metrics) stays in FP32 when the trainer is
        # configured with precision=32-true.
        use_autocast = (
            x.is_cuda and self.encoder_autocast_dtype != torch.float32
        )
        with torch.amp.autocast(
            device_type="cuda",
            dtype=self.encoder_autocast_dtype,
            enabled=use_autocast,
        ):
            if self.pool == "register_token":
                # Transformer encoder path — uses `kv_mask` for padding.
                _, pooled = self.encoder(x, x_sort_value=sort_key, kv_mask=hit_valid)
            elif self.pool == "ssm_cls" and cu_seqlens is not None:
                # Packed-batch SSM-CLS path. ``x_sort_value`` is intentionally
                # None: hits are pre-sorted within each segment by collate, and
                # a global argsort would mix segments.
                _, pooled = self.encoder(
                    x,
                    x_sort_value=None,
                    seq_idx=seq_idx,
                    cu_seqlens=cu_seqlens,
                )
            else:
                # Mamba-2 padded path — uses `seq_idx` (None for padded batches).
                _, pooled = self.encoder(x, x_sort_value=sort_key, seq_idx=seq_idx)

        # Cast pooled summary back to FP32 for the heads / loss / metrics.
        # (The per-hit sequence output is discarded — only used by the
        # encoder-internal DDP tie that keeps its `gate` / norm params in the
        # autograd graph.)
        pooled = pooled.to(torch.float32)

        if self.pool == "ssm_state":
            # Split into forward / backward SSM states, project independently.
            h_fwd = pooled[:, :self.per_dir_dim]
            h_bwd = pooled[:, self.per_dir_dim:]
            z_fwd = self.fwd_head(h_fwd)
            z_bwd = self.bwd_head(h_bwd)
            z = torch.cat([z_fwd, z_bwd], dim=-1)
        else:
            # ssm_cls / register_token — project through pool_head bottleneck.
            z = self.pool_head(pooled)

        # Final regression output (linear final activation).
        # Layout: [d0 slice | reg slice].  When separate_d0_head is enabled
        # d0 is produced by its own parallel branch off `pooled`, so the
        # DFL gradient on d0 bypasses the shared pool_head + output_head.
        pred_reg = self.output_head(z)
        if self.separate_d0_head:
            if self.d0_grad_scale != 1.0:
                pooled_for_d0 = _GradScale.apply(pooled, self.d0_grad_scale)
            else:
                pooled_for_d0 = pooled
            z_d0 = self.d0_pool_head(pooled_for_d0)
            pred_d0 = self.d0_output_head(z_d0)
            pred = torch.cat([pred_d0, pred_reg], dim=-1)
        else:
            pred = pred_reg

        return {"pred": pred, "hidden_state": pooled}

    def predict(self, outputs: dict[str, Tensor]) -> dict[str, Tensor]:
        """Convert raw outputs to physical predictions."""
        return self.loss_module.predict(outputs["pred"])

    def compute_loss(
        self,
        outputs: dict[str, Tensor],
        targets: dict[str, Tensor],
        valid_mask: Tensor | None = None,
        trim_mask: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Compute all parameter losses.

        Parameters
        ----------
        outputs : dict
            From ``forward()``.
        targets : dict[str, Tensor]
            Must contain ``d0``, ``z0``, ``phi``, ``theta``, ``qop`` tensors.
        valid_mask : Tensor | None
            ``(B,)`` bool mask for valid tracks.
        trim_mask : Tensor | None
            Optional per-sample 0/1 weight mask passed through to
            :meth:`TrackParameterLoss.forward` — used by the batch-trimming
            path in the LightningModule to drop the worst-residual samples
            from the backward pass.  ``None`` (the default) means no
            trimming.
        """
        return self.loss_module(
            outputs["pred"], targets, valid_mask=valid_mask, trim_mask=trim_mask
        )


# ============================================================================
# Twin-encoder d0 classifier (separate encoders for core / tail + router)
# ============================================================================


class TwinEncoderD0Classifier(nn.Module):
    """Two encoders × two range-specialised classifier heads + a router.

    Architecture
    ------------
    input_net (shared)
      │
      ├─► encoder_inner → pooled_inner (B, 2·dim) ─► pool_head_inner ─► inner_output_head (K_inner classes)
      │
      ├─► encoder_outer → pooled_outer (B, 2·dim) ─► pool_head_outer ─► outer_output_head (K_outer classes)
      │
      └─► router_head( cat(pooled_inner.detach(), pooled_outer.detach()) ) ─► (B, 2) binary logits

    Output layout of ``pred`` matches
    :class:`RangeSplitClassificationLoss`: ``[K_inner | K_outer | 2]`` so the
    same loss handles mask-based teacher forcing + argmax routing at
    inference.  Stop-grad on the router input means DFL-classifier gradients
    never reach either encoder through the router path — the router is
    trained purely by its own supervised binary CE on ``|truth_d0| > split``.

    Parameters
    ----------
    encoder_inner, encoder_outer : nn.Module
        Two ``BidirectionalMambaCLSEncoder`` instances; each must have
        ``pool == 'ssm_cls'`` semantics (pool returns ``(B, 2·dim)``).
    loss_module : TrackParameterLoss
        Must have exactly one parameter ``d0`` configured with
        ``type: range_split_classification``.  The k_inner / k_outer there
        determine the head widths here.
    """

    def __init__(
        self,
        input_dim: int,
        dim: int,
        encoder_inner: nn.Module,
        encoder_outer: nn.Module,
        loss_module: TrackParameterLoss,
        # Input embedding
        input_net_hidden_layers: int | list[int] | None = 0,
        input_net_dropout: float = 0.0,
        input_net_activation: str = "SiLU",
        # Per-encoder pool/head sizing
        pool_head_output_dim: int = 128,
        pool_head_hidden_layers: int | list[int] | None = 0,
        pool_head_dropout: float = 0.0,
        pool_head_activation: str = "SiLU",
        output_head_hidden_layers: int | list[int] | None = 0,
        output_head_dropout: float = 0.0,
        output_head_activation: str = "SiLU",
        output_head_init_scale: float = 1.0,
        # Router sizing
        router_hidden_layers: int | list[int] | None = None,
        router_dropout: float = 0.0,
        # Data / preprocessing
        input_fields: list[str] | None = None,
        sort_field: str = "s",
        fourier_scales: list[int] | None = None,
        fourier_base: int = 3,
        norm_min: list[float] | None = None,
        norm_max: list[float] | None = None,
        encoder_autocast_dtype: Literal["bfloat16", "float16", "float32"] = "bfloat16",
    ):
        super().__init__()

        self.input_dim = input_dim
        self.dim = dim
        self.input_fields = input_fields or []
        self.encoder_autocast_dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[encoder_autocast_dtype]

        self.fourier_scales = fourier_scales if fourier_scales is not None else [-3, -2, -1, 0, 1, 2, 3]
        self.fourier_base = fourier_base
        fourier_dim = input_dim * 2 * len(self.fourier_scales)

        if norm_min is not None and norm_max is not None:
            self.register_buffer("norm_min", torch.tensor(norm_min, dtype=torch.float32))
            self.register_buffer("norm_max", torch.tensor(norm_max, dtype=torch.float32))
            self.use_norm = True
        else:
            self.use_norm = False

        # Shared input embedding.  Both encoders see the same per-hit
        # representation — they differentiate via their own weights.
        self.input_net = Dense(
            input_size=fourier_dim,
            output_size=dim,
            hidden_layers=input_net_hidden_layers,
            dropout=input_net_dropout,
            activation=self._resolve_activation(input_net_activation),
        )

        self.encoder_inner = encoder_inner
        self.encoder_outer = encoder_outer
        self.loss_module = loss_module

        # Validate the loss is the range-split classifier and extract K_inner / K_outer
        from hepattn.experiments.colliderml_regr.losses import RangeSplitClassificationLoss
        if loss_module.parameter_order != ["d0"]:
            raise ValueError(
                f"TwinEncoderD0Classifier requires parameter_order=['d0']; "
                f"got {loss_module.parameter_order}"
            )
        d0_loss = loss_module.losses["d0"]
        if not isinstance(d0_loss, RangeSplitClassificationLoss):
            raise TypeError(
                "TwinEncoderD0Classifier requires the d0 loss to be "
                f"RangeSplitClassificationLoss; got {type(d0_loss).__name__}"
            )
        self.k_inner = d0_loss.k_inner
        self.k_outer = d0_loss.k_outer

        # Per-encoder pool heads (reduce 2·dim → pool_head_output_dim).
        self.pool_head_inner = Dense(
            input_size=2 * dim,
            output_size=pool_head_output_dim,
            hidden_layers=pool_head_hidden_layers,
            dropout=pool_head_dropout,
            activation=self._resolve_activation(pool_head_activation),
        )
        self.pool_head_outer = Dense(
            input_size=2 * dim,
            output_size=pool_head_output_dim,
            hidden_layers=pool_head_hidden_layers,
            dropout=pool_head_dropout,
            activation=self._resolve_activation(pool_head_activation),
        )

        # Final classification heads per encoder.
        self.output_head_inner = Dense(
            input_size=pool_head_output_dim,
            output_size=self.k_inner,
            hidden_layers=output_head_hidden_layers,
            dropout=output_head_dropout,
            activation=self._resolve_activation(output_head_activation),
        )
        self.output_head_outer = Dense(
            input_size=pool_head_output_dim,
            output_size=self.k_outer,
            hidden_layers=output_head_hidden_layers,
            dropout=output_head_dropout,
            activation=self._resolve_activation(output_head_activation),
        )

        # Router head — reads concat of the TWO encoders' pooled outputs with
        # stop-grad applied (see forward), so gradients don't propagate back.
        router_hidden = router_hidden_layers if router_hidden_layers is not None else [128]
        self.router_head = Dense(
            input_size=4 * dim,  # cat of pooled_inner (2·dim) + pooled_outer (2·dim)
            output_size=2,
            hidden_layers=router_hidden,
            dropout=router_dropout,
            activation=self._resolve_activation(output_head_activation),
        )

        # Scale down final classification weights for near-zero initial logits.
        if output_head_init_scale != 1.0:
            for head in (self.output_head_inner, self.output_head_outer, self.router_head):
                last = head.net[-1]
                assert isinstance(last, nn.Linear)
                with torch.no_grad():
                    last.weight.mul_(output_head_init_scale)
                    if last.bias is not None:
                        last.bias.zero_()

    # Reuse the class activation resolver from TrackParameterRegressor so
    # the activation map (silu/gelu/swiglu/...) is single-source-of-truth.
    _resolve_activation = staticmethod(TrackParameterRegressor._resolve_activation)

    def _normalise(self, x: Tensor) -> Tensor:
        if not self.use_norm:
            return x
        span = (self.norm_max - self.norm_min).clamp(min=1e-8)
        return (x - self.norm_min) / span

    def forward(self, inputs: dict[str, Tensor]) -> dict[str, Tensor]:
        x = inputs["hit_features"]
        sort_key = inputs["hit_time"]
        seq_idx = inputs.get("seq_idx")

        x = self._normalise(x)
        x = fourier_encode(x, self.fourier_scales, self.fourier_base)
        x = self.input_net(x)

        use_autocast = self.encoder_autocast_dtype != torch.float32
        with torch.amp.autocast(
            device_type="cuda",
            dtype=self.encoder_autocast_dtype,
            enabled=use_autocast,
        ):
            _, pooled_inner = self.encoder_inner(x, x_sort_value=sort_key, seq_idx=seq_idx)
            _, pooled_outer = self.encoder_outer(x, x_sort_value=sort_key, seq_idx=seq_idx)

        # Cast pooled summaries back to fp32 for the heads.
        pooled_inner = pooled_inner.to(torch.float32)
        pooled_outer = pooled_outer.to(torch.float32)

        # Inner head (core, K_inner bins) — only the inner encoder feeds it.
        z_inner = self.pool_head_inner(pooled_inner)
        inner_logits = self.output_head_inner(z_inner)

        # Outer head (tail, K_outer bins) — only the outer encoder feeds it.
        z_outer = self.pool_head_outer(pooled_outer)
        outer_logits = self.output_head_outer(z_outer)

        # Router — sees both pooled summaries but with stop-grad so its
        # gradient does not flow into either encoder.
        router_in = torch.cat([pooled_inner.detach(), pooled_outer.detach()], dim=-1)
        router_logits = self.router_head(router_in)

        pred = torch.cat([inner_logits, outer_logits, router_logits], dim=-1)
        return {"pred": pred, "hidden_state": (pooled_inner, pooled_outer)}

    def predict(self, outputs: dict[str, Tensor]) -> dict[str, Tensor]:
        return self.loss_module.predict(outputs["pred"])

    def compute_loss(
        self,
        outputs: dict[str, Tensor],
        targets: dict[str, Tensor],
        valid_mask: Tensor | None = None,
        trim_mask: Tensor | None = None,
    ) -> dict[str, Tensor]:
        return self.loss_module(
            outputs["pred"], targets, valid_mask=valid_mask, trim_mask=trim_mask
        )


# ============================================================================
# LightningModule wrapper
# ============================================================================


class TrackRegressionWrapper(LightningModule):
    """Lightning wrapper for track parameter regression training.

    Parameters
    ----------
    model : TrackParameterRegressor
        The regression model.
    lrs_config : dict
        Learning-rate scheduler configuration with keys:
        ``initial``, ``max``, ``end``, ``pct_start``, ``weight_decay``,
        ``skip_scheduler``.
    optimizer : str
        ``"AdamW"`` or ``"Lion"``.
    """

    def __init__(
        self,
        model: nn.Module,  # TrackParameterRegressor or TwinEncoderD0Classifier (any module exposing forward / predict / compute_loss / loss_module)
        lrs_config: dict[str, Any],
        optimizer: Literal["AdamW", "Lion", "MuonHybrid"] = "AdamW",
        name: str = "TrackRegression",
        pretrained_ckpt_path: str | None = None,
        pretrained_ckpt_strict: bool = True,
        gradient_clip_val: float = 1.0,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.name = name
        self.model = model
        self.lrs_config = lrs_config
        self.opt_name = optimizer
        if "betas" in lrs_config:
            self.opt_betas = tuple(lrs_config["betas"])
        else:
            # Sensible default per-optimizer: AdamW uses (0.9, 0.999), Lion
            # uses (0.9, 0.99).  configure_optimizers() only forwards this to
            # the optimizer if the user actually set ``betas``.
            self.opt_betas = (0.9, 0.999)

        self.gradient_clip_val = float(gradient_clip_val)

        # Load model weights only (no optimizer/scheduler state) for fine-tuning.
        # Use this instead of --ckpt_path when you want a fresh optimizer and LR schedule.
        if pretrained_ckpt_path is not None:
            # Expand env vars (e.g. ${REPRO_ROOT}) and resolve relative paths
            # against the package root so the same YAML is portable across
            # working directories.
            import os as _os
            from pathlib import Path as _Path
            _resolved = _Path(_os.path.expandvars(_os.path.expanduser(str(pretrained_ckpt_path))))
            if not _resolved.is_absolute():
                _pkg_root = _Path(__file__).resolve().parents[4]
                _resolved = _pkg_root / _resolved
            pretrained_ckpt_path = str(_resolved)
            ckpt = torch.load(pretrained_ckpt_path, map_location="cpu", weights_only=False)
            # Lightning checkpoints prefix all keys with "model." (from self.model = model)
            state = {k[len("model."):]: v for k, v in ckpt["state_dict"].items() if k.startswith("model.")}
            # When warm-restarting into a model with different head shapes
            # (e.g. swapping the loss family or enabling separate_d0_head),
            # drop tensors whose shapes no longer match so the encoder can
            # still load.  Heads reinit from scratch.
            own_state = self.model.state_dict()
            dropped: list[str] = []
            sliced: list[str] = []
            filtered: dict[str, Tensor] = {}
            # When separate_d0_head=True, the new shared `output_head` emits
            # only the non-d0 parameters (z0/phi/theta/qop), i.e. it has
            # `_d0_output_dim` fewer output rows than the checkpoint's
            # output_head (which was trained with d0 in slot 0).  Instead of
            # dropping the mismatched weight/bias — which silently re-inits
            # z0/phi/theta/qop from scratch — slice the leading d0 rows off
            # and keep the rest.  This preserves the frozen-head invariant
            # that z0/phi/theta/qop outputs match the baseline checkpoint
            # byte-for-byte.
            separate_d0 = bool(getattr(self.model, "separate_d0_head", False))
            for k, v in state.items():
                if k not in own_state:
                    filtered[k] = v
                    continue
                own_shape = own_state[k].shape
                if own_shape == v.shape:
                    filtered[k] = v
                    continue
                # Shape mismatch — try the d0-slice rescue for the final
                # output_head layer only.  The checkpoint's output_head has
                # d0 occupying the leading slot; the new shared output_head
                # drops it, so its row count is smaller by exactly the old
                # d0 slot width.  All other dims must match identically.
                if (
                    separate_d0
                    and k.startswith("output_head.net.")
                    and k.endswith((".weight", ".bias"))
                    and v.shape[1:] == own_shape[1:]
                    and v.shape[0] > own_shape[0]
                ):
                    d0_slot = v.shape[0] - own_shape[0]
                    filtered[k] = v[d0_slot:].clone()
                    sliced.append(
                        f"{k} ({tuple(v.shape)} → {tuple(own_shape)}, "
                        f"dropped leading {d0_slot} d0 rows)"
                    )
                    continue
                dropped.append(f"{k} ({tuple(v.shape)} → {tuple(own_shape)})")
            missing, unexpected = self.model.load_state_dict(
                filtered, strict=pretrained_ckpt_strict
            )
            if pretrained_ckpt_strict and (missing or unexpected or dropped):
                raise RuntimeError(
                    f"Checkpoint weight mismatch.\n"
                    f"Missing: {missing}\nUnexpected: {unexpected}\n"
                    f"Shape-dropped: {dropped}"
                )
            if sliced:
                print(f"[fine-tune] Warm restart sliced {len(sliced)} "
                      f"output_head tensors to drop d0 leading rows: {sliced}")
            if dropped:
                print(f"[fine-tune] Warm restart dropped {len(dropped)} "
                      f"tensors with shape mismatch: {dropped}")
            if missing:
                print(f"[fine-tune] Missing keys re-initialised: {missing}")
            if unexpected:
                print(f"[fine-tune] Unexpected keys ignored: {unexpected}")
            print(f"[fine-tune] Loaded model weights from {pretrained_ckpt_path}")

    def setup(self, stage: str) -> None:
        """Log a one-shot summary of architecture/parameter/precision state.

        Prints are gated on ``trainer.is_global_zero`` to avoid duplication
        under DDP.  Only fires for ``fit`` stage (not sanity-check/validate).
        """
        if stage != "fit":
            return

        if not getattr(self.trainer, "is_global_zero", True):
            return

        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        pool = getattr(self.model, "pool", "unknown")
        autocast_dtype = getattr(self.model, "encoder_autocast_dtype", None)
        try:
            precision = self.trainer.precision
        except RuntimeError:
            precision = "<not attached>"

        print("-" * 80)
        print(f"[{self.name}] Pool type:             {pool}")
        print(f"[{self.name}] Trainer precision:     {precision}")
        print(f"[{self.name}] Encoder autocast:      {autocast_dtype}")
        print(f"[{self.name}] Total parameters:      {total / 1e6:.3f} M")
        print(f"[{self.name}] Trainable parameters:  {trainable / 1e6:.3f} M")
        print("-" * 80)

    # -- forward / predict --------------------------------------------------

    def forward(self, inputs: dict[str, Tensor]) -> dict[str, Tensor]:
        return self.model(inputs)

    def predict_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self.model(inputs)
        preds = self.model.loss_module.predict_physical(outputs["pred"], targets)
        return preds, targets

    # -- step helpers -------------------------------------------------------

    def _shared_step(
        self,
        batch: tuple[dict[str, Tensor], dict[str, Tensor]],
        stage: str,
    ) -> Tensor:
        inputs, targets = batch
        outputs = self.model(inputs)

        valid_mask = targets.get("track_valid")
        losses = self.model.compute_loss(outputs, targets, valid_mask=valid_mask)
        crossing_metrics = self.model.loss_module.quantile_crossing_metrics(outputs["pred"], valid_mask=valid_mask)
        calibration_metrics = self.model.loss_module.quantile_calibration_metrics(outputs["pred"], targets, valid_mask=valid_mask)

        # Only use sync_dist for val/test — training metrics are local per-rank
        # averages.  Using sync_dist=True during training adds many NCCL
        # allreduces per step which can cause DDP synchronisation issues.
        do_sync = stage != "train"

        # Log every component
        for name, value in losses.items():
            self.log(f"{stage}/{name}", value, sync_dist=do_sync, prog_bar=(name == "total"))

        # Monitor quantile crossings on raw (unconstrained) quantile channels
        for name, value in crossing_metrics.items():
            self.log(f"{stage}/{name}", value, sync_dist=do_sync)

        # Monitor quantile calibration (empirical coverage vs nominal levels)
        for name, value in calibration_metrics.items():
            self.log(f"{stage}/{name}", value, sync_dist=do_sync)

        # MDN component diagnostics (π_k, μ_k, σ_k per mixture component)
        self._log_mdn_components(outputs["pred"], valid_mask, stage)

        # Compute and log per-parameter metrics for all stages.
        # Use predict_physical to add back delta anchors for correct residuals.
        preds = self.model.loss_module.predict_physical(outputs["pred"], targets)
        self._log_metrics(preds, targets, valid_mask, stage)

        return losses["total"]

    # Units and scale factors for precision logging
    _PRECISION_UNITS: dict[str, tuple[str, float]] = {
        "d0": ("[mm]", 1.0),
        "z0": ("[mm]", 1.0),
        "phi": ("[mrad]", 1000.0),
        "theta": ("[mrad]", 1000.0),
        "qop": ("[1/GeV]", 1.0),
    }

    def _log_metrics(
        self,
        preds: dict[str, Tensor],
        targets: dict[str, Tensor],
        valid_mask: Tensor | None,
        stage: str,
    ) -> None:
        """Log per-parameter metrics: MAE, precision, and SSM precision on tight DM subset.

        Metrics
        -------
        - ``{stage}/{name}/mae``: mean absolute error (all stages)
        - ``{stage}/{name}/precision {unit}``: std of (pred - truth) residuals,
          scaled to physical units (all stages)
        - ``{stage}/{name}/ssm_precision_dm {unit}``: SSM precision on
          the double-matched subset — val/test only (needs ACTS DM data).
        - ``{stage}/{name}/ssm_iqr_dm {unit}``: robust σ estimator
          ``(Q75 - Q25) / 1.349`` on the double-matched subset — val/test.
        - ``{stage}/{name}/ssm_rms_dm {unit}``: RMS of the residuals on the
          double-matched subset — val/test.

        ``_dm`` metrics filter to tracks that are **both** ``valid`` AND
        ``acts_dm_mask`` (ACTS Kalman Filter purity > 0.75 AND efficiency
        > 0.75).  The residual is always ``p - t`` against the truth
        target ``t`` — ACTS reco values are only used as the presence
        guard.  Any change here must preserve this double-matched filter
        semantics so that the three DM metrics compare apples to apples.

        ACTS precision is never logged (precomputed / static).
        """
        do_sync = stage != "train"

        for name in self.model.loss_module.parameter_order:
            if name not in preds or name not in targets:
                continue
            p = preds[name]
            t = targets[name]
            if valid_mask is not None:
                p = p[valid_mask]
                t = t[valid_mask]
            if t.numel() == 0:
                continue

            residual = p - t

            # Match evaluation script behavior: wrap phi residuals to [-pi, pi].
            if name == "phi":
                residual = torch.where(residual > math.pi, residual - 2.0 * math.pi, residual)
                residual = torch.where(residual < -math.pi, residual + 2.0 * math.pi, residual)

            # MAE
            self.log(f"{stage}/{name}/mae", residual.abs().mean(), sync_dist=do_sync)

            # Precision: std of residuals in physical units
            if residual.numel() > 1:
                unit, scale = self._PRECISION_UNITS.get(name, ("", 1.0))
                self.log(
                    f"{stage}/{name}/precision {unit}",
                    residual.std() * scale,
                    sync_dist=do_sync,
                )

            # SSM precision on DM subset — val/test only (train has no ACTS data).
            # See the _log_metrics docstring for the exact DM filter semantics.
            if stage in ("val", "test"):
                acts_key = f"acts_reco_{name}"
                if acts_key in targets and "acts_dm_mask" in targets:
                    dm_mask = targets["acts_dm_mask"]
                    if valid_mask is not None:
                        dm_mask = dm_mask[valid_mask]

                    if dm_mask.any():
                        p_dm = p[dm_mask]
                        t_dm = t[dm_mask]
                        ssm_residual_dm = p_dm - t_dm
                        if name == "phi":
                            ssm_residual_dm = torch.where(ssm_residual_dm > math.pi, ssm_residual_dm - 2.0 * math.pi, ssm_residual_dm)
                            ssm_residual_dm = torch.where(ssm_residual_dm < -math.pi, ssm_residual_dm + 2.0 * math.pi, ssm_residual_dm)
                        if ssm_residual_dm.numel() > 1:
                            unit, scale = self._PRECISION_UNITS.get(name, ("", 1.0))
                            # σ — standard deviation (shared DM mask construction)
                            self.log(
                                f"{stage}/{name}/ssm_precision_dm {unit}",
                                ssm_residual_dm.std() * scale,
                                sync_dist=True,
                            )
                            # IQR/1.349 — robust σ estimator (tail-robust)
                            quant_levels = torch.tensor(
                                [0.25, 0.75],
                                device=ssm_residual_dm.device,
                                dtype=ssm_residual_dm.dtype,
                            )
                            q25, q75 = torch.quantile(ssm_residual_dm, quant_levels)
                            self.log(
                                f"{stage}/{name}/ssm_iqr_dm {unit}",
                                (q75 - q25) / 1.349 * scale,
                                sync_dist=True,
                            )
                            # RMS — captures outlier contribution (tail-sensitive)
                            rms = torch.sqrt((ssm_residual_dm ** 2).mean())
                            self.log(
                                f"{stage}/{name}/ssm_rms_dm {unit}",
                                rms * scale,
                                sync_dist=True,
                            )

    def _log_mdn_components(
        self,
        raw: Tensor,
        valid_mask: Tensor | None,
        stage: str,
    ) -> None:
        """Log per-component MDN diagnostics (mean π, μ, σ, effective K, MAP fraction).

        For every parameter whose loss is a ``MixtureDensityLoss``, record
        one scalar per component plus two aggregate diagnostics:
          - ``{stage}/{name}/mdn/pi_{k}``       — batch-mean mixture weight
          - ``{stage}/{name}/mdn/mu_{k}``       — batch-mean predicted location [physical units]
          - ``{stage}/{name}/mdn/sigma_{k}``    — batch-mean predicted spread  [physical units]
          - ``{stage}/{name}/mdn/effective_k``  — ⟨1/Σₖπₖ²⟩, measures how balanced π is
          - ``{stage}/{name}/mdn/map_frac_{k}`` — fraction of tracks whose MAP component is k

        Watching these during training diagnoses two failure modes:
          * One component capturing all tracks (`map_frac_{k} → 1`,
            `effective_k → 1`) → collapse back to single-Gaussian behaviour.
          * A component going dead (`pi_{k} → 0`, unchanging μ/σ) → K is
            too large; reduce in the next run.
        """
        loss_module = self.model.loss_module
        if not any(isinstance(sub, MixtureDensityLoss) for sub in loss_module.losses.values()):
            return

        do_sync = stage != "train"
        for name, sub_loss in loss_module.losses.items():
            if not isinstance(sub_loss, MixtureDensityLoss):
                continue
            start, end = loss_module.get_output_slice(name)
            r = raw[..., start:end]
            if valid_mask is not None:
                r = r[valid_mask]
            if r.numel() == 0:
                continue

            comps = sub_loss.predict_components(r)
            pi = comps["pi"]        # (N, K)
            mu = comps["mu"]        # (N, K) physical
            sigma = comps["sigma"]  # (N, K) physical
            K = sub_loss.n_components

            for k in range(K):
                self.log(f"{stage}/{name}/mdn/pi_{k}", pi[..., k].mean(), sync_dist=do_sync)
                self.log(f"{stage}/{name}/mdn/mu_{k}", mu[..., k].mean(), sync_dist=do_sync)
                self.log(f"{stage}/{name}/mdn/sigma_{k}", sigma[..., k].mean(), sync_dist=do_sync)

            pi_sq_sum = (pi ** 2).sum(dim=-1).clamp_min(1e-12)
            self.log(
                f"{stage}/{name}/mdn/effective_k",
                (1.0 / pi_sq_sum).mean(),
                sync_dist=do_sync,
            )

            k_map = pi.argmax(dim=-1)
            for k in range(K):
                self.log(
                    f"{stage}/{name}/mdn/map_frac_{k}",
                    (k_map == k).float().mean(),
                    sync_dist=do_sync,
                )

    # -- train / val / test ------------------------------------------------

    def training_step(self, batch, batch_idx):
        return {"loss": self._shared_step(batch, "train")}

    def validation_step(self, batch, batch_idx):
        return {"loss": self._shared_step(batch, "val")}

    def test_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self.model(inputs)

        valid_mask = targets.get("track_valid")
        losses = self.model.compute_loss(outputs, targets, valid_mask=valid_mask)
        crossing_metrics = self.model.loss_module.quantile_crossing_metrics(outputs["pred"], valid_mask=valid_mask)
        calibration_metrics = self.model.loss_module.quantile_calibration_metrics(outputs["pred"], targets, valid_mask=valid_mask)

        # Log every component
        for name, value in losses.items():
            self.log(f"test/{name}", value, sync_dist=True, prog_bar=(name == "total"))

        # Monitor quantile crossings on raw (unconstrained) quantile channels
        for name, value in crossing_metrics.items():
            self.log(f"test/{name}", value, sync_dist=True)

        # Monitor quantile calibration
        for name, value in calibration_metrics.items():
            self.log(f"test/{name}", value, sync_dist=True)

        # Compute predictions and metrics (predict_physical adds back delta anchors)
        preds = self.model.loss_module.predict_physical(outputs["pred"], targets)
        self._log_metrics(preds, targets, valid_mask, "test")

        # Full quantile predictions (for quantile-based losses)
        quantile_preds = self.model.loss_module.predict_quantiles(outputs["pred"])

        return {
            "loss": losses["total"],
            "preds": preds,
            "targets": targets,
            "quantile_preds": quantile_preds,
        }

    # -- optimiser / scheduler ---------------------------------------------

    def configure_optimizers(self):
        if self.opt_name.lower() == "muonhybrid":
            # Muon for 2-D interior matrix weights, AdamW for 1-D params
            # (norms, biases, SSM A_log/D, etc.) plus the input_net,
            # pool_head, and output_head Dense stacks.  The split is handled
            # by :func:`split_params_for_muon`; each group's initial LR is
            # set to its *peak* (``muon_max`` and ``max``) so the WSD / cosine
            # schedulers scale both groups proportionally from their own peak.
            # The LR ratio (``initial`` / ``max``) is shared across groups,
            # so the user must pick ``muon_max`` and ``max`` such that their
            # warmup-start and final LRs line up with the single scheduler
            # factor — typically ``muon_max ≈ 3–10× max`` per the Muon blog.
            muon_peak = float(self.lrs_config["muon_max"])
            adamw_peak = float(self.lrs_config["max"])
            param_groups = split_params_for_muon(
                self.model,
                muon_lr=muon_peak,
                muon_weight_decay=float(self.lrs_config.get("muon_weight_decay", self.lrs_config["weight_decay"])),
                adamw_lr=adamw_peak,
                adamw_weight_decay=float(self.lrs_config["weight_decay"]),
                adamw_betas=self.opt_betas if "betas" in self.lrs_config else (0.9, 0.95),
            )
            opt = MuonHybrid(
                param_groups,
                lr=adamw_peak,  # group-level lrs override this default
                momentum=float(self.lrs_config.get("muon_momentum", 0.95)),
                ns_steps=int(self.lrs_config.get("muon_ns_steps", 5)),
            )
        else:
            if self.opt_name.lower() == "adamw":
                opt_cls = AdamW
            elif self.opt_name.lower() == "lion":
                opt_cls = Lion
            else:
                raise ValueError(f"Unknown optimizer: {self.opt_name}")

            opt_kwargs: dict[str, Any] = dict(
                lr=self.lrs_config["initial"],
                weight_decay=self.lrs_config["weight_decay"],
            )
            # Both AdamW and lion_pytorch.Lion accept a ``betas`` kwarg; Lion
            # defaults to (0.9, 0.99) vs AdamW's (0.9, 0.999).  If the user did not
            # set betas in lrs_config we skip passing them so each optimizer keeps
            # its own default.
            if "betas" in self.lrs_config:
                opt_kwargs["betas"] = self.opt_betas

            opt = opt_cls(self.model.parameters(), **opt_kwargs)

        if not self.lrs_config.get("skip_scheduler"):
            schedule = self.lrs_config.get("schedule", "onecycle")
            total_steps = self.trainer.estimated_stepping_batches

            if schedule == "cosine":
                # Linear warmup + cosine annealing
                # Set optimizer LR to max BEFORE creating schedulers so that
                # base_lrs is captured correctly.  LinearLR's start_factor then
                # scales it down to `initial` at step 0, ramping up to `max`.
                for pg in opt.param_groups:
                    pg["lr"] = self.lrs_config["max"]
                warmup_steps = int(float(self.lrs_config["pct_start"]) * total_steps)
                warmup_sch = torch.optim.lr_scheduler.LinearLR(
                    opt,
                    start_factor=self.lrs_config["initial"] / self.lrs_config["max"],
                    end_factor=1.0,
                    total_iters=warmup_steps,
                )
                cosine_sch = torch.optim.lr_scheduler.CosineAnnealingLR(
                    opt,
                    T_max=total_steps - warmup_steps,
                    eta_min=self.lrs_config["end"],
                )
                sch = torch.optim.lr_scheduler.SequentialLR(
                    opt,
                    schedulers=[warmup_sch, cosine_sch],
                    milestones=[warmup_steps],
                )
            elif schedule == "cosine_freeze":
                # Three-phase schedule for frozen-backbone fine-tuning:
                #   Phase 1: [0, warmup_steps)        LinearLR  initial → max    (frozen)
                #   Phase 2: [warmup_steps, unfreeze) ConstantLR at max          (frozen)
                #   Phase 3: [unfreeze, total_steps)  CosineAnnealingLR
                #                                     unfreeze_lr → end         (unfrozen)
                # An external freeze callback (configured per-experiment)
                # is responsible for setting requires_grad at the unfreeze
                # milestone; this schedule drops the LR there so the
                # now-trainable backbone doesn't receive the (max) LR.
                warmup_steps = int(float(self.lrs_config["pct_start"]) * total_steps)
                unfreeze_epoch = int(self.lrs_config["unfreeze_epoch"])
                max_epochs = max(1, int(self.trainer.max_epochs))
                steps_per_epoch = max(1, total_steps // max_epochs)
                unfreeze_step = unfreeze_epoch * steps_per_epoch
                unfreeze_lr = float(self.lrs_config["unfreeze_lr"])

                if not (0 < warmup_steps <= unfreeze_step < total_steps):
                    raise ValueError(
                        "Invalid cosine_freeze schedule: "
                        f"warmup_steps={warmup_steps}, unfreeze_step={unfreeze_step}, "
                        f"total_steps={total_steps} — require "
                        "0 < warmup_steps <= unfreeze_step < total_steps"
                    )

                # Capture base_lr = max at scheduler construction time so that
                # LinearLR can scale it down to `initial` via start_factor.
                for pg in opt.param_groups:
                    pg["lr"] = self.lrs_config["max"]

                warmup_sch = torch.optim.lr_scheduler.LinearLR(
                    opt,
                    start_factor=self.lrs_config["initial"] / self.lrs_config["max"],
                    end_factor=1.0,
                    total_iters=warmup_steps,
                )
                hold_sch = torch.optim.lr_scheduler.ConstantLR(
                    opt,
                    factor=1.0,
                    total_iters=unfreeze_step - warmup_steps,
                )
                cosine_sch = torch.optim.lr_scheduler.CosineAnnealingLR(
                    opt,
                    T_max=total_steps - unfreeze_step,
                    eta_min=self.lrs_config["end"],
                )
                # Override the cosine scheduler's base_lrs to start decay from
                # unfreeze_lr (typically ~10x lower than max).  CosineAnnealingLR
                # reads from `self.base_lrs` at every step, so this reliably
                # produces `unfreeze_lr → end` over the remaining steps
                # regardless of what the optimizer's param-group LR currently is.
                cosine_sch.base_lrs = [unfreeze_lr for _ in cosine_sch.base_lrs]

                sch = torch.optim.lr_scheduler.SequentialLR(
                    opt,
                    schedulers=[warmup_sch, hold_sch, cosine_sch],
                    milestones=[warmup_steps, unfreeze_step],
                )
            elif schedule == "wsd":
                # Warmup-Stable-Decay (Hägele et al. 2024, arXiv:2405.18392).
                # Three phases, all driven by a single SequentialLR:
                #   [0, warmup)             LinearLR   initial → max
                #   [warmup, decay_start)   ConstantLR at max
                #   [decay_start, total)    CosineAnnealingLR  max → end
                # Recommended default for continuing from a converged checkpoint
                # (domain-shift fine-tuning): short warmup (pct_start ≈ 0.02),
                # long stable plateau, short cosine cooldown (decay_pct ≈ 0.15).
                # For multi-group optimizers (MuonHybrid) each group keeps its
                # own peak LR — LinearLR / CosineAnnealingLR apply the same
                # relative factor to every group's base_lr.
                warmup_pct = float(self.lrs_config.get("pct_start", 0.02))
                decay_pct = float(self.lrs_config.get("decay_pct", 0.15))
                warmup_steps = int(warmup_pct * total_steps)
                decay_start = int((1.0 - decay_pct) * total_steps)
                if not (0 < warmup_steps < decay_start < total_steps):
                    raise ValueError(
                        "Invalid WSD schedule: "
                        f"warmup_steps={warmup_steps}, decay_start={decay_start}, "
                        f"total_steps={total_steps} — require "
                        "0 < warmup_steps < decay_start < total_steps"
                    )

                if len(opt.param_groups) == 1:
                    # Single-group path: mirror the cosine branch and set the
                    # base LR to `max` so LinearLR's start_factor scales it down
                    # to `initial` at step 0.
                    for pg in opt.param_groups:
                        pg["lr"] = self.lrs_config["max"]
                # else: MuonHybrid — each group's 'lr' was set to its own peak
                # at construction time via split_params_for_muon, so we leave
                # them alone and LinearLR applies the same factor per group.

                start_factor = float(self.lrs_config["initial"]) / float(self.lrs_config["max"])
                warmup_sch = torch.optim.lr_scheduler.LinearLR(
                    opt,
                    start_factor=start_factor,
                    end_factor=1.0,
                    total_iters=warmup_steps,
                )
                hold_sch = torch.optim.lr_scheduler.ConstantLR(
                    opt,
                    factor=1.0,
                    total_iters=decay_start - warmup_steps,
                )
                cosine_sch = torch.optim.lr_scheduler.CosineAnnealingLR(
                    opt,
                    T_max=total_steps - decay_start,
                    eta_min=float(self.lrs_config["end"]),
                )
                sch = torch.optim.lr_scheduler.SequentialLR(
                    opt,
                    schedulers=[warmup_sch, hold_sch, cosine_sch],
                    milestones=[warmup_steps, decay_start],
                )
            else:
                sch = torch.optim.lr_scheduler.OneCycleLR(
                    opt,
                    max_lr=self.lrs_config["max"],
                    total_steps=total_steps,
                    div_factor=self.lrs_config["max"] / self.lrs_config["initial"],
                    final_div_factor=self.lrs_config["initial"] / self.lrs_config["end"],
                    pct_start=float(self.lrs_config["pct_start"]),
                )

            return [opt], [{"scheduler": sch, "interval": "step"}]

        return opt
