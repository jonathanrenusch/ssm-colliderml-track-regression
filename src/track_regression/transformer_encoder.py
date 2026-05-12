"""Transformer encoder baseline with a learned CLS token.

This is a thin wrapper around :class:`track_regression._lib.encoder.Encoder` that:

- Adds a single learned ``cls_token`` at the front of the sequence.
- Returns ``(sequence_output, cls_tensor)`` instead of just the sequence
  output — matching the API of the SSM encoders so that
  :class:`TrackParameterRegressor` can use the same ``pool='register_token'``
  pathway.

Design: composition, not inheritance.
--------------------------------------
Rather than subclassing :class:`Encoder` and duplicating its sophisticated
forward logic (flash-varlen unpadding, sliding window masks, sorting,
etc.), we *contain* an ``Encoder`` that is constructed without its own
register-token support (``num_register_tokens=None``).  ``EncoderWithCLS``
then performs:

1. Optional sort-by-``x_sort_value`` (same convention as the SSM encoder).
2. Prepend the learned CLS token to the sorted sequence (and the
   corresponding mask if present).
3. Call the inner :class:`Encoder`'s ``forward`` with the CLS-augmented
   sequence — but with ``x_sort_value=None`` so the inner encoder does
   not re-sort.
4. Split the output into ``(cls_tensor, hit_output)`` and un-sort the hit
   output back to the original input order.
5. Return ``(hit_output, cls_tensor)``.

This keeps the shared :class:`Encoder` untouched — the trackml experiment
is unaffected.

Also includes a DDP unused-parameter tie (``cls_out + 0.0 *
hit_output.sum()``) so that every encoder parameter contributes to the
autograd graph even when the downstream model only consumes the CLS
output.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from track_regression._lib.encoder import Encoder


def _fourier_encode_scalar(
    x: Tensor,
    fourier_scales: list[int],
    fourier_base: int,
) -> Tensor:
    """Multi-scale Fourier features for a scalar per-token signal.

    Input ``x`` has shape ``(*, 1)``; output ``(*, 2 * len(fourier_scales))``
    by concatenating ``sin(x / base^n)`` and ``cos(x / base^n)`` over the
    scale ladder. Kept local to avoid a cross-import of ``model.py``'s
    ``fourier_encode`` (same formula, scalar-specific signature).
    """
    sin = [torch.sin(x / (fourier_base**n)) for n in fourier_scales]
    cos = [torch.cos(x / (fourier_base**n)) for n in fourier_scales]
    return torch.cat(sin + cos, dim=-1)


class EncoderWithCLS(nn.Module):
    """Transformer encoder that returns both the sequence output and a CLS token.

    Parameters
    ----------
    dim : int
        Model dimension (embedding size).
    cls_init_scale : float
        Standard deviation of the learned CLS token initialisation.
    num_cls_tokens : int
        Number of learned CLS tokens prepended to the sequence (e.g. 2
        for the parameter-matched-to-SSM-CLS baseline).
    posenc_fourier_scales : list[int] | None
        Exponent ladder for the Fourier-of-time positional encoding. When
        non-None **and** ``x_sort_value`` is supplied at forward time, an
        additive content-derived posenc is computed from the per-hit
        sort-key value (in this project that's truth time, the on-helix
        arc-length proxy) and added to the embedded tokens before the
        CLS tokens are prepended. ``None`` (default) → no posenc, and
        the transformer remains strictly permutation-invariant in the
        hits.
    posenc_fourier_base : int
        Base for the Fourier ladder (period_i = base^scale_i).
    posenc_time_scale : float
        Scalar divisor applied to the time signal before Fourier
        encoding. Tune so that ``time / posenc_time_scale`` for typical
        tracks lands in the [0, ~few] range the Fourier basis covers
        well. The truth-time range on this dataset is roughly 0-45 ns;
        the default 5.0 puts the bulk of tracks into [0, ~3].
    posenc_init_scale : float
        Std-dev of the initial Linear projection of the Fourier features
        to ``dim``. Small (0.02) so the posenc starts as a perturbation
        on top of the input embedding and grows under training signal.
    **encoder_kwargs
        All other kwargs are forwarded to :class:`track_regression._lib.encoder.Encoder`.
        ``num_register_tokens`` is silently dropped if present — the CLS
        token is managed at this layer, not by the inner Encoder.
    """

    def __init__(
        self,
        dim: int,
        cls_init_scale: float = 0.02,
        num_cls_tokens: int = 1,
        # Posenc is ON by default — the transformer is otherwise strictly
        # permutation-invariant in the hits (Fourier encoding of x,y,z,r,…
        # only conveys per-hit geometry, not on-helix arc length).
        # Pass ``posenc_fourier_scales: []`` to disable.
        posenc_fourier_scales: list[int] = (-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5),
        posenc_fourier_base: int = 2,
        posenc_time_scale: float = 1.0,
        posenc_init_scale: float = 0.02,
        **encoder_kwargs: Any,
    ) -> None:
        super().__init__()
        if num_cls_tokens < 1:
            raise ValueError(f"num_cls_tokens must be >= 1; got {num_cls_tokens}")
        self.dim = dim
        self.num_cls_tokens = int(num_cls_tokens)

        # The inner Encoder must not own a register token — we manage the
        # CLS ourselves so we can read it out cleanly.  Silently drop the
        # key to keep YAML configs flexible.
        encoder_kwargs.pop("num_register_tokens", None)

        self.encoder = Encoder(dim=dim, **encoder_kwargs)
        # (1, num_cls_tokens, dim) — broadcasts to batch in forward.
        self.cls_token = nn.Parameter(
            torch.randn(1, self.num_cls_tokens, dim) * cls_init_scale
        )

        # Final RMSNorm on the encoder output.  The inner Encoder uses
        # pre-norm layers (norm → attn/ffn → residual) but has no final
        # norm, so the residual stream magnitude grows with depth.
        # Normalising before CLS extraction stabilises the regression head.
        self.final_norm = nn.RMSNorm(dim)

        # ── Content-derived Fourier-of-time positional encoding ─────────────
        # Disabled if the scales list is empty.
        self.posenc_fourier_scales = list(posenc_fourier_scales)
        self.posenc_fourier_base = int(posenc_fourier_base)
        self.posenc_time_scale = float(posenc_time_scale)
        if self.posenc_fourier_scales:
            k = len(self.posenc_fourier_scales)
            # Linear maps (sin+cos of K scales) → dim. Small init so the
            # posenc starts as a perturbation on top of the embedded tokens.
            self.posenc_proj = nn.Linear(2 * k, dim, bias=True)
            with torch.no_grad():
                self.posenc_proj.weight.mul_(posenc_init_scale)
                self.posenc_proj.bias.zero_()
        else:
            self.posenc_proj = None

    @property
    def pool_dim(self) -> int:
        """Dimension of the concatenated CLS readout: ``num_cls_tokens * dim``."""
        return self.num_cls_tokens * self.dim

    def forward(
        self,
        x: Tensor,
        x_sort_value: Tensor | None = None,
        kv_mask: Tensor | None = None,
        **kwargs: Any,
    ) -> tuple[Tensor, Tensor]:
        """Encode a sequence and return ``(hit_output, cls_tensor)``.

        Parameters
        ----------
        x : Tensor
            Input of shape ``(B, N, D)``.
        x_sort_value : Tensor | None
            Values to sort tokens by before encoding (optional).  The CLS
            token is inserted *after* sorting. In this project the sort
            key is also the truth time used by the optional posenc, so
            ``x_sort_value`` doubles as the posenc argument when the
            posenc is configured.
        kv_mask : Tensor | None
            Boolean mask of shape ``(B, N)``.  ``True`` = valid token,
            ``False`` = padding.  A column of ``True`` is prepended for
            the CLS token.
        **kwargs
            Forwarded to the inner :class:`Encoder`.

        Returns
        -------
        tuple[Tensor, Tensor]
            - ``hit_output`` of shape ``(B, N, D)`` in the original input
              order (CLS token stripped, un-sorted).
            - ``cls_tensor`` of shape ``(B, D)``.
        """
        B = x.shape[0]

        # Optional content-derived posenc, added BEFORE sort so the
        # posenc tensor travels with each token through the gather.
        # Without this the transformer would be permutation-invariant in
        # the hits — Fourier encoding of x,y,z,... only conveys
        # geometric coordinate; it does not convey on-helix arc length.
        # ``x_sort_value`` is the truth time (linear in arc length),
        # which is the right thing to embed.
        if self.posenc_proj is not None and x_sort_value is not None:
            t = (x_sort_value / self.posenc_time_scale).unsqueeze(-1)  # (B, N, 1)
            pe = _fourier_encode_scalar(
                t, self.posenc_fourier_scales, self.posenc_fourier_base,
            )                                                          # (B, N, 2K)
            pe = self.posenc_proj(pe.to(x.dtype))                      # (B, N, dim)
            x = x + pe

        # Sort first (if requested) — keep track of the sort index so we
        # can un-sort the hit output later.  The inner Encoder is called
        # with ``x_sort_value=None`` below so it does not re-sort.
        x_sort_idx: Tensor | None = None
        if x_sort_value is not None:
            x_sort_idx = torch.argsort(x_sort_value, dim=-1)
            x = torch.gather(x, -2, x_sort_idx.unsqueeze(-1).expand_as(x))
            if kv_mask is not None:
                kv_mask = torch.gather(kv_mask, -1, x_sort_idx)

        # Prepend the learned CLS tokens (one or more).
        cls_tok = self.cls_token.expand(B, -1, -1).to(dtype=x.dtype)
        x_aug = torch.cat([cls_tok, x], dim=1)

        # Extend the mask with ``True`` for each CLS position so all CLS
        # tokens participate in attention.
        kv_mask_aug: Tensor | None = None
        if kv_mask is not None:
            cls_mask = torch.ones(
                (B, self.num_cls_tokens), dtype=kv_mask.dtype, device=kv_mask.device
            )
            kv_mask_aug = torch.cat([cls_mask, kv_mask], dim=1)

        # Run the inner Encoder without its own sorting.
        seq_out = self.encoder(
            x_aug,
            x_sort_value=None,
            kv_mask=kv_mask_aug,
            **kwargs,
        )

        # Normalise before splitting — the pre-norm Encoder has no final
        # LayerNorm so the residual stream magnitude grows with depth.
        seq_out = self.final_norm(seq_out)

        # Split CLS and hit outputs.  Concatenate the CLS rows into a
        # single readout vector of shape (B, num_cls_tokens * D), matching
        # the convention used by BidirectionalMambaCLSEncoder when
        # num_cls_tokens > 1 (one register per "side").  For
        # num_cls_tokens == 1 this collapses to (B, D), preserving exact
        # backward compatibility with prior 1-register checkpoints.
        cls_rows = seq_out[:, : self.num_cls_tokens, :]    # (B, K, D)
        cls_out = cls_rows.reshape(B, self.num_cls_tokens * self.dim)
        hit_out = seq_out[:, self.num_cls_tokens :, :]     # (B, N, D)

        # Un-sort the hit output back to the original input order.
        if x_sort_idx is not None:
            x_unsort_idx = torch.argsort(x_sort_idx, dim=-1)
            hit_out = torch.gather(
                hit_out, -2, x_unsort_idx.unsqueeze(-1).expand_as(hit_out)
            )

        # DDP unused-parameter tie: pull the hit-output path (and therefore
        # every internal encoder parameter) into the autograd graph even
        # when the downstream model consumes only the CLS output.
        # Numerically a no-op.  The .float() prevents bf16 overflow on the
        # sum (long sequences can exceed bf16 range).
        cls_out = cls_out + 0.0 * hit_out.float().sum()

        return hit_out, cls_out
