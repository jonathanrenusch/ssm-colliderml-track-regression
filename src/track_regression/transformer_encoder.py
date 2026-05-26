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

    def _apply_posenc(self, x: Tensor, hit_time: Tensor | None) -> Tensor:
        """Add the optional Fourier-of-time posenc to ``x`` (shape-polymorphic).

        Operates identically on padded ``(B, N, D)`` and packed
        ``(1, total_L, D)`` inputs — every op runs along the feature axis
        (``dim=-1``) and broadcasts over the batch/sequence plane.
        Without this perturbation the transformer is permutation-invariant
        in the hits (the input-feature Fourier conveys per-hit geometry,
        not on-helix arc length).
        """
        if self.posenc_proj is None or hit_time is None:
            return x
        t = (hit_time / self.posenc_time_scale).unsqueeze(-1)          # (..., N, 1)
        pe = _fourier_encode_scalar(
            t, self.posenc_fourier_scales, self.posenc_fourier_base,
        )                                                              # (..., N, 2K)
        pe = self.posenc_proj(pe.to(x.dtype))                          # (..., N, dim)
        return x + pe

    def forward(
        self,
        x: Tensor,
        x_sort_value: Tensor | None = None,
        kv_mask: Tensor | None = None,
        seq_idx: Tensor | None = None,
        cu_seqlens: Tensor | None = None,
        **kwargs: Any,
    ) -> tuple[Tensor, Tensor]:
        """Encode a sequence and return ``(hit_output, cls_tensor)``.

        Two modes, selected by whether ``cu_seqlens`` is provided:

        - **Padded** (``cu_seqlens is None``): ``x`` is ``(B, N, D)``,
          optionally masked with ``kv_mask``. Tokens are sorted by
          ``x_sort_value``; CLS tokens are prepended; the inner Encoder
          handles flash-varlen unpadding if needed; CLS is split off and
          ``hit_output`` is un-sorted back to input order.

        - **Packed** (``cu_seqlens`` given): ``x`` is ``(1, total_L, D)``
          with all batch tracks concatenated along ``total_L`` and
          per-token track IDs in ``seq_idx``. The data layer pre-sorts
          each segment by truth time, so no global argsort is performed
          (a global sort would mix segments). CLS tokens are interleaved
          at the **start** of each segment via an argsort-based gather;
          the inner Encoder layers are driven with augmented
          ``cu_seqlens`` so flash-varlen attention masks across
          segments naturally. Only ``attn_type='flash-varlen'`` and
          ``attn_type='torch'`` are supported on this path.

        Parameters
        ----------
        x : Tensor
            ``(B, N, D)`` padded, or ``(1, total_L, D)`` packed.
        x_sort_value : Tensor | None
            Padded: per-token sort key, also used for the posenc.
            Packed: per-token truth time for the posenc only (no sort).
        kv_mask : Tensor | None
            Padded mode: boolean ``(B, N)``. Must be ``None`` in packed
            mode.
        seq_idx : Tensor | None
            Packed mode: per-token track ID ``(1, total_L)``.
        cu_seqlens : Tensor | None
            Packed mode: cumulative segment ends ``(B + 1,)``.
        **kwargs
            Forwarded to the inner :class:`Encoder` (padded mode only).

        Returns
        -------
        tuple[Tensor, Tensor]
            - ``hit_output``: ``(B, N, D)`` padded or ``(1, total_L, D)``
              packed, in input order (CLS token stripped).
            - ``cls_tensor``: ``(B, num_cls_tokens * D)``.
        """
        if cu_seqlens is not None:
            if seq_idx is None:
                raise ValueError(
                    "Packed mode requires both seq_idx and cu_seqlens; "
                    "got cu_seqlens but seq_idx is None."
                )
            if kv_mask is not None:
                raise ValueError(
                    "kv_mask is unused in packed mode; got a non-None mask."
                )
            return self._forward_packed(
                x, hit_time=x_sort_value, seq_idx=seq_idx, cu_seqlens=cu_seqlens,
            )
        if seq_idx is not None:
            raise ValueError(
                "seq_idx given without cu_seqlens — for padded mode pass "
                "neither, for packed mode pass both."
            )

        B = x.shape[0]

        # Optional content-derived posenc, added BEFORE sort so the
        # posenc tensor travels with each token through the gather.
        x = self._apply_posenc(x, x_sort_value)

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

    def _forward_packed(
        self,
        x: Tensor,
        hit_time: Tensor | None,
        seq_idx: Tensor,
        cu_seqlens: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Packed-batch forward path.

        Layout:
        - ``x`` : ``(1, total_L, D)`` — all batch tracks concatenated.
        - ``hit_time`` : ``(1, total_L)`` — per-token truth time used by
          the posenc; carries no sort semantics here (segments are
          pre-sorted on disk).
        - ``seq_idx`` : ``(1, total_L)`` int — per-token track id.
        - ``cu_seqlens`` : ``(B + 1,)`` int — cumulative segment ends.

        Builds an augmented stream of length ``total_L + K * B`` where
        ``K = num_cls_tokens``, with K learned CLS rows prepended at
        every segment start, via the argsort-based interleave pattern
        from mamba_cls.py:_forward_packed. The inner Encoder's layer
        stack is driven directly with augmented ``cu_seqlens`` so
        ``flash_attn_varlen_func`` masks attention across segments
        naturally. ``attn_type='torch'`` is also accepted (for the fp32
        equivalence test) — a block-diagonal segment mask is built and
        passed as ``attn_mask`` instead.
        """
        if x.dim() != 3 or x.shape[0] != 1:
            raise ValueError(
                f"_forward_packed expects x of shape (1, total_L, D); "
                f"got {tuple(x.shape)}"
            )

        # --- Posenc on the original tokens, before the CLS interleave.
        # The CLS tokens themselves intentionally receive no posenc
        # (they are content-free register-like readouts).
        x = self._apply_posenc(x, hit_time)

        cu = cu_seqlens.to(torch.long)
        B = cu.shape[0] - 1
        total_L = x.shape[1]
        D = x.shape[2]
        K = self.num_cls_tokens
        if int(cu[-1].item()) != total_L:
            raise ValueError(
                f"cu_seqlens[-1]={int(cu[-1].item())} disagrees with "
                f"total_L={total_L}."
            )

        device = x.device
        seg_arange = torch.arange(B, device=device, dtype=torch.long)
        seq_idx_flat = seq_idx[0].to(torch.long)

        # Augmented positions.
        #   Each segment grows by K at the start, so:
        #     cls_pos[s, k] = cu[s] + s*K + k                  for k in 0..K-1
        #     orig_token_pos[t] = t + (seq(t) + 1) * K
        cls_block_starts = cu[:-1] + seg_arange * K                       # (B,)
        cls_offsets = torch.arange(K, device=device, dtype=torch.long)    # (K,)
        cls_positions = (cls_block_starts.unsqueeze(-1) + cls_offsets)    # (B, K)
        cls_positions_flat = cls_positions.reshape(B * K)                 # (B*K,)

        arange_total = torch.arange(total_L, device=device, dtype=torch.long)
        aug_token_positions = arange_total + (seq_idx_flat + 1) * K       # (total_L,)

        aug_total = total_L + B * K

        # Build augmented sequence via argsort-based interleave (same
        # pattern as mamba_cls.py:_forward_packed). No in-place writes
        # into a zeros tensor → CLS gradient flows correctly.
        cls_tok = self.cls_token[0].to(dtype=x.dtype)                     # (K, D)
        cls_values = cls_tok.unsqueeze(0).expand(B, K, D).reshape(B * K, D)

        all_values = torch.cat([x[0], cls_values], dim=0)                 # (aug_total, D)
        all_positions = torch.cat(
            [aug_token_positions, cls_positions_flat], dim=0,
        )                                                                  # (aug_total,)
        # `all_positions` is a permutation of [0, aug_total). Its
        # argsort places each value at its target position.
        inv_perm = torch.argsort(all_positions)
        x_aug = all_values[inv_perm].unsqueeze(0).contiguous()            # (1, aug_total, D)

        # Per-token augmented segment ids — needed if we build a
        # torch-attention block mask (no cost otherwise).
        cls_seg_ids = (
            seg_arange.unsqueeze(-1).expand(B, K).reshape(B * K)
        )                                                                  # (B*K,)
        all_seq_ids = torch.cat([seq_idx_flat, cls_seg_ids], dim=0)
        aug_seq_ids = all_seq_ids[inv_perm]                               # (aug_total,)

        # Augmented cu_seqlens (each segment grew by K).
        aug_seg_lens = (cu[1:] - cu[:-1]) + K                             # (B,) long
        aug_cu = torch.empty(B + 1, device=device, dtype=torch.int32)
        aug_cu[0] = 0
        aug_cu[1:] = torch.cumsum(aug_seg_lens.to(torch.int32), dim=0)
        max_aug = int(aug_seg_lens.max().item())

        # --- Attention plumbing per-backend.
        inner = self.encoder
        attn_type = inner.attn_type
        layer_kwargs: dict[str, Any] = {"kv_mask": None}
        if attn_type == "flash-varlen":
            layer_kwargs["varlen_kwargs"] = {
                "cu_seqlens": aug_cu, "max_seqlen": max_aug,
            }
        elif attn_type == "torch":
            # Block-diagonal segment mask: True iff same segment.
            block = (
                aug_seq_ids.unsqueeze(-1) == aug_seq_ids.unsqueeze(-2)
            )                                                              # (aug_total, aug_total)
            layer_kwargs["attn_mask"] = block.unsqueeze(0)                # (1, aug_total, aug_total)
        else:
            raise ValueError(
                f"Packed-batch transformer supports attn_type in "
                f"('flash-varlen', 'torch'); got '{attn_type}'."
            )

        # --- Drive the inner Encoder's layer stack directly. The
        # vendored Encoder.forward path runs flash-varlen unpadding from
        # a kv_mask; in packed mode we are *already* in the (1, total, D)
        # layout that unpadding would produce, so we bypass it and feed
        # the layers our pre-built varlen_kwargs (or block attn_mask).
        # No edits to _lib/encoder.py needed.
        initial_values: dict | None = {} if inner.value_residual else None
        for layer in inner.layers:
            x_aug = layer(
                x_aug,
                score_mod=inner.score_mod,
                initial_values=initial_values,
                **layer_kwargs,
            )

        # Final norm before CLS extraction (the inner Encoder has no
        # final norm — same convention as the padded path).
        x_aug = self.final_norm(x_aug)

        # CLS readouts: gather the K rows per segment at their known
        # augmented positions.  Shape (B, K, D) → (B, K*D).
        cls_rows = x_aug[0, cls_positions_flat, :].view(B, K, D)
        cls_out = cls_rows.reshape(B, K * self.dim)

        # Strip CLS: gather only the original-hit positions to recover
        # (1, total_L, D) in the input (= packed-collate) order.
        hit_out = x_aug[0:1, aug_token_positions, :]

        # DDP unused-parameter tie — same as padded path.
        cls_out = cls_out + 0.0 * hit_out.float().sum()

        return hit_out, cls_out
