"""Bidirectional Mamba-2 encoder with learned CLS tokens.

This is an alternative pooling strategy to :mod:`mamba_state`.  Instead of
extracting the final SSM recurrent hidden state (``nheads * headdim *
d_state`` per direction), we learn two CLS tokens that are inserted at the
terminal positions of each scan direction:

- ``cls_fwd`` is **appended** to the sorted sequence so the forward SSM
  sees it *last* — after accumulating state across every hit.
- ``cls_bwd`` is **prepended** to the sorted sequence so that, after the
  ``torch.flip`` inside :class:`BidirectionalMambaLayer`, it becomes the
  last token the backward SSM sees — again, after accumulating state
  across every hit.

Both CLS tokens flow through all ``num_layers`` encoder layers so they
accumulate representation across depth, ViT/Vision-Mamba style.  The
final layer is a custom :class:`BidirectionalMambaCLSFinalLayer` that
exposes the per-direction Mamba-2 outputs at the CLS positions *before*
the gated bidirectional merge — which would otherwise contaminate the
two readouts with their opposite-direction partner's irrelevant
"only-seen-one-token" output.

The encoder returns a pooled tensor of shape ``(B, 2 * dim)`` formed by
concatenating ``(cls_fwd_out, cls_bwd_out)``, alongside the (sorted, then
un-sorted) per-hit sequence output.  Paired with the
``pool='ssm_cls'`` branch of :class:`TrackParameterRegressor`, this
bypasses the ``state_head`` projection entirely (the output_head takes
``2 * dim``-dim input directly).

DDP unused-parameter tie
------------------------
If the downstream model consumes only the CLS output and discards the
per-hit sequence output, the ``gate`` parameters inside each layer would
have no gradient — causing DDP to raise "unused parameter" errors.  To
avoid this, the forward adds ``0.0 * sequence_output.sum()`` to the CLS
output, which forces the sequence-output path (and therefore all
internal layer parameters) into the autograd graph without changing the
numerical value.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

try:
    from mamba_ssm.modules.mamba2 import Mamba2

    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False
    Mamba2 = nn.Module  # type: ignore[assignment, misc]

from track_regression.mamba_state import BidirectionalMambaLayer, _build_mamba


def _segment_flip_indices(cu_seqlens: Tensor, total_len: int) -> Tensor:
    """Build a ``(total_len,)`` gather index that reverses each segment.

    Given ``cu_seqlens = [c_0, c_1, ..., c_B]`` with ``c_0 = 0`` and
    ``c_B = total_len``, the segment ``s`` covers packed positions
    ``[c_s, c_{s+1})``. This helper returns a long tensor ``idx`` such
    that ``x.gather(1, idx[None, :, None].expand(...))`` reverses every
    segment in place — i.e. the token at position ``c_s + i`` is mapped
    to position ``c_s + (L_s - 1 - i)`` where ``L_s = c_{s+1} - c_s``.

    Implementation: for every absolute position ``p``, find the segment
    via ``bucketize(p, cu_seqlens[1:])``, then compute
    ``new_p = c_s + c_{s+1} - 1 - p``. Vectorised, no loop.

    Segment-wise flip is its own inverse — calling ``gather`` twice with
    the same ``idx`` recovers the original tensor.

    Parameters
    ----------
    cu_seqlens : Tensor
        Cumulative segment lengths, shape ``(B+1,)``, dtype long or
        int32 (cast internally to long for arithmetic).
    total_len : int
        Total packed sequence length (equal to ``cu_seqlens[-1]``).
    """
    cs = cu_seqlens.to(torch.long)
    arange = torch.arange(total_len, device=cs.device, dtype=torch.long)
    # right=True: a boundary value (e.g. position == cs[s+1]) belongs to the
    # *next* segment, matching the half-open interval [cs[s], cs[s+1]).
    seg = torch.bucketize(arange, cs[1:], right=True)
    seg_start = cs[seg]
    seg_end = cs[seg + 1]
    return seg_start + seg_end - 1 - arange


class BidirectionalMambaCLSFinalLayer(nn.Module):
    """Final bidirectional Mamba layer that exposes per-direction CLS readouts.

    Structurally identical to :class:`BidirectionalMambaLayer` (norm →
    forward Mamba-2 + backward Mamba-2 → gated merge → residual) but in
    addition to the gated sequence output it returns:

    - ``cls_fwd_out = x_fwd[:, -1, :]`` — the forward Mamba-2's output at
      the terminal position of the sequence (where ``cls_fwd`` lives).
      This is ungated (and therefore uncontaminated by the backward
      output at the same position, which has only seen one token).
    - ``cls_bwd_out = x_bwd[:, 0, :]`` — the backward Mamba-2's output at
      position 0 after the post-scan flip, which is the flipped-scan's
      terminal position (where ``cls_bwd`` lives after flipping).

    Layout convention expected on input: ``(cls_bwd, h_0, …, h_{L-1},
    cls_fwd)`` of shape ``(B, L+2, D)``.
    """

    def __init__(
        self,
        dim: int,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        headdim: int = 64,
        ngroups: int = 1,
        chunk_size: int = 256,
        norm: str = "LayerNorm",
        dropout: float = 0.0,
        mamba_impl: str = "stock",
    ):
        super().__init__()
        self.dim = dim

        if norm == "LayerNorm":
            self.norm = nn.LayerNorm(dim)
        elif norm == "RMSNorm":
            self.norm = nn.RMSNorm(dim)
        else:
            raise ValueError(f"Unknown norm: {norm}")

        mamba_kwargs = {
            "d_model": dim,
            "d_state": d_state,
            "d_conv": d_conv,
            "expand": expand,
            "headdim": headdim,
            "ngroups": ngroups,
            "chunk_size": chunk_size,
        }
        self.forward_mamba = _build_mamba(mamba_kwargs, mamba_impl)
        self.backward_mamba = _build_mamba(mamba_kwargs, mamba_impl)

        self.gate = nn.Linear(dim, dim)
        self.gate_activation = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(
        self,
        x: Tensor,
        seq_idx: Tensor | None = None,
        flip_indices: Tensor | None = None,
        cls_fwd_positions: Tensor | None = None,
        cls_bwd_positions: Tensor | None = None,
        lens: Tensor | None = None,
        cu_seqlens: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Forward pass.

        In padded mode (``flip_indices is None``) the layout is
        ``(B, L+2, D)`` with CLS tokens at fixed positions ``[:, 0, :]``
        and ``[:, -1, :]``. In packed mode the layout is
        ``(1, sum_i(L_i + 2), D)`` and CLS positions are passed via the
        ``cls_fwd_positions`` / ``cls_bwd_positions`` ``(B,)`` index
        tensors.
        """
        skip = x
        x_norm = self.norm(x).contiguous()

        if (
            cu_seqlens is not None
            and getattr(self, "_packed_fused", False)
        ):
            # v5p packed-stream fused path (see fused_bidi_scan_packed).
            from .mamba_short import fused_bidi_scan_packed

            x_fwd, x_bwd = fused_bidi_scan_packed(self, x_norm, cu_seqlens)
        elif (
            flip_indices is not None
            and flip_indices.dim() == 2
            and hasattr(self, "_fused_in_w")
        ):
            # V4.1 fused path (see mamba_short.fused_bidi_scan).
            from .mamba_short import fused_bidi_scan

            x_fwd, x_bwd = fused_bidi_scan(self, x_norm, flip_indices, lens)
        else:
            x_fwd = self.forward_mamba(x_norm, seq_idx=seq_idx)

            if flip_indices is None:
                x_bwd_in = torch.flip(x_norm, dims=[1]).contiguous()
                x_bwd_out = self.backward_mamba(x_bwd_in, seq_idx=seq_idx)
                x_bwd = torch.flip(x_bwd_out, dims=[1])
            else:
                if flip_indices.dim() == 1:
                    gather_idx = flip_indices.unsqueeze(0).unsqueeze(-1).expand_as(x_norm)
                else:
                    # Padded-static mode — per-row (B, L) valid-prefix flip.
                    gather_idx = flip_indices.unsqueeze(-1).expand_as(x_norm)
                x_bwd_in = torch.gather(x_norm, 1, gather_idx).contiguous()
                x_bwd_out = self.backward_mamba(x_bwd_in, seq_idx=seq_idx)
                x_bwd = torch.gather(x_bwd_out, 1, gather_idx)

        gate = self.gate_activation(self.gate(x_norm))
        x_combined = gate * x_fwd + (1 - gate) * x_bwd
        output = skip + self.dropout(x_combined)

        # Per-direction CLS readouts — extracted from the gated, residualized
        # output.  Previously these were taken from the raw scan outputs
        # (x_fwd[:, -1] / x_bwd[:, 0]) which bypassed the stabilising gate
        # and residual connection, causing ~5 OOM gradient explosion.
        if cls_fwd_positions is None:
            # Padded mode — fixed terminal positions per batch row.
            cls_fwd_out = output[:, -1, :]   # forward scan terminal → cls_fwd
            cls_bwd_out = output[:, 0, :]    # backward scan terminal → cls_bwd
        elif output.shape[0] == 1 and cls_bwd_positions is not None:
            # Packed mode — gather at per-segment terminal positions.
            cls_fwd_out = output[0, cls_fwd_positions, :]  # (B, D)
            cls_bwd_out = output[0, cls_bwd_positions, :]  # (B, D)
        else:
            # Padded-static mode — cls_fwd sits at the per-row compacted
            # position Lr+1; cls_bwd is always at column 0.
            rows = torch.arange(output.shape[0], device=output.device)
            cls_fwd_out = output[rows, cls_fwd_positions, :]  # (B, D)
            cls_bwd_out = output[:, 0, :]                     # (B, D)

        return output, cls_fwd_out, cls_bwd_out


class BidirectionalMambaCLSEncoder(nn.Module):
    """Bidirectional Mamba-2 encoder with learned CLS tokens.

    Stacks ``num_layers - 1`` plain :class:`BidirectionalMambaLayer`
    layers followed by one :class:`BidirectionalMambaCLSFinalLayer`.  All
    layers see the CLS-augmented sequence ``(cls_bwd, hits, cls_fwd)``;
    the CLS tokens therefore accumulate representation across depth.

    Parameters
    ----------
    num_layers, dim, d_state, d_conv, expand, headdim, ngroups, chunk_size, norm, dropout
        See :class:`BidirectionalMambaLayer`.
    cls_init_scale : float
        Standard deviation of the CLS-token initialisation.
    residual_depth_init : bool
        When True, rescale every ``out_proj.weight`` (forward and backward
        Mamba-2 of every layer, intermediate and final) by ``1/sqrt(2 *
        num_layers)`` after construction. This is the standard
        residual-depth init prescription (one factor of 1/sqrt(N_residuals)
        on the projection that writes back into the residual stream) so
        that the variance accumulated across depth is approximately
        constant. Each :class:`BidirectionalMambaLayer` contributes one
        residual-stream addition (the gated merge of forward+backward),
        and the residual is applied at *both* forward and backward
        out_projs through the gate, so the canonical 1/sqrt(2N) factor is
        applied to both. Off by default to preserve checkpoint
        state-dict compatibility with existing runs.
    """

    def __init__(
        self,
        num_layers: int,
        dim: int,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        headdim: int = 64,
        ngroups: int = 1,
        chunk_size: int = 256,
        norm: str = "LayerNorm",
        dropout: float = 0.0,
        cls_init_scale: float = 0.02,
        residual_depth_init: bool = False,
        mamba_impl: str = "stock",
    ):
        super().__init__()
        assert num_layers >= 1, "Need at least one layer"

        self.num_layers = num_layers
        self.dim = dim

        # Learned CLS tokens.  Initialised with a small Gaussian.
        self.cls_fwd = nn.Parameter(torch.randn(1, 1, dim) * cls_init_scale)
        self.cls_bwd = nn.Parameter(torch.randn(1, 1, dim) * cls_init_scale)

        # mamba_impl="short" builds the native Mamba2Short kernel directly, so the
        # model needs only torch (no mamba_ssm/causal_conv1d/nvcc). Checkpoints load
        # unchanged; apply_variant("v5pc") still enables the fused Triton path.
        common = dict(
            dim=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            headdim=headdim,
            ngroups=ngroups,
            chunk_size=chunk_size,
            norm=norm,
            dropout=dropout,
            mamba_impl=mamba_impl,
        )

        # Intermediate layers (plain bidirectional, CLS passes through as just
        # two more tokens — no special handling needed at intermediate depth).
        self.layers = nn.ModuleList(
            [BidirectionalMambaLayer(**common) for _ in range(num_layers - 1)]
        )

        # Final layer returns per-direction CLS outputs without gating.
        self.final_layer = BidirectionalMambaCLSFinalLayer(**common)

        # Post-encoder normalisation on the full augmented sequence.
        if norm == "LayerNorm":
            self.final_norm = nn.LayerNorm(dim)
        elif norm == "RMSNorm":
            self.final_norm = nn.RMSNorm(dim)
        else:
            self.final_norm = nn.Identity()

        # Normalise CLS readouts before they enter the regression head.
        # The per-direction CLS outputs are extracted from raw Mamba-2 scan
        # outputs (ungated, no residual) and bypass ``final_norm`` above.
        # Without this, their unconstrained magnitude causes gradient spikes.
        self.cls_norm = nn.RMSNorm(dim)

        # Residual-depth init rescaling on out_proj. Applied AFTER all
        # submodules are constructed so it overrides the Mamba2 default
        # init for these specific tensors.
        if residual_depth_init:
            scale = 1.0 / math.sqrt(2.0 * num_layers)
            with torch.no_grad():
                for layer in self.layers:
                    layer.forward_mamba.out_proj.weight.mul_(scale)
                    layer.backward_mamba.out_proj.weight.mul_(scale)
                self.final_layer.forward_mamba.out_proj.weight.mul_(scale)
                self.final_layer.backward_mamba.out_proj.weight.mul_(scale)

    @property
    def pool_dim(self) -> int:
        """Dimension of the concatenated ``(cls_fwd, cls_bwd)`` pooled output."""
        return 2 * self.dim

    def forward(
        self,
        x: Tensor,
        x_sort_value: Tensor | None = None,
        seq_idx: Tensor | None = None,
        cu_seqlens: Tensor | None = None,
        **kwargs,  # noqa: ARG002 — API compatibility with other encoders
    ) -> tuple[Tensor, Tensor]:
        """Encode a sequence and return ``(seq_output, cls_concat)``.

        Two layouts are supported, selected by whether ``cu_seqlens`` is
        provided:

        - **Padded** (default, ``cu_seqlens is None``): ``x`` is
          ``(B, N, D)`` with one track per batch row. The encoder
          optionally argsorts by ``x_sort_value``, prepends/appends CLS
          tokens at fixed positions, and runs the bidirectional scan
          with global flip.
        - **Packed** (``cu_seqlens`` given): ``x`` is
          ``(1, total_L, D)`` with all batch tracks concatenated and
          per-token segment IDs in ``seq_idx``. CLS tokens are
          interleaved at each segment boundary and the backward Mamba
          uses a segment-wise flip. ``x_sort_value`` is ignored —
          packed callers must pre-sort hits per segment in collate.

        Parameters
        ----------
        x : Tensor
            Padded ``(B, N, D)`` *or* packed ``(1, total_L, D)``.
        x_sort_value : Tensor | None
            Padded mode only — values to sort tokens by before
            encoding. Ignored when ``cu_seqlens`` is provided.
        seq_idx : Tensor | None
            Padded mode: unused (sequences are already split across
            batch dim). Packed mode: per-token segment ID, shape
            ``(1, total_L)`` int32.
        cu_seqlens : Tensor | None
            Cumulative segment lengths for packed mode, shape
            ``(B + 1,)``. ``cu_seqlens[0] = 0``, ``cu_seqlens[B] =
            total_L``. When provided, switches to the packed path.

        Returns
        -------
        tuple[Tensor, Tensor]
            - Sequence output, shape ``(B, N, D)`` (padded) or
              ``(1, total_L, D)`` (packed). CLS tokens are stripped.
            - Pooled CLS summary, shape ``(B, 2 * D)``.
        """
        if cu_seqlens is not None:
            use_static = getattr(self, "_static_mode", False)
            if use_static and getattr(self, "_auto_kernel", False) and not self.training:
                # auto mode: eval/val/test ride the fused packed path (v5pc).
                use_static = False
            if use_static:
                return self._forward_padded_static(x, cu_seqlens)
            return self._forward_packed(x, seq_idx, cu_seqlens)

        B = x.shape[0]

        # The legacy padded path is knowingly NOT equivalent to packed mode
        # for variable-length tracks: pad tokens leak into the scan state,
        # cls_fwd sits after the pad slots (wrong conv window / state), and
        # pads argsort to the front (hit_time == 0.0). Kept for backward
        # compatibility only. Use packed mode (cu_seqlens) or the padded-
        # static campaign path (enable_static_mode) instead.
        import warnings

        warnings.warn(
            "BidirectionalMambaCLSEncoder padded path is deprecated: it is "
            "not physics-equivalent to packed mode for variable-length "
            "tracks (pad leakage + CLS placement + hit_time-0 sort). Use "
            "packed batches or enable_static_mode().",
            DeprecationWarning,
            stacklevel=2,
        )

        # Optional sort (e.g. by distance from IP).  CLS tokens are inserted
        # AFTER this step — the plan calls this out explicitly because
        # appending/prepending before sort would scramble the CLS positions.
        x_sort_idx = None
        if x_sort_value is not None:
            x_sort_idx = torch.argsort(x_sort_value, dim=-1)
            x = torch.gather(x, -2, x_sort_idx.unsqueeze(-1).expand_as(x)).contiguous()

        # Insert CLS tokens at the terminal positions of each scan direction.
        # Layout: (cls_bwd, h_0, ..., h_{L-1}, cls_fwd).
        cls_bwd_tok = self.cls_bwd.expand(B, -1, -1).to(dtype=x.dtype)
        cls_fwd_tok = self.cls_fwd.expand(B, -1, -1).to(dtype=x.dtype)
        x_aug = torch.cat([cls_bwd_tok, x, cls_fwd_tok], dim=1).contiguous()

        # Intermediate layers — CLS tokens flow through unchanged.
        for layer in self.layers:
            x_aug = layer(x_aug, seq_idx=seq_idx)

        # Final layer returns per-direction CLS readouts ungated.
        x_aug, cls_fwd_out, cls_bwd_out = self.final_layer(x_aug, seq_idx=seq_idx)

        x_aug = self.final_norm(x_aug)

        # Strip CLS tokens from the sequence output — keep only the hit positions.
        x_hits = x_aug[:, 1:-1, :]

        # Un-sort hits back to original order.
        if x_sort_idx is not None:
            x_unsort_idx = torch.argsort(x_sort_idx, dim=-1)
            x_hits = torch.gather(x_hits, -2, x_unsort_idx.unsqueeze(-1).expand_as(x_hits))

        # Normalise the per-direction CLS readouts (raw Mamba-2 scan
        # outputs with unconstrained magnitude) before concatenation.
        cls_fwd_out = self.cls_norm(cls_fwd_out)
        cls_bwd_out = self.cls_norm(cls_bwd_out)

        # Concatenate the per-direction CLS readouts → (B, 2*dim).
        cls_concat = torch.cat([cls_fwd_out, cls_bwd_out], dim=-1)

        # DDP unused-parameter tie: pull the per-hit sequence-output path
        # (and therefore the `gate` weights inside each layer) into the
        # autograd graph even when the downstream model discards x_hits.
        # Numerically a no-op.  The .float() prevents bf16 overflow on the
        # sum (long sequences can exceed bf16 range).
        cls_concat = cls_concat + 0.0 * x_hits.float().sum()

        return x_hits, cls_concat

    def enable_static_mode(self, compile_core: bool = False) -> None:
        """Switch packed inputs onto the padded-static (B, 22, D) path.

        Campaign hook (see ``mamba_short.apply_variant``): the packed batch
        from the unchanged dataloader is scattered into compacted static
        rows ``[cls_bwd, h_0..h_{Lr-1}, cls_fwd, PAD..]`` on GPU — pads
        strictly trail in both scan directions (per-row valid-prefix flip),
        so no masking is needed inside the Mamba blocks (Scheme A).
        """
        self._static_mode = True
        self._static_core_fn = self._static_core
        if compile_core:
            self._static_core_fn = torch.compile(self._static_core, dynamic=False)

    def _static_core(
        self,
        x_aug: Tensor,
        valid: Tensor,
        flip_idx: Tensor,
        cls_fwd_pos: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Static-shape layer stack — the compilable core."""
        vm = valid.unsqueeze(-1).to(x_aug.dtype)
        # Precompute once: per-row last valid index for the in-kernel flip
        # (deriving it inside every layer costs a materialisation each time).
        lens32 = cls_fwd_pos.to(torch.int32).contiguous()
        for layer in self.layers:
            # Safety re-zero of pad rows: not required for parity (pads
            # strictly trail), but bounds pad-row garbage and keeps the
            # DDP-tie sum over x_hits finite.
            x_aug = layer(x_aug, seq_idx=None, flip_indices=flip_idx, lens=lens32) * vm
        x_aug, cls_fwd_out, cls_bwd_out = self.final_layer(
            x_aug,
            seq_idx=None,
            flip_indices=flip_idx,
            cls_fwd_positions=cls_fwd_pos,
            cls_bwd_positions=None,
            lens=lens32,
        )
        x_aug = self.final_norm(x_aug) * vm
        cls_fwd_out = self.cls_norm(cls_fwd_out)
        cls_bwd_out = self.cls_norm(cls_bwd_out)
        return x_aug, cls_fwd_out, cls_bwd_out

    def _forward_padded_static(
        self,
        x: Tensor,
        cu_seqlens: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Packed input -> compacted static rows -> quadratic-dual stack.

        Returns the same layout as :meth:`_forward_packed`:
        ``(1, total_L, D)`` sequence output (CLS stripped) + ``(B, 2*D)``
        pooled CLS summary — the rest of the model is unchanged.
        """
        from .mamba_short import (
            build_static_aux,
            packed_to_padded_static,
            padded_static_to_packed,
        )

        aux = build_static_aux(cu_seqlens)
        x_pad, row, col = packed_to_padded_static(x, cu_seqlens, aux)

        B, S, D = x_pad.shape
        # Grad-safe CLS insertion: the CLS slots (columns 0 and Lr+1) are
        # zero in x_pad, so a one-hot add places the tokens functionally —
        # never an in-place write (would sever CLS-token gradients).
        p = torch.arange(S, device=x_pad.device)
        onehot_bwd = (p == 0).to(x_pad.dtype).view(1, S, 1)
        onehot_fwd = (p.unsqueeze(0) == aux["cls_fwd_pos"].unsqueeze(1)).to(
            x_pad.dtype
        ).unsqueeze(-1)
        x_aug = (
            x_pad
            + onehot_bwd * self.cls_bwd.to(x_pad.dtype)
            + onehot_fwd * self.cls_fwd.to(x_pad.dtype)
        )

        x_aug, cls_fwd_out, cls_bwd_out = self._static_core_fn(
            x_aug, aux["valid"], aux["flip_idx"], aux["cls_fwd_pos"]
        )

        # Strip CLS: hits live at (row, col) — gather back to packed layout.
        x_hits = padded_static_to_packed(x_aug, row, col)

        cls_concat = torch.cat([cls_fwd_out, cls_bwd_out], dim=-1)
        # DDP unused-parameter tie (numerically a no-op) — same as packed path.
        cls_concat = cls_concat + 0.0 * x_hits.float().sum()
        return x_hits, cls_concat

    def _packed_core(
        self,
        x_aug: Tensor,
        aug_seq_idx: Tensor,
        flip_indices: Tensor,
        aug_cu: Tensor,
        cls_fwd_positions: Tensor,
        cls_bwd_positions: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Layer stack on the packed augmented stream — compilable with
        dynamic shapes (v5p+compile: `enable_packed_compile`)."""
        for layer in self.layers:
            x_aug = layer(
                x_aug, seq_idx=aug_seq_idx, flip_indices=flip_indices,
                cu_seqlens=aug_cu,
            )
        x_aug, cls_fwd_out, cls_bwd_out = self.final_layer(
            x_aug,
            seq_idx=aug_seq_idx,
            flip_indices=flip_indices,
            cls_fwd_positions=cls_fwd_positions,
            cls_bwd_positions=cls_bwd_positions,
            cu_seqlens=aug_cu,
        )
        x_aug = self.final_norm(x_aug)
        cls_fwd_out = self.cls_norm(cls_fwd_out)
        cls_bwd_out = self.cls_norm(cls_bwd_out)
        return x_aug, cls_fwd_out, cls_bwd_out

    def enable_packed_compile(self) -> None:
        """Compile the packed layer stack with dynamic sequence length."""
        self._packed_core_fn = torch.compile(self._packed_core, dynamic=True)

    def _forward_packed(
        self,
        x: Tensor,
        seq_idx: Tensor | None,
        cu_seqlens: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Packed-batch forward path. See :meth:`forward` for layout."""
        if x.dim() != 3 or x.shape[0] != 1:
            raise ValueError(
                f"packed forward requires x of shape (1, total_L, D); got {tuple(x.shape)}"
            )
        cu = cu_seqlens.to(torch.long)
        B = cu.shape[0] - 1
        total_L = x.shape[1]
        D = x.shape[2]
        # The consistency check is a host sync — skip it while a CUDA graph is
        # being captured (capture forbids .item(); the capture harness pads the
        # stream to a static total_L with dummy segments, so cu[-1] == total_L
        # holds by construction there).
        if not (x.is_cuda and torch.cuda.is_current_stream_capturing()):
            if int(cu[-1].item()) != total_L:
                raise ValueError(
                    f"cu_seqlens[-1]={int(cu[-1].item())} disagrees with total_L={total_L}"
                )

        device = x.device
        seg_arange = torch.arange(B, device=device, dtype=torch.long)

        # Augmented positions. Each segment gains a cls_bwd at its start
        # and a cls_fwd at its end:
        #   cls_bwd_pos[s]      = cu[s]   + 2*s
        #   original token pos  = orig_p  + 2*seg(orig_p) + 1
        #   cls_fwd_pos[s]      = cu[s+1] + 2*s + 1
        cls_bwd_positions = cu[:-1] + 2 * seg_arange           # (B,)
        cls_fwd_positions = cu[1:]  + 2 * seg_arange + 1       # (B,)

        if seq_idx is None:
            # Derive per-token segment IDs from cu_seqlens. ``right=True`` so
            # a position equal to a segment boundary maps to the *next*
            # segment (half-open interval convention).
            arange_total = torch.arange(total_L, device=device, dtype=torch.long)
            seq_idx_flat = torch.bucketize(arange_total, cu[1:], right=True)
        else:
            seq_idx_flat = seq_idx[0].to(torch.long)

        arange_total = torch.arange(total_L, device=device, dtype=torch.long)
        aug_token_positions = arange_total + 2 * seq_idx_flat + 1  # (total_L,)

        aug_total = total_L + 2 * B

        # Build augmented sequence and seq_idx via a single argsort-based
        # interleave. This keeps every operation differentiable and avoids
        # in-place writes into a fresh tensor (which would leak grad).
        cls_bwd_tok = self.cls_bwd[0, 0].to(dtype=x.dtype)  # (D,)
        cls_fwd_tok = self.cls_fwd[0, 0].to(dtype=x.dtype)  # (D,)

        all_values = torch.cat(
            [
                x[0],                                            # (total_L, D)
                cls_bwd_tok.unsqueeze(0).expand(B, D),            # (B, D)
                cls_fwd_tok.unsqueeze(0).expand(B, D),            # (B, D)
            ],
            dim=0,
        )  # (aug_total, D)
        all_positions = torch.cat(
            [aug_token_positions, cls_bwd_positions, cls_fwd_positions], dim=0
        )  # (aug_total,)
        # `all_positions` is a permutation of [0, aug_total). The inverse
        # permutation places each value at its target position.
        inv_perm = torch.argsort(all_positions)
        x_aug = all_values[inv_perm].unsqueeze(0).contiguous()  # (1, aug_total, D)

        all_seq_ids = torch.cat(
            [seq_idx_flat, seg_arange, seg_arange], dim=0
        ).to(torch.int32)
        aug_seq_idx = all_seq_ids[inv_perm].unsqueeze(0).contiguous()  # (1, aug_total)

        # Augmented cu_seqlens (each segment grew by 2) for the segment-flip helper.
        # Built with pure device ops (no host-scalar writes): ``aug_cu[0] = 0``
        # is a synchronizing H2D memcpy that breaks CUDA-graph capture.
        seg_lens_aug = (cu[1:] - cu[:-1]) + 2  # (B,) long
        aug_cu = torch.nn.functional.pad(torch.cumsum(seg_lens_aug, dim=0), (1, 0))

        flip_indices = _segment_flip_indices(aug_cu, aug_total)

        # Intermediate layers — CLS tokens flow through unchanged, and the
        # backward Mamba uses segment-wise flip via `flip_indices` (or, on
        # the v5p fused path, in-kernel per-segment flips via `aug_cu`).
        core_fn = getattr(self, "_packed_core_fn", None) or self._packed_core
        x_aug, cls_fwd_out, cls_bwd_out = core_fn(
            x_aug, aug_seq_idx, flip_indices, aug_cu,
            cls_fwd_positions, cls_bwd_positions,
        )

        # Strip CLS tokens by gathering only original-hit positions.
        x_hits = x_aug[0:1, aug_token_positions, :]  # (1, total_L, D)

        cls_concat = torch.cat([cls_fwd_out, cls_bwd_out], dim=-1)  # (B, 2*D)

        # Same DDP unused-parameter tie as in the padded path — training only:
        # the global sum makes every track's pooled vector NaN if ANY token in
        # the packed stream is NaN (e.g. inert dummy-pad segments at inference),
        # and at inference the autograd tie is meaningless anyway.
        if self.training:
            cls_concat = cls_concat + 0.0 * x_hits.float().sum()

        return x_hits, cls_concat
