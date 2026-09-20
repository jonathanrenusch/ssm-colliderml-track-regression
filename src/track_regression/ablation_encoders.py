"""Encoders for the 2026-09 cross-architecture ablation study.

Purely **additive**: nothing in this module is imported by the production
model path.  Every class here honours the encoder contract that
:class:`track_regression.model.TrackParameterRegressor` expects::

    forward(x, x_sort_value=None, seq_idx=None, cu_seqlens=None,
            kv_mask=None, **kwargs) -> (sequence_output, pooled)

and exposes ``pool_dim``.  With ``pool='register_token'`` and
``pool_dim == 256`` the regressor builds *exactly* the paper model's head
stack — ``pool_head = Dense(256 -> 128, hidden [128])`` followed by
``output_head(128 -> 35)`` — so the readout is identical to the
bidirectional Mamba-2 baseline down to the parameter.

Two packed-batch conventions are honoured throughout:

* ``x`` is ``(1, total_L, D)`` with ``cu_seqlens`` the cumulative segment
  boundaries; the collate has already ordered the hits inside each segment
  (true-time order) and the encoder must **not** re-sort.
* the returned ``sequence_output`` is in the same packed layout and is
  discarded by the regressor; it exists so that the DDP unused-parameter
  tie can pull every encoder weight into the autograd graph.

The DDP tie is applied **in training mode only** — a global sum over the
packed stream turns every track's pooled vector into NaN if any token in
the batch is NaN, which bit the campaign at inference time
(CLAUDE.md §4.27).
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from track_regression.transformer_encoder import EncoderWithCLS


# ---------------------------------------------------------------------------
# packed <-> padded helpers (shared)
# ---------------------------------------------------------------------------


def _packed_to_padded(x: Tensor, cu_seqlens: Tensor, extra_slots: int = 0):
    """``(1, total_L, D)`` -> ``(B, max_len + extra_slots, D)`` zero-padded.

    Returns ``(x_pad, row, pos, lens)`` where ``row``/``pos`` are the
    scatter indices of the original tokens, so the inverse gather is
    ``x_pad[row, pos]``.  Pads strictly trail the valid prefix of every
    row, which is what makes a *causal* forward scan pad-safe without a
    mask (Scheme A, mamba_short.py).
    """
    if x.dim() != 3 or x.shape[0] != 1:
        raise ValueError(
            f"packed forward requires x of shape (1, total_L, D); got {tuple(x.shape)}"
        )
    cu = cu_seqlens.to(torch.long)
    B = cu.numel() - 1
    total_L, D = x.shape[1], x.shape[2]
    lens = cu[1:] - cu[:-1]
    max_len = int(lens.max().item())
    arange = torch.arange(total_L, device=x.device, dtype=torch.long)
    seg = torch.bucketize(arange, cu[1:], right=True)
    pos = arange - cu[seg]
    row = seg
    x_pad = x.new_zeros(B, max_len + extra_slots, D)
    x_pad[row, pos] = x[0]
    return x_pad, row, pos, lens


# ---------------------------------------------------------------------------
# 1. Bidirectional GRU
# ---------------------------------------------------------------------------


class BiGRUCLSEncoder(nn.Module):
    """Bidirectional multi-layer GRU with a final-hidden-state readout.

    The recurrent analogue of the bidirectional Mamba-2 baseline: the pooled
    vector is ``concat(h_fwd_last_layer, h_bwd_last_layer)``, i.e. the two
    terminal states of the last layer — the same "state at each scan
    terminus" readout the SSM's two CLS tokens implement, without the
    learned tokens.

    ``hidden_size`` is the trunk-matching knob (the study targets a trunk of
    ~628 k parameters; ``hidden_size=120`` lands there).  The pooled vector
    ``(B, 2 * hidden_size)`` is projected to ``pool_out_dim`` so the
    downstream head stack is byte-identical to the other architectures.

    The cuDNN fused GRU kernel is used as-is (the GRU's best available
    kernel); packed input is converted to the padded + ``pack_padded_sequence``
    layout internally, which is exactly how cuDNN wants it.
    """

    def __init__(
        self,
        dim: int,
        hidden_size: int = 120,
        num_layers: int = 2,
        pool_out_dim: int = 256,
        dropout: float = 0.0,
        norm: str = "RMSNorm",
    ) -> None:
        super().__init__()
        if dropout:
            raise ValueError("the ablation protocol forbids dropout")
        self.dim = int(dim)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self._pool_out_dim = int(pool_out_dim)

        self.gru = nn.GRU(
            input_size=self.dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=True,
            bias=True,
            dropout=0.0,
        )
        # Normalise the terminal states before the head, mirroring the SSM
        # encoder's ``cls_norm`` (raw recurrent states have unconstrained
        # magnitude and spike the head gradients otherwise).
        pooled_in = 2 * self.hidden_size
        if norm == "RMSNorm":
            self.pool_norm: nn.Module = nn.RMSNorm(pooled_in)
        elif norm == "LayerNorm":
            self.pool_norm = nn.LayerNorm(pooled_in)
        else:
            self.pool_norm = nn.Identity()
        self.pool_proj: nn.Module = (
            nn.Identity()
            if pooled_in == self._pool_out_dim
            else nn.Linear(pooled_in, self._pool_out_dim)
        )

    @property
    def pool_dim(self) -> int:
        return self._pool_out_dim

    # -- readout ----------------------------------------------------------
    def _readout(self, h_n: Tensor) -> Tensor:
        """``h_n`` is ``(num_layers * 2, B, H)`` -> ``(B, 2H)`` last layer."""
        return torch.cat([h_n[-2], h_n[-1]], dim=-1)

    def _pool(self, pooled: Tensor) -> Tensor:
        return self.pool_proj(self.pool_norm(pooled))

    # -- forward ----------------------------------------------------------
    def forward(
        self,
        x: Tensor,
        x_sort_value: Tensor | None = None,  # noqa: ARG002 — segments are pre-sorted
        seq_idx: Tensor | None = None,  # noqa: ARG002
        cu_seqlens: Tensor | None = None,
        kv_mask: Tensor | None = None,
        **kwargs,  # noqa: ARG002 — API parity with the other encoders
    ) -> tuple[Tensor, Tensor]:
        if cu_seqlens is not None:
            return self._forward_packed(x, cu_seqlens)
        return self._forward_padded(x, kv_mask)

    def _run(self, x_pad: Tensor, lens: Tensor) -> tuple[Tensor, Tensor]:
        max_len = x_pad.shape[1]
        packed = pack_padded_sequence(
            x_pad, lens.to("cpu", torch.int64), batch_first=True, enforce_sorted=False
        )
        out, h_n = self.gru(packed)
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=max_len)
        return out, h_n

    def _forward_packed(self, x: Tensor, cu_seqlens: Tensor):
        x_pad, row, pos, lens = _packed_to_padded(x, cu_seqlens)
        out, h_n = self._run(x_pad, lens)
        pooled = self._pool(self._readout(h_n))
        seq_out = out[row, pos].unsqueeze(0)  # (1, total_L, 2H)
        if self.training:
            pooled = pooled + 0.0 * seq_out.float().sum()
        return seq_out, pooled

    def _forward_padded(self, x: Tensor, kv_mask: Tensor | None):
        B, N, _ = x.shape
        if kv_mask is None:
            lens = torch.full((B,), N, dtype=torch.long, device=x.device)
        else:
            lens = kv_mask.to(torch.long).sum(-1).clamp(min=1)
        out, h_n = self._run(x, lens)
        pooled = self._pool(self._readout(h_n))
        if self.training:
            pooled = pooled + 0.0 * out.float().sum()
        return out, pooled


# ---------------------------------------------------------------------------
# 2. Forward-only Mamba-2
# ---------------------------------------------------------------------------


class _ForwardMambaLayer(nn.Module):
    """The paper's bidirectional block with the reverse scan and the
    direction-merge gate removed: ``x + Mamba2(norm(x))``."""

    def __init__(self, dim: int, norm: str = "RMSNorm", dropout: float = 0.0, **mamba_kwargs):
        super().__init__()
        from track_regression.mamba_short import Mamba2Short

        if norm == "RMSNorm":
            self.norm: nn.Module = nn.RMSNorm(dim)
        elif norm == "LayerNorm":
            self.norm = nn.LayerNorm(dim)
        else:
            raise ValueError(f"Unknown norm: {norm}")
        self.forward_mamba = Mamba2Short(d_model=dim, **mamba_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x_norm = self.norm(x).contiguous()
        y = self.forward_mamba(x_norm)
        return x + self.dropout(y), y


class ForwardMambaCLSEncoder(nn.Module):
    """Unidirectional Mamba-2 encoder with a single learned CLS token.

    Bidirectionality ablation of the paper encoder.  Structure per layer is
    the paper block with ``backward_mamba`` and the merge ``gate`` deleted;
    the learned ``cls_fwd`` token is appended after each track's last hit so
    the forward scan reads it out last, and the final layer exposes the raw
    (pre-residual) Mamba-2 output at that position — the same convention as
    :class:`track_regression.mamba_cls.BidirectionalMambaCLSFinalLayer`.

    The internal layout is the padded-static one (``[h_0..h_{L-1}, cls, PAD]``):
    every op in the block is positionwise or *causal*, so trailing pads
    cannot influence a valid output and no mask is needed (Scheme A, see
    ``mamba_short.py``).  This is the same arithmetic the bidirectional model
    trains under (``KernelSwapCallback`` variant ``auto`` -> Mamba2Short
    quadratic dual), so the two encoders are numerically comparable.

    Width is raised via ``expand`` to match the bidirectional model's trunk
    parameter count (one block per layer instead of two).
    """

    def __init__(
        self,
        num_layers: int,
        dim: int,
        d_state: int = 64,
        d_conv: int = 1,
        expand: int = 4,
        headdim: int = 32,
        ngroups: int = 1,
        chunk_size: int = 256,
        norm: str = "RMSNorm",
        dropout: float = 0.0,
        cls_init_scale: float = 0.02,
        residual_depth_init: bool = True,
        pool_out_dim: int = 256,
        compile_core: bool = False,
    ) -> None:
        super().__init__()
        assert num_layers >= 1
        self.num_layers = int(num_layers)
        self.dim = int(dim)
        self._pool_out_dim = int(pool_out_dim)

        self.cls_fwd = nn.Parameter(torch.randn(1, 1, dim) * cls_init_scale)

        common = dict(
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            headdim=headdim,
            ngroups=ngroups,
            chunk_size=chunk_size,
        )
        self.layers = nn.ModuleList(
            [_ForwardMambaLayer(dim, norm=norm, dropout=dropout, **common)
             for _ in range(self.num_layers)]
        )

        if norm == "LayerNorm":
            self.final_norm: nn.Module = nn.LayerNorm(dim)
        elif norm == "RMSNorm":
            self.final_norm = nn.RMSNorm(dim)
        else:
            self.final_norm = nn.Identity()
        self.cls_norm = nn.RMSNorm(dim)
        self.pool_proj: nn.Module = (
            nn.Identity() if dim == self._pool_out_dim else nn.Linear(dim, self._pool_out_dim)
        )

        if residual_depth_init:
            # One residual write per layer here (the bidirectional block has
            # two, through the gate), so the canonical factor is 1/sqrt(N).
            scale = 1.0 / math.sqrt(self.num_layers)
            with torch.no_grad():
                for layer in self.layers:
                    layer.forward_mamba.out_proj.weight.mul_(scale)

        self._core_fn = self._core
        if compile_core:
            self._core_fn = torch.compile(self._core, dynamic=False)

    @property
    def pool_dim(self) -> int:
        return self._pool_out_dim

    # -- core -------------------------------------------------------------
    def _core(self, x_aug: Tensor, valid: Tensor) -> tuple[Tensor, Tensor]:
        vm = valid.unsqueeze(-1).to(x_aug.dtype)
        y = x_aug
        for layer in self.layers[:-1]:
            y, _ = layer(y)
            y = y * vm
        y, raw = self.layers[-1](y)
        y = self.final_norm(y) * vm
        return y, raw

    def _encode_padded(self, x_pad: Tensor, lens: Tensor):
        """``x_pad`` already has one free slot per row for the CLS token."""
        B, S, D = x_pad.shape
        p = torch.arange(S, device=x_pad.device)
        # CLS sits immediately after the last hit of each row; the slot is
        # zero in x_pad, so a one-hot add places it without an in-place write
        # (which would sever the CLS token's gradient).
        onehot = (p.unsqueeze(0) == lens.unsqueeze(1)).to(x_pad.dtype).unsqueeze(-1)
        x_aug = x_pad + onehot * self.cls_fwd.to(x_pad.dtype)
        valid = p.unsqueeze(0) <= lens.unsqueeze(1)          # hits + the CLS slot
        y, raw = self._core_fn(x_aug, valid)
        cls_out = raw[torch.arange(B, device=x_pad.device), lens]   # (B, D)
        pooled = self.pool_proj(self.cls_norm(cls_out))
        return y, pooled

    # -- forward ----------------------------------------------------------
    def forward(
        self,
        x: Tensor,
        x_sort_value: Tensor | None = None,  # noqa: ARG002
        seq_idx: Tensor | None = None,  # noqa: ARG002
        cu_seqlens: Tensor | None = None,
        kv_mask: Tensor | None = None,
        **kwargs,  # noqa: ARG002
    ) -> tuple[Tensor, Tensor]:
        if cu_seqlens is not None:
            x_pad, row, pos, lens = _packed_to_padded(x, cu_seqlens, extra_slots=1)
            y, pooled = self._encode_padded(x_pad, lens)
            seq_out = y[row, pos].unsqueeze(0)
        else:
            B, N, D = x.shape
            if kv_mask is None:
                lens = torch.full((B,), N, dtype=torch.long, device=x.device)
            else:
                lens = kv_mask.to(torch.long).sum(-1)
            x_pad = torch.cat([x, x.new_zeros(B, 1, D)], dim=1)
            y, pooled = self._encode_padded(x_pad, lens)
            seq_out = y[:, :N, :]
        if self.training:
            pooled = pooled + 0.0 * seq_out.float().sum()
        return seq_out, pooled


# ---------------------------------------------------------------------------
# 3. MLP on the flattened track (lowest-priority baseline)
# ---------------------------------------------------------------------------


class FlatMLPEncoder(nn.Module):
    """Order-blind baseline: zero-pad every track to ``max_len`` embedded
    tokens, flatten, and run an MLP.  No token mixing beyond the first
    linear layer, so the only sequence information available is the fixed
    slot index of each hit.
    """

    def __init__(
        self,
        dim: int,
        max_len: int = 20,
        hidden_layers: tuple[int, ...] = (512, 512),
        pool_out_dim: int = 256,
        activation: str = "SiLU",
        norm: str = "RMSNorm",
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.max_len = int(max_len)
        self._pool_out_dim = int(pool_out_dim)
        act = getattr(nn, activation)
        sizes = [self.max_len * self.dim, *hidden_layers]
        blocks: list[nn.Module] = []
        for a, b in zip(sizes[:-1], sizes[1:]):
            blocks += [nn.Linear(a, b), act()]
        blocks += [nn.Linear(sizes[-1], self._pool_out_dim)]
        self.mlp = nn.Sequential(*blocks)
        self.pool_norm: nn.Module = (
            nn.RMSNorm(self._pool_out_dim) if norm == "RMSNorm" else nn.Identity()
        )

    @property
    def pool_dim(self) -> int:
        return self._pool_out_dim

    def forward(
        self,
        x: Tensor,
        x_sort_value: Tensor | None = None,  # noqa: ARG002
        seq_idx: Tensor | None = None,  # noqa: ARG002
        cu_seqlens: Tensor | None = None,
        kv_mask: Tensor | None = None,  # noqa: ARG002
        **kwargs,  # noqa: ARG002
    ) -> tuple[Tensor, Tensor]:
        if cu_seqlens is not None:
            x_pad, row, pos, lens = _packed_to_padded(x, cu_seqlens)
            if x_pad.shape[1] < self.max_len:
                x_pad = torch.cat(
                    [x_pad, x_pad.new_zeros(x_pad.shape[0], self.max_len - x_pad.shape[1], self.dim)],
                    dim=1,
                )
            x_pad = x_pad[:, : self.max_len]
            pooled = self.pool_norm(self.mlp(x_pad.reshape(x_pad.shape[0], -1)))
            seq_out = x_pad[row, pos].unsqueeze(0)
        else:
            B, N, D = x.shape
            if N < self.max_len:
                x = torch.cat([x, x.new_zeros(B, self.max_len - N, D)], dim=1)
            xin = x[:, : self.max_len]
            pooled = self.pool_norm(self.mlp(xin.reshape(B, -1)))
            seq_out = x[:, :N, :]
        if self.training:
            pooled = pooled + 0.0 * seq_out.float().sum()
        return seq_out, pooled


# ---------------------------------------------------------------------------
# 4. Transformer with an index positional encoding
# ---------------------------------------------------------------------------


class _PaddedRoutedTransformerCLS(EncoderWithCLS):
    """:class:`~track_regression.transformer_encoder.EncoderWithCLS` that
    routes **packed** batches through its own *padded* attention path.

    Why.  ``EncoderWithCLS._forward_packed`` with ``attn_type='torch'`` builds
    a dense block-diagonal mask over the whole packed stream and runs SDPA on
    it: for a 2048-track batch of <=20-hit tracks that is a
    ~31k x 31k attention matrix per head — O((B*L)^2) work and ~40 GB of
    activations for O(B*L^2) of useful arithmetic.  (The efficient packed
    route in that class is ``flash-varlen``, which is bf16/fp16 only and so is
    unavailable under this study's strict-fp32 rule.)

    Converting the packed batch to ``(B, L_max, D)`` with a key mask computes
    exactly the same function — attention is confined to a track either way —
    at a few hundredths of the cost.  ``tests/test_ablation_transformer.py``
    checks the two agree to 1e-6.

    Subclasses choose the scalar signal that drives the inherited
    Fourier positional encoding.
    """

    #: when True the padded row order is the stored hit order (no argsort)
    _identity_order = True

    def _posenc_signal(self, x_pad: Tensor, lens: Tensor, hit_time_pad: Tensor | None) -> Tensor | None:
        raise NotImplementedError

    def _forward_via_padded(self, x_pad, mask, lens, hit_time_pad):
        key = self._posenc_signal(x_pad, lens, hit_time_pad)
        return EncoderWithCLS.forward(self, x_pad, x_sort_value=key, kv_mask=mask)

    def forward(
        self,
        x: Tensor,
        x_sort_value: Tensor | None = None,
        kv_mask: Tensor | None = None,
        seq_idx: Tensor | None = None,  # noqa: ARG002
        cu_seqlens: Tensor | None = None,
        **kwargs,  # noqa: ARG002
    ) -> tuple[Tensor, Tensor]:
        if cu_seqlens is None:
            B, N = x.shape[0], x.shape[1]
            lens = (kv_mask.to(torch.long).sum(-1) if kv_mask is not None
                    else torch.full((B,), N, dtype=torch.long, device=x.device))
            mask = (kv_mask if kv_mask is not None
                    else torch.ones(B, N, dtype=torch.bool, device=x.device))
            ht = x_sort_value
            return self._forward_via_padded(x, mask, lens, ht)

        x_pad, row, pos, lens = _packed_to_padded(x, cu_seqlens)
        B, S = x_pad.shape[0], x_pad.shape[1]
        mask = torch.zeros(B, S, dtype=torch.bool, device=x.device)
        mask[row, pos] = True
        ht_pad = None
        if x_sort_value is not None:
            ht_pad = x_pad.new_zeros(B, S)
            ht_pad[row, pos] = x_sort_value[0].to(x_pad.dtype)
        hit_out, pooled = self._forward_via_padded(x_pad, mask, lens, ht_pad)
        seq_out = hit_out[row, pos].unsqueeze(0)
        return seq_out, pooled


class IndexPosEncTransformerCLS(_PaddedRoutedTransformerCLS):
    """Transformer baseline of the study: positional encoding from the
    **within-track hit index**.

    ``EncoderWithCLS`` derives its additive positional encoding from
    ``x_sort_value``, which the regressor fills with ``inputs["hit_time"]`` —
    the digitised ``tracker_hits.time`` column.  In the v3
    (post-digitization-fix) stores that column is **identically zero for
    every hit of every dataset** (measured 2026-09-14; in v2 it was zero on
    strips only).  A constant posenc makes the transformer exactly
    permutation-invariant in the hits, so it could not use the true-time hit
    ordering the recurrent and state-space baselines scan over — an accidental
    handicap, not an architectural property.

    This class changes *only* the source of the scalar posenc signal to
    ``0, 1, 2, ...`` within each track, i.e. the position in the stored
    (true-time) order: exactly the ordering information the scanning encoders
    consume, no more.  The Fourier ladder, its base, the projection, the
    attention and every other hyper-parameter are inherited unchanged.
    """

    def _posenc_signal(self, x_pad, lens, hit_time_pad):  # noqa: ARG002
        B, S = x_pad.shape[0], x_pad.shape[1]
        return (torch.arange(S, device=x_pad.device, dtype=x_pad.dtype)
                .unsqueeze(0).expand(B, S))


class TimePosEncTransformerCLS(_PaddedRoutedTransformerCLS):
    """Literal-protocol transformer: the stock hit-time posenc.  On v3 data
    ``hit_time`` is identically zero, so this is the **order-blind**
    (bag-of-hits) transformer.  Diagnostic arm only."""

    def _posenc_signal(self, x_pad, lens, hit_time_pad):  # noqa: ARG002
        if hit_time_pad is None:
            B, S = x_pad.shape[0], x_pad.shape[1]
            return x_pad.new_zeros(B, S)
        return hit_time_pad
