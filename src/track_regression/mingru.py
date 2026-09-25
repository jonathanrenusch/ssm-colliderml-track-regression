"""Bidirectional minGRU encoder and the non-selective diagonal linear RNN.

minGRU (Feng et al. 2024, "Were RNNs All We Needed?") removes the recurrent
state from the GRU gates:

    z_t = sigmoid(W_z x_t),   n_t = W_n x_t,   h_t = (1 - z_t) * h_{t-1} + z_t * n_t

i.e. a linear, elementwise first-order recurrence ``h_t = a_t h_{t-1} + b_t``
with ``a_t = 1 - z_t`` in (0, 1) and ``b_t = z_t n_t``.  ``a_t`` and ``b_t``
for every hit come from one GEMM over the packed stream, and the scan itself
is elementwise, so a fused kernel keeps ``h`` in registers
(:mod:`track_regression.ops.mingru_short_triton`).

Each layer runs a forward and a reverse scan over the track; the terminal
states of the two directions (the forward state at the last hit and the
reverse state at the first hit) of the last layer are concatenated,
RMS-normalised and projected to the pooled output.

The diagonal linear RNN (:class:`DiagRNNCLSEncoder`) is the same encoder with
an input-*independent* decay ``a`` (a learned per-channel constant): the
non-selective ablation arm.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from track_regression.layout import fused_kernels_enabled, packed_to_padded


# ---------------------------------------------------------------------------
# scan (training / reference path)
# ---------------------------------------------------------------------------


def _hillis_steele(a: Tensor, b: Tensor) -> Tensor:
    """Inclusive scan of ``h_t = a_t h_{t-1} + b_t`` over dim 1 of ``(B, L, D)``.

    Affine maps compose, ``(a1, b1) o (a2, b2) = (a1 a2, a1 b2 + b1)``, so
    ``ceil(log2 L)`` elementwise rounds suffice (5 at L = 20).
    """
    L = a.shape[1]
    d = 1
    while d < L:
        a_sh = torch.nn.functional.pad(a[:, :-d], (0, 0, d, 0), value=1.0)
        b_sh = torch.nn.functional.pad(b[:, :-d], (0, 0, d, 0), value=0.0)
        b = b + a * b_sh
        a = a * a_sh
        d *= 2
    return b


class _ScanFn(torch.autograd.Function):
    """Linear scan with an explicit adjoint.

    The adjoint of a linear recurrence is itself a linear recurrence,
    ``g_t = dL/dh_t + a_{t+1} g_{t+1}`` (reverse scan), with
    ``dL/db_t = g_t`` and ``dL/da_t = g_t h_{t-1}`` -- so only ``a`` and ``h``
    are stored instead of the whole Hillis-Steele graph.
    """

    @staticmethod
    def forward(ctx, a: Tensor, b: Tensor) -> Tensor:
        h = _hillis_steele(a, b)
        ctx.save_for_backward(a, h)
        return h

    @staticmethod
    def backward(ctx, grad_h: Tensor):
        a, h = ctx.saved_tensors
        a_shift = torch.cat([a[:, 1:], a.new_zeros(a.shape[0], 1, a.shape[2])], dim=1)
        g = _hillis_steele(a_shift.flip(1), grad_h.flip(1)).flip(1)
        h_prev = torch.cat([h.new_zeros(h.shape[0], 1, h.shape[2]), h[:, :-1]], dim=1)
        return g * h_prev, g


def mingru_scan(a: Tensor, b: Tensor) -> Tensor:
    """``h_t = a_t h_{t-1} + b_t`` over dim 1 (differentiable)."""
    return _ScanFn.apply(a, b)


def mingru_scan_ref(a: Tensor, b: Tensor, reverse: bool = False) -> Tensor:
    """Sequential reference of the scan (tests only)."""
    B, L, D = a.shape
    h = a.new_zeros(B, D)
    out = []
    rng = range(L - 1, -1, -1) if reverse else range(L)
    for t in rng:
        h = a[:, t] * h + b[:, t]
        out.append(h)
    if reverse:
        out = out[::-1]
    return torch.stack(out, dim=1)


def _terminal(hf: Tensor, hb: Tensor, lens: Tensor) -> Tensor:
    """Forward state at the last hit and reverse state at the first hit."""
    last = (lens - 1).clamp(min=0)
    return torch.cat([hf[torch.arange(hf.shape[0], device=hf.device), last], hb[:, 0]], dim=-1)


# ---------------------------------------------------------------------------
# layers
# ---------------------------------------------------------------------------


class _MinGRULayer(nn.Module):
    """One bidirectional minGRU layer.

    A single ``in_proj`` produces both directions' ``[z | n]`` in one GEMM,
    laid out ``[z_fwd | n_fwd | z_bwd | n_bwd]``; the fused kernel addresses
    each direction by a channel offset.
    """

    def __init__(self, in_dim: int, hidden: int):
        super().__init__()
        self.hidden = int(hidden)
        self._core = self._core_eager
        self.in_proj = nn.Linear(int(in_dim), 4 * self.hidden)
        with torch.no_grad():
            self.in_proj.bias.zero_()        # z ~ 0.5 at init: neither saturates nor forgets

    def enable_compiled_core(self) -> None:
        self._core = torch.compile(self._core_eager, dynamic=False)

    def _core_eager(self, x: Tensor, valid: Tensor):
        """``x``: (B, S, in_dim) padded, ``valid``: (B, S).  A zero gate at the
        pads makes them the identity map (a = 1, b = 0), so the reverse
        direction is a plain flip of the whole padded row."""
        H = self.hidden
        zn = self.in_proj(x)
        z_f, n_f, z_b, n_b = zn.split(H, dim=-1)
        v = valid.unsqueeze(-1).to(zn.dtype)
        # the recurrence (a cumulative product over the track) always runs in fp32
        zf = torch.sigmoid(z_f.float()) * v.float()
        zb = torch.sigmoid(z_b.float()) * v.float()
        n_f, n_b = n_f.float(), n_b.float()
        B = x.shape[0]
        # both directions in one scan, stacked along the batch axis
        a = torch.cat([1.0 - zf, (1.0 - zb).flip(1)], dim=0)
        b = torch.cat([zf * n_f, (zb * n_b).flip(1)], dim=0)
        with torch.autocast("cuda", enabled=False):
            h = mingru_scan(a, b)
        h = h.to(x.dtype)
        hf, hb = h[:B], h[B:].flip(1)
        return torch.cat([hf, hb], dim=-1), hf, hb

    def forward(self, x: Tensor, lens: Tensor) -> tuple[Tensor, Tensor]:
        """``x``: (B, S, in_dim) padded -> ``(seq (B, S, 2H), terminal (B, 2H))``."""
        p = torch.arange(x.shape[1], device=x.device)
        seq, hf, hb = self._core(x, p.unsqueeze(0) < lens.unsqueeze(1))
        return seq, _terminal(hf, hb, lens)


class _DiagRNNLayer(nn.Module):
    """``h_t = a * h_{t-1} + b_t`` with ``a`` a learned per-channel constant
    (no gate: the projection emits only ``b``, ``D -> 2H``)."""

    def __init__(self, in_dim: int, hidden: int):
        super().__init__()
        self.hidden = int(hidden)
        self.in_proj = nn.Linear(int(in_dim), 2 * self.hidden)
        # a = sigmoid(logit), spread over (0.5, 0.99): a range of memory horizons
        self.decay_logit = nn.Parameter(torch.logit(torch.linspace(0.5, 0.99, 2 * self.hidden)))
        with torch.no_grad():
            self.in_proj.bias.zero_()

    def forward(self, x: Tensor, lens: Tensor) -> tuple[Tensor, Tensor]:
        H = self.hidden
        B, S = x.shape[0], x.shape[1]
        p = torch.arange(S, device=x.device)
        v = (p.unsqueeze(0) < lens.unsqueeze(1)).unsqueeze(-1).to(x.dtype)
        b_f, b_b = (self.in_proj(x) * v).split(H, dim=-1)
        a = torch.sigmoid(self.decay_logit)
        one = torch.ones(1, device=x.device, dtype=x.dtype)
        # pads are the identity map (a = 1, b = 0) so the flip trick holds
        af = torch.where(v.bool(), a[:H].view(1, 1, H).expand(B, S, H), one)
        ab = torch.where(v.bool(), a[H:].view(1, 1, H).expand(B, S, H), one)
        h = mingru_scan(torch.cat([af, ab.flip(1)], 0), torch.cat([b_f, b_b.flip(1)], 0))
        hf, hb = h[:B], h[B:].flip(1)
        return torch.cat([hf, hb], dim=-1), _terminal(hf, hb, lens)


# ---------------------------------------------------------------------------
# encoders
# ---------------------------------------------------------------------------


class MinGRUCLSEncoder(nn.Module):
    """Bidirectional minGRU encoder with a terminal-state readout.

    Returns ``(hit_output, pooled (B, pool_out_dim))`` for a packed batch.
    Training and the reference inference path pad the batch to the static
    length ``max_len`` (one compiled graph); the fused inference path runs the
    packed Triton scan with no padding anywhere.
    """

    _layer_cls = _MinGRULayer
    _has_fused_kernel = True

    def __init__(
        self,
        dim: int,
        hidden_size: int = 192,
        num_layers: int = 2,
        pool_out_dim: int = 256,
        max_len: int = 20,
        compile_core: bool = True,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.hidden = int(hidden_size)
        self.num_layers = int(num_layers)
        self._pool_out_dim = int(pool_out_dim)
        self.max_len = int(max_len)
        dims = [self.dim] + [2 * self.hidden] * (self.num_layers - 1)
        self.layers = nn.ModuleList([self._layer_cls(d, self.hidden) for d in dims])
        if compile_core:
            for layer in self.layers:
                layer.enable_compiled_core()
        self.pool_norm = nn.RMSNorm(2 * self.hidden)
        self.pool_proj = nn.Linear(2 * self.hidden, self._pool_out_dim)

    @property
    def pool_dim(self) -> int:
        return self._pool_out_dim

    def forward(self, x: Tensor, cu_seqlens: Tensor, seq_idx: Tensor | None = None):  # noqa: ARG002
        if (self._has_fused_kernel and x.is_cuda and not torch.is_grad_enabled()
                and fused_kernels_enabled()):
            from track_regression.ops.mingru_short_triton import mingru_bidi_packed
            cu = cu_seqlens.to(torch.int32)
            h = x[0]
            H = self.hidden
            for layer in self.layers:
                # the kernel reads fp32/fp16/bf16 and accumulates the scan in fp32
                h = mingru_bidi_packed(layer.in_proj(h).contiguous(), cu, H, self.max_len)
            term = torch.cat([h[cu[1:].long() - 1, :H], h[cu[:-1].long(), H:]], dim=-1)
            return h.unsqueeze(0), self.pool_proj(self.pool_norm(term))

        x_pad, row, pos, lens = packed_to_padded(x, cu_seqlens)
        S = x_pad.shape[1]
        if S > self.max_len:
            raise ValueError(f"track longer than max_len={self.max_len}: {S}")
        h = torch.cat([x_pad, x_pad.new_zeros(x_pad.shape[0], self.max_len - S, x_pad.shape[2])], dim=1)
        term = None
        for layer in self.layers:
            h, term = layer(h, lens)
        pooled = self.pool_proj(self.pool_norm(term))
        seq_out = h[row, pos].unsqueeze(0)
        if self.training:
            # keep every parameter in the autograd graph (DDP), numerically a no-op
            pooled = pooled + 0.0 * seq_out.float().sum()
        return seq_out, pooled


class DiagRNNCLSEncoder(MinGRUCLSEncoder):
    """:class:`MinGRUCLSEncoder` with :class:`_DiagRNNLayer` blocks
    (non-selective ablation arm; no fused kernel, always the eager path)."""

    _layer_cls = _DiagRNNLayer
    _has_fused_kernel = False

    def __init__(self, dim: int, hidden_size: int = 270, num_layers: int = 2,
                 pool_out_dim: int = 256, max_len: int = 20) -> None:
        super().__init__(dim=dim, hidden_size=hidden_size, num_layers=num_layers,
                         pool_out_dim=pool_out_dim, max_len=max_len, compile_core=False)

    def forward(self, x: Tensor, cu_seqlens: Tensor, seq_idx: Tensor | None = None):
        # always fp32: the un-normalised state can exceed the fp16 range
        with torch.autocast("cuda", enabled=False):
            return super().forward(x.float(), cu_seqlens, seq_idx)
