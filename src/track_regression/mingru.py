"""minGRU — a *linear-recurrent* GRU for the short-sequence track-fitting task.

Why this module exists
----------------------
The 2026-09 architecture ablation found that a bidirectional GRU matches the
bidirectional Mamba-2 on physics (geometry identical to <0.005, q/p slightly
better at low pT and on hadrons).  What it cannot match is the *kernel*: a
classical GRU's reset gate makes h_t nonlinear in h_{t-1}, so every one of the
L timesteps is a dependent H x H GEMM and no amount of kernel work removes the
sequential chain.

minGRU (Feng et al. 2024, "Were RNNs All We Needed?") drops the h_{t-1}
dependence inside the gates:

    z_t = sigmoid(W_z x_t)                       (input only)
    n_t = W_n x_t                                (input only)
    h_t = (1 - z_t) * h_{t-1} + z_t * n_t

which is a **linear, elementwise first-order recurrence**

    h_t = a_t * h_{t-1} + b_t,  a_t = 1 - z_t in (0,1),  b_t = z_t * n_t.

Two consequences, both decisive here:

* ``a_t`` and ``b_t`` for *every* token come from ONE GEMM over the whole
  packed stream — perfectly parallel, no sequential dependency.
* the recurrence itself is elementwise, so a fused kernel keeps ``h`` in
  registers and the L steps cost L fused multiply-adds per channel instead of
  L matrix products.  At L <= 20 that is essentially free, and the work is
  embarrassingly parallel over (track x channel).

Because ``a_t`` in (0,1), the scan is unconditionally stable and needs none of
the log-space machinery the original paper uses for long sequences.

The layer/readout structure mirrors :class:`BiGRUCLSEncoder` exactly
(bidirectional, terminal-hidden-state readout, RMSNorm + projection to the
shared 256-d pool), so the comparison isolates the recurrence.
"""

from __future__ import annotations

import os

import torch
from torch import Tensor, nn


# ---------------------------------------------------------------------------
# reference scan (training path; the Triton kernel must match this exactly)
# ---------------------------------------------------------------------------


class _ScanFn(torch.autograd.Function):
    """Linear scan ``h_t = a_t h_{t-1} + b_t`` with an EXPLICIT adjoint.

    The pure-PyTorch "kernel" for the training path.  Letting autograd
    differentiate through the Hillis-Steele rounds stores ~2 x rounds
    intermediates of shape (B, L, D) and replays them backwards; the adjoint
    of a linear recurrence is itself a linear recurrence, so we can compute it
    in one reverse scan from ``a`` and ``h`` alone:

        g_t = dL/dh_t + a_{t+1} g_{t+1}      (reverse scan)
        dL/db_t = g_t
        dL/da_t = g_t * h_{t-1}

    That saves the whole Hillis-Steele graph: two tensors are stashed instead
    of ~10, and backward costs one scan instead of replaying five rounds.
    """

    @staticmethod
    def forward(ctx, a: Tensor, b: Tensor) -> Tensor:
        h = _hillis_steele(a, b)
        ctx.save_for_backward(a, h)
        return h

    @staticmethod
    def backward(ctx, grad_h: Tensor):
        a, h = ctx.saved_tensors
        L = a.shape[1]
        # reverse scan for the adjoint: g_t = grad_h_t + a_{t+1} g_{t+1}
        a_shift = torch.cat([a[:, 1:], a.new_zeros(a.shape[0], 1, a.shape[2])], dim=1)
        g = _hillis_steele(a_shift.flip(1), grad_h.flip(1)).flip(1)
        # dL/da_t = g_t * h_{t-1}   (h_{-1} = 0)
        h_prev = torch.cat([h.new_zeros(h.shape[0], 1, h.shape[2]), h[:, :-1]], dim=1)
        return g * h_prev, g


def mingru_scan_autograd(a: Tensor, b: Tensor) -> Tensor:
    """Scan with the explicit adjoint (training path)."""
    return _ScanFn.apply(a, b)


def _hillis_steele(a: Tensor, b: Tensor) -> Tensor:
    """Inclusive scan of the affine maps, no autograd bookkeeping intended."""
    L = a.shape[1]
    d = 1
    while d < L:
        a_sh = torch.nn.functional.pad(a[:, :-d], (0, 0, d, 0), value=1.0)
        b_sh = torch.nn.functional.pad(b[:, :-d], (0, 0, d, 0), value=0.0)
        b = b + a * b_sh
        a = a * a_sh
        d *= 2
    return b


def mingru_scan_parallel(a: Tensor, b: Tensor) -> Tensor:
    """Same recurrence as :func:`mingru_scan_ref`, by Hillis-Steele prefix scan.

    ``h_t = a_t h_{t-1} + b_t`` is an affine map, and affine maps compose:
    ``(a1,b1) o (a2,b2) = (a1 a2, a1 b2 + b1)``.  After ``k`` rounds the pair at
    position ``t`` represents the map from ``h_{t-2^k}`` to ``h_t``, so
    ``ceil(log2 L)`` rounds suffice — **5 rounds at L = 20 instead of 20
    sequential steps**, and each round is two elementwise multiplies over the
    whole batch.  This is what makes the training path parallel in the
    sequence dimension as well (and keeps the autograd graph 4x shorter).
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


def mingru_scan_ref(a: Tensor, b: Tensor, reverse: bool = False) -> Tensor:
    """``h_t = a_t * h_{t-1} + b_t`` over dim=1 of ``(B, L, D)``.

    ``reverse`` runs the scan from the end of the sequence.  Pads must trail
    the valid prefix; the recurrence is causal so they cannot influence a
    valid output in the forward direction, and the caller flips the valid
    prefix for the backward one.
    """
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


class _MinGRULayer(nn.Module):
    """One bidirectional minGRU layer.

    A single ``in_proj`` produces both directions' ``[z | n]`` in one GEMM
    (the two directions read the same input), laid out as
    ``[z_fwd | n_fwd | z_bwd | n_bwd]`` so the fused kernel can address its
    direction by a channel offset instead of taking a slice — the pattern
    ``ssd_short_triton`` uses (``ZX_OFF``), which avoids Inductor
    materialising a copy per opaque-op input (kernel campaign, night 1).
    """

    def __init__(self, in_dim: int, hidden: int):
        super().__init__()
        self.in_dim = int(in_dim)
        self.hidden = int(hidden)
        self._core = self._core_eager
        self.in_proj = nn.Linear(self.in_dim, 4 * self.hidden)
        # Start with a gate bias that keeps roughly half the previous state:
        # z ~ 0.5 means the scan neither saturates nor forgets at init.
        with torch.no_grad():
            self.in_proj.bias.zero_()

    def enable_compiled_core(self) -> None:
        """Compile the scan core (static shapes).  The training-path analogue
        of the Mamba campaign's ``v3c``: the 20-step loop becomes a handful of
        fused kernels instead of ~80 eager launches per layer-direction."""
        self._core = torch.compile(self._core_eager, dynamic=False)

    def _core_eager(self, x: Tensor, valid: Tensor):
        """``valid``: (B, S) bool.  Zeroing the gate at pads makes them the
        identity map (a = 1, b = 0), so the backward direction is a plain
        ``flip`` of the whole padded row — no per-row gather needed."""
        H = self.hidden
        zn = self.in_proj(x)
        z_f, n_f, z_b, n_b = zn.split(H, dim=-1)
        v = valid.unsqueeze(-1).to(zn.dtype)
        # --- precision boundary (fp16 training) --------------------------
        # fp16 and TF32 both carry 10 mantissa bits, and TF32 is validated at
        # <=0.3 % drift, so fp16 GEMMs are precise enough.  What fp16 lacks is
        # RANGE (5 exponent bits).  The recurrence is a cumulative product over
        # the track, which is exactly where a short exponent bites -- and it is
        # 0.25 % of the FLOPs, so keeping it in fp32 costs nothing.  Matches the
        # measured fp16 "partial" (fine) vs "full" (broken) inference split.
        zf = torch.sigmoid(z_f.float()) * v.float()
        zb = torch.sigmoid(z_b.float()) * v.float()
        n_f, n_b = n_f.float(), n_b.float()
        # Both directions in ONE scan: the backward direction is the same
        # recurrence on the flipped row (pads are the identity map), so stack
        # them along the batch axis and halve the number of launches.
        B = x.shape[0]
        a = torch.cat([1.0 - zf, (1.0 - zb).flip(1)], dim=0)
        b = torch.cat([zf * n_f, (zb * n_b).flip(1)], dim=0)
        scan = (mingru_scan_autograd if os.environ.get("TRK_MINGRU_ADJOINT", "1") == "1"
                else mingru_scan_parallel)
        with torch.autocast("cuda", enabled=False):
            h = scan(a, b)
        h = h.to(x.dtype)
        hf, hb = h[:B], h[B:].flip(1)
        return torch.cat([hf, hb], dim=-1), hf, hb

    def forward(self, x: Tensor, lens: Tensor) -> tuple[Tensor, Tensor]:
        """``x``: (B, S, in_dim) padded, pads trailing.  Returns
        ``(seq (B, S, 2H), terminal (B, 2H))``."""
        H = self.hidden
        use_triton = (
            x.is_cuda
            and not torch.is_grad_enabled()
            and os.environ.get("TRK_MINGRU_KERNEL", "auto") != "off"
        )
        if use_triton:
            from track_regression.ops.mingru_short_triton import mingru_bidi_fused
            zn = self.in_proj(x).contiguous()
            hf, hb = mingru_bidi_fused(zn, lens, H)
            seq = torch.cat([hf, hb], dim=-1)
        else:
            p = torch.arange(x.shape[1], device=x.device)
            seq, hf, hb = self._core(x, p.unsqueeze(0) < lens.unsqueeze(1))
        B = x.shape[0]
        last = (lens - 1).clamp(min=0)
        term = torch.cat(
            [hf[torch.arange(B, device=x.device), last], hb[:, 0]], dim=-1
        )
        return seq, term


def _prefix_flip_index(lens: Tensor, S: int, device) -> Tensor:
    """(B, S) gather index reversing each row's valid prefix; pads identity."""
    p = torch.arange(S, device=device)
    last = (lens - 1).clamp(min=0).unsqueeze(1)
    return torch.where(p.unsqueeze(0) <= last, last - p.unsqueeze(0),
                       p.unsqueeze(0).expand(lens.shape[0], S))


class MinGRUCLSEncoder(nn.Module):
    """Bidirectional minGRU encoder with a terminal-hidden-state readout.

    Structurally identical to :class:`BiGRUCLSEncoder` — same depth, same
    bidirectionality, same readout, same projection to ``pool_out_dim`` — so
    the only difference from the classical GRU is that the gates no longer see
    the recurrent state.
    """

    def __init__(
        self,
        dim: int,
        hidden_size: int = 194,
        num_layers: int = 2,
        pool_out_dim: int = 256,
        dropout: float = 0.0,
        norm: str = "RMSNorm",
        max_len: int = 20,
        compile_core: bool = True,
    ) -> None:
        super().__init__()
        if dropout:
            raise ValueError("the ablation protocol forbids dropout")
        self.dim = int(dim)
        self.hidden = int(hidden_size)
        self.num_layers = int(num_layers)
        self._pool_out_dim = int(pool_out_dim)
        dims = [self.dim] + [2 * self.hidden] * (self.num_layers - 1)
        self.layers = nn.ModuleList(
            [_MinGRULayer(d, self.hidden) for d in dims]
        )
        # Static padded length: every track has <= max_len hits, so padding to
        # a constant makes the whole encoder shape-static and torch.compile
        # sees one graph (padded-static layout, as in mamba_short).
        self.max_len = int(max_len)
        if compile_core:
            for layer in self.layers:
                layer.enable_compiled_core()
        pooled_in = 2 * self.hidden
        self.pool_norm: nn.Module = (
            nn.RMSNorm(pooled_in) if norm == "RMSNorm"
            else (nn.LayerNorm(pooled_in) if norm == "LayerNorm" else nn.Identity())
        )
        self.pool_proj: nn.Module = (
            nn.Identity() if pooled_in == self._pool_out_dim
            else nn.Linear(pooled_in, self._pool_out_dim)
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
        kv_mask: Tensor | None = None,
        **kwargs,  # noqa: ARG002
    ) -> tuple[Tensor, Tensor]:
        from track_regression.ablation_encoders import _packed_to_padded

        # ---- fully packed inference path: no pad rows anywhere ------------
        if (getattr(self, "_use_packed_kernel", True)
                and cu_seqlens is not None and x.is_cuda and not torch.is_grad_enabled()
                and os.environ.get("TRK_MINGRU_KERNEL", "auto") not in ("off", "padded")):
            from track_regression.ops.mingru_short_triton import mingru_bidi_packed
            cu = cu_seqlens.to(torch.int32)
            h = x[0]
            H = self.hidden
            for layer in self.layers:
                # No cast at the boundary: the kernel takes fp32/fp16/bf16 and
                # converts on load in registers, accumulating the recurrence in
                # fp32 regardless.  Materialising an fp32 copy of this (T, 4H)
                # tensor instead costs ~20 % of the forward (measured).
                h = mingru_bidi_packed(
                    layer.in_proj(h).contiguous(), cu, H, self.max_len)
            term = torch.cat(
                [h[cu[1:].long() - 1, :H], h[cu[:-1].long(), H:]], dim=-1)
            pooled = self.pool_proj(self.pool_norm(term))
            return h.unsqueeze(0), pooled

        if cu_seqlens is not None:
            x_pad, row, pos, lens = _packed_to_padded(x, cu_seqlens)
            packed = True
        else:
            B, N, _ = x.shape
            lens = (kv_mask.to(torch.long).sum(-1).clamp(min=1) if kv_mask is not None
                    else torch.full((B,), N, dtype=torch.long, device=x.device))
            x_pad, row, pos, packed = x, None, None, False

        S = x_pad.shape[1]
        if S < self.max_len:                       # pad to the static length
            x_pad = torch.cat(
                [x_pad, x_pad.new_zeros(x_pad.shape[0], self.max_len - S, x_pad.shape[2])],
                dim=1)
        elif S > self.max_len:
            raise ValueError(f"track longer than max_len={self.max_len}: {S}")

        h = x_pad
        term = None
        for layer in self.layers:
            h, term = layer(h, lens)

        pooled = self.pool_proj(self.pool_norm(term))
        seq_out = h[row, pos].unsqueeze(0) if packed else h[:, :S]
        if self.training:
            pooled = pooled + 0.0 * seq_out.float().sum()
        return seq_out, pooled


# ---------------------------------------------------------------------------
# Simpler still: input-INDEPENDENT decay (LRU / S4D-style diagonal linear RNN)
# ---------------------------------------------------------------------------


class _DiagRNNLayer(nn.Module):
    """``h_t = a * h_{t-1} + b_t`` with ``a`` a LEARNED PER-CHANNEL CONSTANT.

    One step simpler than minGRU: the gate disappears entirely, so the
    projection emits only ``b`` (``D -> 2H`` instead of ``D -> 4H``) and the
    recurrence is a fixed exponential decay per channel.  Because ``a`` no
    longer depends on the input, the scan degenerates to a depthwise
    convolution with kernel ``[1, a, a^2, ...]`` — it can be evaluated without
    any scan at all if one wants.

    Note (measured, not assumed): this is **not** cheaper at matched parameter
    count — halving the projection just buys back width, and dense-projection
    FLOPs are 2 x params x tokens either way.  It is tested because it is
    *simpler*, and because a fixed decay is a stronger structural prior.
    """

    def __init__(self, in_dim: int, hidden: int):
        super().__init__()
        self.in_dim, self.hidden = int(in_dim), int(hidden)
        self.in_proj = nn.Linear(self.in_dim, 2 * self.hidden)
        # a = sigmoid(logit); init spread over (0.5, 0.99) so channels cover a
        # range of memory horizons across the <=20 hits of a track.
        lo = torch.logit(torch.linspace(0.5, 0.99, 2 * self.hidden))
        self.decay_logit = nn.Parameter(lo)
        with torch.no_grad():
            self.in_proj.bias.zero_()

    def forward(self, x: Tensor, lens: Tensor) -> tuple[Tensor, Tensor]:
        H = self.hidden
        B, S = x.shape[0], x.shape[1]
        p = torch.arange(S, device=x.device)
        v = (p.unsqueeze(0) < lens.unsqueeze(1)).unsqueeze(-1).to(x.dtype)
        b_all = self.in_proj(x) * v                      # (B, S, 2H) = [fwd | bwd]
        b_f, b_b = b_all.split(H, dim=-1)
        a = torch.sigmoid(self.decay_logit)              # (2H,)
        a_f, a_b = a[:H], a[H:]
        # pads must be the identity map (a = 1, b = 0) so the flip trick holds
        af = torch.where(v.bool(), a_f.view(1, 1, H).expand(B, S, H),
                         torch.ones(1, device=x.device, dtype=x.dtype))
        ab = torch.where(v.bool(), a_b.view(1, 1, H).expand(B, S, H),
                         torch.ones(1, device=x.device, dtype=x.dtype))
        scan = (mingru_scan_autograd if os.environ.get("TRK_MINGRU_ADJOINT", "1") == "1"
                else mingru_scan_parallel)
        h = scan(torch.cat([af, ab.flip(1)], 0), torch.cat([b_f, b_b.flip(1)], 0))
        hf, hb = h[:B], h[B:].flip(1)
        seq = torch.cat([hf, hb], dim=-1)
        last = (lens - 1).clamp(min=0)
        term = torch.cat([hf[torch.arange(B, device=x.device), last], hb[:, 0]], dim=-1)
        return seq, term


class DiagRNNCLSEncoder(MinGRUCLSEncoder):
    """:class:`MinGRUCLSEncoder` with :class:`_DiagRNNLayer` blocks.

    The fused packed kernel is minGRU-specific (it reads a 4H ``[z|n|z|n]``
    projection); this variant emits only 2H, so it runs the eager padded path
    until it gets a kernel of its own.
    """

    _use_packed_kernel = False

    def __init__(self, dim: int, hidden_size: int = 274, num_layers: int = 2,
                 pool_out_dim: int = 256, dropout: float = 0.0,
                 norm: str = "RMSNorm", max_len: int = 20, compile_core: bool = False):
        super().__init__(dim=dim, hidden_size=hidden_size, num_layers=num_layers,
                         pool_out_dim=pool_out_dim, dropout=dropout, norm=norm,
                         max_len=max_len, compile_core=False)
        dims = [int(dim)] + [2 * int(hidden_size)] * (int(num_layers) - 1)
        self.layers = nn.ModuleList([_DiagRNNLayer(d, int(hidden_size)) for d in dims])


# ---------------------------------------------------------------------------
# minLSTM — the sibling of minGRU (Feng et al. 2024)
# ---------------------------------------------------------------------------


class _MinLSTMLayer(nn.Module):
    """``h_t = f'_t * h_{t-1} + i'_t * g_t`` with input-only gates.

    minGRU couples the two gates (``a = 1 - z``, ``b = z n``); minLSTM keeps
    them independent and renormalises so ``f' + i' = 1``, which is what keeps
    the recurrence stable without a tanh.  One more projection than minGRU
    (3H vs 4H per direction... here 6H for both directions), still linear and
    elementwise, so the same scan and the same adjoint apply.
    """

    def __init__(self, in_dim: int, hidden: int):
        super().__init__()
        self.in_dim, self.hidden = int(in_dim), int(hidden)
        self.in_proj = nn.Linear(self.in_dim, 6 * self.hidden)   # [f|i|g] x 2 dirs
        with torch.no_grad():
            self.in_proj.bias.zero_()

    def forward(self, x: Tensor, lens: Tensor) -> tuple[Tensor, Tensor]:
        H = self.hidden
        B, S = x.shape[0], x.shape[1]
        p = torch.arange(S, device=x.device)
        v = (p.unsqueeze(0) < lens.unsqueeze(1)).unsqueeze(-1)
        q = self.in_proj(x)
        f_f, i_f, g_f, f_b, i_b, g_b = q.split(H, dim=-1)

        def gates(fr, ir, g):
            fr, ir, g = fr.float(), ir.float(), g.float()
            f, i = torch.sigmoid(fr), torch.sigmoid(ir)
            tot = f + i + 1e-6
            a, b = f / tot, (i / tot) * g
            vv = v.to(a.dtype)
            return a * vv + (1.0 - vv), b * vv      # pads = identity map

        af, bf = gates(f_f, i_f, g_f)
        ab, bb = gates(f_b, i_b, g_b)
        scan = (mingru_scan_autograd if os.environ.get("TRK_MINGRU_ADJOINT", "1") == "1"
                else mingru_scan_parallel)
        with torch.autocast("cuda", enabled=False):
            h = scan(torch.cat([af, ab.flip(1)], 0), torch.cat([bf, bb.flip(1)], 0))
        h = h.to(x.dtype)
        hf, hb = h[:B], h[B:].flip(1)
        last = (lens - 1).clamp(min=0)
        return (torch.cat([hf, hb], dim=-1),
                torch.cat([hf[torch.arange(B, device=x.device), last], hb[:, 0]], dim=-1))


class MinLSTMCLSEncoder(MinGRUCLSEncoder):
    """minGRU encoder with :class:`_MinLSTMLayer` blocks (decoupled gates)."""

    _use_packed_kernel = False

    def __init__(self, dim: int, hidden_size: int = 160, num_layers: int = 2,
                 pool_out_dim: int = 256, dropout: float = 0.0,
                 norm: str = "RMSNorm", max_len: int = 20, compile_core: bool = False):
        super().__init__(dim=dim, hidden_size=hidden_size, num_layers=num_layers,
                         pool_out_dim=pool_out_dim, dropout=dropout, norm=norm,
                         max_len=max_len, compile_core=False)
        dims = [int(dim)] + [2 * int(hidden_size)] * (int(num_layers) - 1)
        self.layers = nn.ModuleList([_MinLSTMLayer(d, int(hidden_size)) for d in dims])




# ---------------------------------------------------------------------------
# Complex-decay linear RNN (LRU-style) -- an oscillatory prior for helices
# ---------------------------------------------------------------------------


def _complex_scan(ar: Tensor, ai: Tensor, br: Tensor, bi: Tensor
                  ) -> tuple[Tensor, Tensor]:
    """Hillis-Steele scan of ``h_t = a_t h_{t-1} + b_t`` over COMPLEX ``a, b``.

    Carried as real pairs so autograd stays on real tensors: the affine
    composition ``(a1,b1) o (a2,b2) = (a1 a2, a1 b2 + b1)`` is the same, with
    the products expanded into real and imaginary parts.  Same 5 rounds at
    L = 20 as the real scan.
    """
    L = ar.shape[1]
    d = 1
    pad = torch.nn.functional.pad
    while d < L:
        arh = pad(ar[:, :-d], (0, 0, d, 0), value=1.0)
        aih = pad(ai[:, :-d], (0, 0, d, 0), value=0.0)
        brh = pad(br[:, :-d], (0, 0, d, 0), value=0.0)
        bih = pad(bi[:, :-d], (0, 0, d, 0), value=0.0)
        br, bi = br + ar * brh - ai * bih, bi + ar * bih + ai * brh
        ar, ai = ar * arh - ai * aih, ar * aih + ai * arh
        d *= 2
    return br, bi


def _complex_scan_const(r: Tensor, theta: Tensor, br: Tensor, bi: Tensor
                        ) -> tuple[Tensor, Tensor]:
    """Hillis-Steele scan when ``a`` is CONSTANT along the sequence.

    ``a`` here is a learned per-channel eigenvalue, not an input-dependent
    gate, so the composed multiplier after ``k`` rounds is just ``a^(2^k)`` --
    the same for every position, and available in closed form from the polar
    parameterisation (``r^(2^k) e^{i 2^k theta}``).  The scan therefore never
    materialises an ``a`` tensor and carries only ``b``: one complex
    multiply-add per round instead of the general version's two, and no
    ``torch.where`` masking of pads.

    Pads need no identity element here: ``b`` is zeroed at pads by the caller,
    the forward readout is taken at the last valid index, and the backward
    direction flips the whole row so its pads lead a zero state.
    """
    L = br.shape[1]
    pad = torch.nn.functional.pad
    d = 1
    while d < L:
        ar = (r ** d) * torch.cos(d * theta)
        ai = (r ** d) * torch.sin(d * theta)
        brh = pad(br[:, :-d], (0, 0, d, 0), value=0.0)
        bih = pad(bi[:, :-d], (0, 0, d, 0), value=0.0)
        br, bi = br + ar * brh - ai * bih, bi + ar * bih + ai * brh
        d *= 2
    return br, bi


class _ComplexLRULayer(nn.Module):
    """``h_t = r e^{i w} h_{t-1} + b_t`` with a LEARNED COMPLEX decay.

    Motivated by the geometry rather than picked off a list: a charged track
    is a helix, so its azimuth advances almost linearly along the hit
    sequence.  A real decay (minGRU, :class:`_DiagRNNLayer`) can only forget;
    a complex eigenvalue can *rotate*, which is the natural basis for a signal
    that is periodic in phi.  Parameterisation follows the LRU (Orvieto et al.
    2023) and is stable by construction: ``r = exp(-exp(nu)) in (0, 1)``.

    ``hidden`` counts REAL output channels per direction; internally there are
    ``hidden / 2`` complex channels whose real and imaginary parts are both
    returned, so the projection is ``D -> 2 * hidden`` exactly as in
    :class:`_DiagRNNLayer` and the parameter count is directly comparable.
    """

    def __init__(self, in_dim: int, hidden: int):
        super().__init__()
        if int(hidden) % 2:
            raise ValueError("ComplexLRU needs an even hidden width (re/im pairs)")
        self.in_dim, self.hidden = int(in_dim), int(hidden)
        self.c = self.hidden // 2                       # complex channels / dir
        self.in_proj = nn.Linear(self.in_dim, 4 * self.c)
        nu = torch.log(-torch.log(torch.linspace(0.6, 0.99, 2 * self.c)))
        self.nu = nn.Parameter(nu)
        # phase advance per hit: 0 (pure decay) up to ~pi/4, so a 20-hit track
        # can express several radians of accumulated rotation.
        self.theta = nn.Parameter(torch.linspace(0.0, 0.8, 2 * self.c))
        with torch.no_grad():
            self.in_proj.bias.zero_()

    def forward(self, x: Tensor, lens: Tensor) -> tuple[Tensor, Tensor]:
        C, B, S = self.c, x.shape[0], x.shape[1]
        p = torch.arange(S, device=x.device)
        v = (p.unsqueeze(0) < lens.unsqueeze(1)).unsqueeze(-1)
        q = self.in_proj(x).float() * v                 # pads -> b = 0
        bfr, bfi, bbr, bbi = q.split(C, dim=-1)
        r = torch.exp(-torch.exp(self.nu.float())).view(1, 1, 2 * C)
        th = self.theta.float().view(1, 1, 2 * C)
        with torch.autocast("cuda", enabled=False):
            hr, hi = _complex_scan_const(
                r, th,
                torch.cat([bfr, bbr.flip(1)], -1), torch.cat([bfi, bbi.flip(1)], -1))
        hf = torch.cat([hr[..., :C], hi[..., :C]], dim=-1).to(x.dtype) * v
        hb = (torch.cat([hr[..., C:], hi[..., C:]], dim=-1).flip(1)).to(x.dtype) * v
        seq = torch.cat([hf, hb], dim=-1)
        last = (lens - 1).clamp(min=0)
        term = torch.cat([hf[torch.arange(B, device=x.device), last], hb[:, 0]], dim=-1)
        return seq, term


class ComplexLRUCLSEncoder(MinGRUCLSEncoder):
    """:class:`MinGRUCLSEncoder` with complex-decay (oscillatory) blocks."""

    _use_packed_kernel = False

    def __init__(self, dim: int, hidden_size: int = 274, num_layers: int = 2,
                 pool_out_dim: int = 256, dropout: float = 0.0,
                 norm: str = "RMSNorm", max_len: int = 20, compile_core: bool = False):
        super().__init__(dim=dim, hidden_size=hidden_size, num_layers=num_layers,
                         pool_out_dim=pool_out_dim, dropout=dropout, norm=norm,
                         max_len=max_len, compile_core=False)
        dims = [int(dim)] + [2 * int(hidden_size)] * (int(num_layers) - 1)
        self.layers = nn.ModuleList([_ComplexLRULayer(d, int(hidden_size)) for d in dims])


# ---------------------------------------------------------------------------
# Inward-only minGRU: one scan, outermost hit -> innermost, read out at the hit
# nearest the beamline
# ---------------------------------------------------------------------------


class _MinGRUInwardLayer(nn.Module):
    """A single INWARD minGRU scan (outermost hit -> innermost).

    Hits are stored in detector geometry order, innermost first, so the
    "inward" direction is the reverse scan and it terminates at hit 0 — the
    hit nearest the beamline, which is where the perigee parameters are
    defined.  That mirrors a Kalman filter propagating in to the beamline, and
    it puts the freshest state where the answer is read.

    Dropping the second direction is worth more than the obvious factor two,
    because bidirectionality doubles BOTH a layer's output width and the next
    layer's input width: for 2 layers at H = 194 the projection FLOPs go
    800.8 k -> 249.9 k per token, **3.2x**, and the scan moves half the bytes.
    """

    def __init__(self, in_dim: int, hidden: int):
        super().__init__()
        self.in_dim, self.hidden = int(in_dim), int(hidden)
        self.in_proj = nn.Linear(self.in_dim, 2 * self.hidden)   # [z | n], one direction
        with torch.no_grad():
            self.in_proj.bias.zero_()

    def forward(self, x: Tensor, lens: Tensor) -> tuple[Tensor, Tensor]:
        H = self.hidden
        B, S = x.shape[0], x.shape[1]
        p = torch.arange(S, device=x.device)
        v = (p.unsqueeze(0) < lens.unsqueeze(1)).unsqueeze(-1)
        zn = self.in_proj(x)
        z, n = zn.split(H, dim=-1)
        z = torch.sigmoid(z.float()) * v.float()      # pads -> a = 1, b = 0
        n = n.float()
        scan = (mingru_scan_autograd if os.environ.get("TRK_MINGRU_ADJOINT", "1") == "1"
                else mingru_scan_parallel)
        with torch.autocast("cuda", enabled=False):
            # flip the whole padded row: pads are the identity map, so the
            # reverse scan needs no per-row gather (same trick as the bidi layer)
            h = scan((1.0 - z).flip(1), (z * n).flip(1))
        seq = h.flip(1).to(x.dtype) * v               # back to storage order
        return seq, seq[:, 0]                         # terminal = innermost hit


class MinGRUInwardCLSEncoder(MinGRUCLSEncoder):
    """:class:`MinGRUCLSEncoder` with a single INWARD scan.

    Same recipe, same heads; only the direction count changes.  Motivated by
    the ablation finding that a one-directional Mamba-2 matches the
    bidirectional one on every test set (the regression reads a POOLED vector,
    and one full pass has already seen every hit), plus the physics argument
    that the readout belongs at the hit nearest the beamline.

    NOTE the readout is ``seq[:, 0]`` — the innermost hit — not the terminal
    element of the stored order.
    """

    _use_packed_kernel = True         # its own single-direction packed kernel

    def forward(self, x, x_sort_value=None, seq_idx=None, cu_seqlens=None,
                kv_mask=None, **kwargs):
        """Packed inference path: one fused inward scan per layer."""
        if (cu_seqlens is not None and x.is_cuda and not torch.is_grad_enabled()
                and os.environ.get("TRK_MINGRU_KERNEL", "auto") not in ("off", "padded")):
            from track_regression.ops.mingru_short_triton import mingru_inward_packed
            cu = cu_seqlens.to(torch.int32)
            h = x[0]
            H = self.hidden
            for layer in self.layers:
                h = mingru_inward_packed(
                    layer.in_proj(h).contiguous(), cu, H, self.max_len)
            # readout = the innermost hit of each track = the first packed row
            term = h[cu[:-1].long()]
            return h.unsqueeze(0), self.pool_proj(self.pool_norm(term))
        return super().forward(x, x_sort_value=x_sort_value, seq_idx=seq_idx,
                               cu_seqlens=cu_seqlens, kv_mask=kv_mask, **kwargs)

    def __init__(self, dim: int, hidden_size: int = 194, num_layers: int = 2,
                 pool_out_dim: int = 256, dropout: float = 0.0,
                 norm: str = "RMSNorm", max_len: int = 20, compile_core: bool = False):
        super().__init__(dim=dim, hidden_size=hidden_size, num_layers=num_layers,
                         pool_out_dim=pool_out_dim, dropout=dropout, norm=norm,
                         max_len=max_len, compile_core=False)
        dims = [int(dim)] + [int(hidden_size)] * (int(num_layers) - 1)   # H, not 2H
        self.layers = nn.ModuleList([_MinGRUInwardLayer(d, int(hidden_size)) for d in dims])
        pooled_in = int(hidden_size)
        self.pool_norm = (nn.RMSNorm(pooled_in) if norm == "RMSNorm"
                          else (nn.LayerNorm(pooled_in) if norm == "LayerNorm" else nn.Identity()))
        self.pool_proj = (nn.Identity() if pooled_in == self._pool_out_dim
                          else nn.Linear(pooled_in, self._pool_out_dim))
