"""Short-sequence (L<=22) Mamba-2 evaluation via the single-chunk SSD quadratic dual.

The stock ``mamba_ssm`` chunked selective scan is engineered for sequence
lengths >> 8000.  Our physics prior caps tracks at 20 hits (+2 CLS tokens),
where the identical SSD recurrence collapses to one dense lower-triangular
L x L matrix product per (batch, head) -- chunk-free, embarrassingly
parallel, and with no CUDA grid-dimension batch ceiling.

This module provides:

- :class:`Mamba2Short` -- drop-in replacement for ``mamba_ssm.Mamba2`` with
  *identical parameter names/shapes* (the trained checkpoint's state_dict
  loads with ``strict=True``).  Pure-torch quadratic-dual forward.
- :func:`mamba2_block_ref` -- an *independent* fp64-capable reference
  implementation (segsum/einsum evaluation order, following
  ``mamba_ssm/modules/ssd_minimal.py``) used as the numerics oracle.
- Packed <-> padded-static conversion utilities (arithmetic only -- no
  argsort/bucketize/item(), so the static path is compile/graph friendly).
- :func:`apply_variant` -- the campaign's variant plumbing hook used by
  ``scripts/perf/bench_variant.py`` (v0 / v2p / v3 / v3c / v4).

Mathematical contract (verified against installed mamba_ssm==2.3.0 source):
the Mamba2 update is NEVER altered -- this is an algebraically identical
re-expression.  Parity-critical facts mirrored here:

- in_proj split order ``[z, x, B, C, dt]``; no projection biases.
- depthwise width-``d_conv`` causal conv over the joint xBC (with bias),
  then split x/B/C; SiLU activation.
- ``dt = softplus(dt_raw + dt_bias)`` with the softplus *threshold-20*
  linearization (``F.softplus`` default matches the Triton kernel's
  ``tl.where(dt <= 20, softplus(dt), dt)``).
- ``A = -exp(A_log.float())``; per-head scalars A, D, dt_bias.
- D-skip uses the raw post-conv x (not x*dt).
- gated RMSNorm with ``norm_before_gate=False``:
  ``y = RMSNorm(y * silu(z)) * weight``, eps=1e-5, fp32 accumulation.
- B and C are shared across heads (ngroups=1 supported only).
- softplus/cumsum/segsum/exp always computed in fp32 (or the module dtype
  if wider, e.g. fp64 for the reference oracle).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

__all__ = [
    "Mamba2Short",
    "mamba2_block_ref",
    "packed_to_padded_static",
    "padded_static_to_packed",
    "build_static_aux",
    "apply_variant",
    "STATIC_LEN",
]

# 20 hits (dataset manifest max_hits) + 2 CLS tokens.
STATIC_LEN = 22


# ---------------------------------------------------------------------------
# The module
# ---------------------------------------------------------------------------


class Mamba2Short(nn.Module):
    """Drop-in Mamba2 with a chunk-free single-chunk SSD forward for L<=22.

    Constructor signature mirrors the subset of ``mamba_ssm.Mamba2`` kwargs
    used in this repo; parameter names and shapes are identical so trained
    checkpoints load unchanged (``strict=True``).

    The forward accepts padded ``(B, L, D)`` input.  Padding is handled by
    *layout* (Scheme A): pads must strictly trail the valid tokens -- every
    op below is positionwise or causal, so trailing pads cannot influence
    valid outputs.  No masks are needed inside the block.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 128,
        d_conv: int = 4,
        expand: int = 2,
        headdim: int = 64,
        ngroups: int = 1,
        chunk_size: int = 256,  # accepted for API parity; unused (chunk-free)
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        if ngroups != 1:
            raise NotImplementedError("Mamba2Short supports ngroups=1 only")
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.headdim = headdim
        self.ngroups = ngroups
        self.chunk_size = chunk_size
        self.d_inner = expand * d_model
        self.d_ssm = self.d_inner
        assert self.d_ssm % headdim == 0
        self.nheads = self.d_ssm // headdim
        self.activation = "silu"
        # Implementation selector: "torch" (V3) or "triton" (V4, added later).
        self.impl = "torch"

        d_in_proj = 2 * self.d_inner + 2 * self.ngroups * d_state + self.nheads
        self.in_proj = nn.Linear(d_model, d_in_proj, bias=False, **factory_kwargs)

        conv_dim = self.d_ssm + 2 * self.ngroups * d_state
        self.conv_dim = conv_dim
        self.conv1d = nn.Conv1d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            bias=True,
            kernel_size=d_conv,
            groups=conv_dim,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        # Same init as stock Mamba2 (values are overwritten by the checkpoint;
        # kept for standalone correctness of randomly initialised modules).
        dt = torch.exp(
            torch.rand(self.nheads, **factory_kwargs)
            * (math.log(0.1) - math.log(0.001))
            + math.log(0.001)
        ).clamp(min=1e-4)
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        self.dt_bias._no_weight_decay = True

        A = torch.empty(self.nheads, dtype=torch.float32, device=device).uniform_(1, 16)
        self.A_log = nn.Parameter(torch.log(A).to(dtype=dtype))
        self.A_log._no_weight_decay = True

        self.D = nn.Parameter(torch.ones(self.nheads, device=device))
        self.D._no_weight_decay = True

        # State-dict compatible with mamba_ssm's RMSNormGated: one weight
        # vector named ``norm.weight``.  The gated-norm math is implemented
        # in _gated_rmsnorm below (norm_before_gate=False semantics).
        self.norm = nn.Module()
        self.norm.weight = nn.Parameter(torch.ones(self.d_ssm, **factory_kwargs))
        self.norm.eps = 1e-5

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False, **factory_kwargs)

    # -- pieces -------------------------------------------------------------

    def _causal_conv(self, xBC: Tensor) -> Tensor:
        """Depthwise causal width-``d_conv`` conv as shifted MACs.

        Avoids cuDNN entirely (cuDNN's fp32 conv defaults to TF32); plain
        mul/add is exact IEEE in fp32 and fuses under torch.compile.
        xBC: (B, L, conv_dim) -> (B, L, conv_dim), then SiLU.
        """
        w = self.conv1d.weight.squeeze(1)  # (conv_dim, d_conv)
        L = xBC.shape[1]
        xp = F.pad(xBC, (0, 0, self.d_conv - 1, 0))  # zeros at the causal left edge
        out = xp[:, 0:L, :] * w[:, 0]
        for k in range(1, self.d_conv):
            out = out + xp[:, k : k + L, :] * w[:, k]
        out = out + self.conv1d.bias
        return F.silu(out)

    def _gated_rmsnorm(self, y: Tensor, z: Tensor) -> Tensor:
        """``RMSNorm(y * silu(z)) * weight`` -- norm_before_gate=False, fp32+."""
        dtype = y.dtype
        compute_dtype = torch.float64 if dtype == torch.float64 else torch.float32
        g = y.to(compute_dtype) * F.silu(z.to(compute_dtype))
        rstd = torch.rsqrt(g.square().mean(dim=-1, keepdim=True) + self.norm.eps)
        return (g * rstd * self.norm.weight.to(compute_dtype)).to(dtype)

    def _ssd_quadratic(self, x: Tensor, dt: Tensor, B: Tensor, C: Tensor) -> Tensor:
        """Single-chunk SSD dual: Y = (Lmat o (C B^T)) @ (x*dt) + D o x.

        x: (Bt, L, H, P); dt: (Bt, L, H) [post-softplus]; B, C: (Bt, L, N).
        Returns (Bt, L, H, P).
        """
        Bt, L, H, P = x.shape
        compute_dtype = torch.float64 if x.dtype == torch.float64 else torch.float32

        A = -torch.exp(self.A_log.to(compute_dtype))  # (H,)
        dt_c = dt.to(compute_dtype)
        cumA = torch.cumsum(dt_c * A, dim=1)  # (Bt, L, H)

        # segsum decay matrix, computed as differences BEFORE exp; strictly
        # lower+diag kept, upper set to -inf pre-exp (=> exact 0 post-exp).
        diff = cumA.unsqueeze(2) - cumA.unsqueeze(1)  # (Bt, L_l, L_s, H)
        tril = torch.ones(L, L, dtype=torch.bool, device=x.device).tril()
        diff = diff.masked_fill(~tril.unsqueeze(0).unsqueeze(-1), float("-inf"))
        Lmat = torch.exp(diff)  # (Bt, L, L, H)

        # G = <C_l, B_s>, shared across heads (ngroups=1).
        G = torch.matmul(C.to(compute_dtype), B.to(compute_dtype).transpose(1, 2))  # (Bt, L, L)

        M = Lmat * G.unsqueeze(-1)  # (Bt, L, L, H)

        xdt = x.to(compute_dtype) * dt_c.unsqueeze(-1)  # (Bt, L, H, P)
        # (Bt,H,L,L) @ (Bt,H,L,P) -> (Bt,H,L,P)
        Y = torch.matmul(M.permute(0, 3, 1, 2), xdt.permute(0, 2, 1, 3))
        Y = Y.permute(0, 2, 1, 3)  # (Bt, L, H, P)
        Y = Y + self.D.to(compute_dtype).view(1, 1, H, 1) * x.to(compute_dtype)
        return Y.to(x.dtype)

    # -- forward ------------------------------------------------------------

    def forward(
        self,
        u: Tensor,
        seqlen: int | None = None,
        seq_idx: Tensor | None = None,
        cu_seqlens: Tensor | None = None,
        inference_params=None,
    ) -> Tensor:
        """Padded ``(B, L, D)`` forward.  API mirrors ``Mamba2.forward``.

        ``seq_idx``/``cu_seqlens``/packed layouts are NOT supported -- the
        static path guarantees per-row sequences with trailing pads, which
        is the whole point (Scheme A).
        """
        if seq_idx is not None or cu_seqlens is not None or inference_params is not None:
            raise NotImplementedError(
                "Mamba2Short is padded-static only (Scheme A); got packed args"
            )
        if seqlen is not None:
            u = u.reshape(-1, seqlen, u.shape[-1])

        Bt, L, _ = u.shape
        H, P, N = self.nheads, self.headdim, self.d_state

        zxbcdt = self.in_proj(u)  # (Bt, L, d_in_proj)

        if self.impl == "triton" and u.is_cuda and u.dtype == torch.float32:
            # V4: fused conv+SiLU+dt+SSD-dual+D-skip in one Triton kernel;
            # gated norm + out_proj stay outside (compiled/cuBLAS).
            from track_regression.ops.ssd_short_triton import ssd_short_fwd

            z = zxbcdt[..., : self.d_ssm]
            Y = ssd_short_fwd(
                zxbcdt,
                self.conv1d.weight,
                self.conv1d.bias,
                self.dt_bias,
                self.A_log,
                self.D,
                H, P, N,
            )
            y = self._gated_rmsnorm(Y, z)
            return self.out_proj(y)

        z, xBC, dt_raw = torch.split(
            zxbcdt, [self.d_ssm, self.conv_dim, self.nheads], dim=-1
        )
        xBC = self._causal_conv(xBC)
        x, B, C = torch.split(xBC, [self.d_ssm, N, N], dim=-1)

        compute_dtype = torch.float64 if u.dtype == torch.float64 else torch.float32
        dt = F.softplus(dt_raw.to(compute_dtype) + self.dt_bias.to(compute_dtype))

        Y = self._ssd_quadratic(x.view(Bt, L, H, P), dt.to(u.dtype), B, C)
        y = self._gated_rmsnorm(Y.reshape(Bt, L, self.d_ssm), z)
        return self.out_proj(y)


class Mamba2ShortWithState(Mamba2Short):
    """Mamba2Short that can also return the terminal SSM state (closed form).

    Drop-in for ``mamba_state.Mamba2WithState`` (same ``Mamba2Output``
    contract, final_state (B, H, P, N)).  The single-chunk closed form gives
    h_T[b,h,p,n] = sum_s exp(cumA_T - cumA_s) * dt_s * x_s[p] * B_s[n] —
    no scan, no chunk state passing.  Padded-only (like the state encoder).
    """

    def forward(  # type: ignore[override]
        self,
        u: Tensor,
        seq_idx: Tensor | None = None,
        *,
        return_state: bool = False,
    ):
        from track_regression.mamba_state import Mamba2Output

        if not return_state:
            return Mamba2Output(output=super().forward(u, seq_idx=seq_idx), final_state=None)

        Bt, L, _ = u.shape
        H, P, N = self.nheads, self.headdim, self.d_state
        zxbcdt = self.in_proj(u)
        z, xBC, dt_raw = torch.split(
            zxbcdt, [self.d_ssm, self.conv_dim, self.nheads], dim=-1
        )
        xBC = self._causal_conv(xBC)
        x, B, C = torch.split(xBC, [self.d_ssm, N, N], dim=-1)

        compute_dtype = torch.float64 if u.dtype == torch.float64 else torch.float32
        dt = F.softplus(dt_raw.to(compute_dtype) + self.dt_bias.to(compute_dtype))
        xh = x.view(Bt, L, H, P)

        Y = self._ssd_quadratic(xh, dt.to(u.dtype), B, C)
        y = self._gated_rmsnorm(Y.reshape(Bt, L, self.d_ssm), z)
        out = self.out_proj(y)

        # Terminal state, closed form (fp32+): decay from each position to T.
        A = -torch.exp(self.A_log.to(compute_dtype))
        cumA = torch.cumsum(dt * A, dim=1)                      # (Bt, L, H)
        decay = torch.exp(cumA[:, -1:, :] - cumA)               # (Bt, L, H)
        w = decay * dt                                          # (Bt, L, H)
        final = torch.einsum(
            "blh,blhp,bln->bhpn",
            w, xh.to(compute_dtype), B.to(compute_dtype),
        ).to(u.dtype)
        return Mamba2Output(output=out, final_state=final)


# ---------------------------------------------------------------------------
# Independent fp64-capable reference (numerics oracle O1/O2)
# ---------------------------------------------------------------------------


def _segsum_ref(x: Tensor) -> Tensor:
    """Stable segment sum, evaluation order copied from ssd_minimal.segsum."""
    T = x.size(-1)
    x = x.unsqueeze(-1).expand(*x.shape, T)
    mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=-1)
    x = x.masked_fill(~mask, 0)
    x_segsum = torch.cumsum(x, dim=-2)
    mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=0)
    return x_segsum.masked_fill(~mask, -torch.inf)


def mamba2_block_ref(module: nn.Module, u: Tensor) -> Tensor:
    """Reference forward for a Mamba2/Mamba2Short parameter set.

    Independent evaluation order (einsum-based, following
    ``mamba_ssm/modules/ssd_minimal.py`` with a single chunk + explicit
    conv/gated-norm/projections).  Runs entirely in the dtype of ``u``
    (pass fp64 inputs + a ``.double()`` module for the oracle).
    """
    d_ssm = module.d_ssm if hasattr(module, "d_ssm") else module.d_inner
    nheads, headdim, d_state = module.nheads, module.headdim, module.d_state
    d_conv = module.d_conv
    conv_dim = d_ssm + 2 * d_state

    zxbcdt = F.linear(u, module.in_proj.weight)
    z, xBC, dt_raw = torch.split(zxbcdt, [d_ssm, conv_dim, nheads], dim=-1)

    # Causal depthwise conv via F.conv1d (groups), independent of the
    # shifted-MAC evaluation in Mamba2Short.  IEEE in fp64 regardless.
    xBC_t = xBC.transpose(1, 2)
    conv = F.conv1d(
        xBC_t,
        module.conv1d.weight.to(u.dtype),
        module.conv1d.bias.to(u.dtype),
        padding=d_conv - 1,
        groups=conv_dim,
    )[..., : u.shape[1]]
    xBC = F.silu(conv.transpose(1, 2))
    x, B, C = torch.split(xBC, [d_ssm, d_state, d_state], dim=-1)

    dt = F.softplus(dt_raw + module.dt_bias.to(u.dtype))  # (Bt, L, H)
    A = -torch.exp(module.A_log.to(u.dtype))  # (H,)

    X = x.view(*x.shape[:2], nheads, headdim)
    Adt = (dt * A).permute(0, 2, 1)  # (Bt, H, L)
    Lmat = torch.exp(_segsum_ref(Adt))  # (Bt, H, L, L)
    Bm = B.unsqueeze(2).expand(-1, -1, nheads, -1)  # (Bt, L, H, N)
    Cm = C.unsqueeze(2).expand(-1, -1, nheads, -1)
    Y = torch.einsum("blhn,bshn,bhls,bshp->blhp", Cm, Bm, Lmat, X * dt.unsqueeze(-1))
    Y = Y + module.D.to(u.dtype).view(1, 1, nheads, 1) * X

    y = Y.reshape(*Y.shape[:2], d_ssm)
    g = y * F.silu(z)
    eps = module.norm.eps if hasattr(module.norm, "eps") else 1e-5
    rstd = torch.rsqrt(g.square().mean(dim=-1, keepdim=True) + eps)
    y = g * rstd * module.norm.weight.to(u.dtype)
    return F.linear(y, module.out_proj.weight)


# ---------------------------------------------------------------------------
# Packed <-> padded-static conversion (arithmetic only; no argsort)
# ---------------------------------------------------------------------------


def build_static_aux(cu_seqlens: Tensor, static_len: int = STATIC_LEN) -> dict:
    """Index tensors for the static layout, derived from cu_seqlens alone.

    Layout per row r (hit count Lr): ``[cls_bwd, h_0..h_{Lr-1}, cls_fwd, PAD..]``
    -- pads strictly trailing in the forward order, and (via ``flip_idx``)
    strictly trailing in the backward order too.
    """
    cu = cu_seqlens.long()
    lengths = cu[1:] - cu[:-1]  # (B,) hit counts
    Bt = lengths.shape[0]
    device = cu.device
    if int(lengths.max()) + 2 > static_len:
        raise ValueError(
            f"track with {int(lengths.max())} hits does not fit static_len={static_len}"
        )
    p = torch.arange(static_len, device=device).unsqueeze(0)  # (1, S)
    lr = lengths.unsqueeze(1)  # (B, 1)
    valid = p <= lr + 1  # (B, S) cls_bwd + hits + cls_fwd
    # Valid-prefix flip: reverse positions 0..Lr+1, identity on the pad tail.
    flip_idx = torch.where(p <= lr + 1, lr + 1 - p, p)  # (B, S)
    cls_fwd_pos = lengths + 1  # (B,)
    return {
        "lengths": lengths,
        "valid": valid,
        "flip_idx": flip_idx,
        "cls_fwd_pos": cls_fwd_pos,
        "static_len": static_len,
        "batch": Bt,
    }


def packed_to_padded_static(
    x: Tensor, cu_seqlens: Tensor, aux: dict
) -> tuple[Tensor, Tensor, Tensor]:
    """(1, total_L, D) packed hits -> (B, S, D) compacted static rows.

    Returns (x_pad, row, col); (row, col) are reused for the inverse gather.
    Hits land at columns 1..Lr; columns 0 and Lr+1 are left zero for the
    CLS tokens (inserted functionally by the encoder).
    """
    assert x.dim() == 3 and x.shape[0] == 1
    cu = cu_seqlens.long()
    total = x.shape[1]
    device = x.device
    token = torch.arange(total, device=device)
    row = torch.bucketize(token, cu[1:], right=True)  # (T,) segment id
    col = token - cu[row] + 1  # hits start at column 1
    x_pad = x.new_zeros(aux["batch"], aux["static_len"], x.shape[-1])
    x_pad = x_pad.index_put((row, col), x[0])
    return x_pad, row, col


def padded_static_to_packed(x_pad: Tensor, row: Tensor, col: Tensor) -> Tensor:
    """(B, S, D) static rows -> (1, total_L, D) packed hits (inverse gather)."""
    return x_pad[row, col].unsqueeze(0)


# ---------------------------------------------------------------------------
# V4.1 — fused bidirectional evaluation of one layer's two Mamba2Short blocks
# ---------------------------------------------------------------------------


def fused_bidi_scan(
    layer: nn.Module, x_norm: Tensor, flip_idx: Tensor, lens: Tensor | None = None
) -> tuple[Tensor, Tensor]:
    """Both scan directions with ONE in_proj GEMM and NO flip gathers.

    Requires ``apply_variant(..., 'v4')`` to have registered the fused
    weight buffers on the layer.  Returns (x_fwd, x_bwd) post-out_proj, in
    physical (unflipped) positions — drop-in for the layer's standard path.
    ``lens`` (int32, = Lr+1 per row) should be precomputed once per forward
    by the caller; deriving it per layer costs a materialisation.
    """
    from track_regression.ops.ssd_short_triton import gated_rmsnorm, ssd_short_fwd

    fm, bm = layer.forward_mamba, layer.backward_mamba
    H, P, N = fm.nheads, fm.headdim, fm.d_state
    dproj = fm.in_proj.weight.shape[0]
    if lens is None:
        lens = flip_idx[:, 0].to(torch.int32)  # flip of position 0 == Lr+1

    Bt, L, _ = x_norm.shape

    zx_f = fm.in_proj(x_norm).contiguous()
    zx_b = bm.in_proj(x_norm).contiguous()
    k2 = bool(getattr(fm, "_kernel2", False))
    y_f = ssd_short_fwd(
        zx_f, fm.conv1d.weight, fm.conv1d.bias, fm.dt_bias, fm.A_log, fm.D,
        H, P, N, None, False, 0, 0, k2,
    )
    y_b = ssd_short_fwd(
        zx_b, bm.conv1d.weight, bm.conv1d.bias, bm.dt_bias, bm.A_log, bm.D,
        H, P, N, lens, True, 0, 0, k2,
    )
    # Positionwise fused gated norm — z is the leading slice of each zx row,
    # so the full row tensor is passed (no slice copy). Physical space on
    # both sides (positionwise ops commute with the flip permutation).
    yn_f = gated_rmsnorm(y_f, zx_f.view(Bt * L, dproj), fm.norm.weight, fm.norm.eps)
    yn_b = gated_rmsnorm(y_b, zx_b.view(Bt * L, dproj), bm.norm.weight, bm.norm.eps)
    return fm.out_proj(yn_f), bm.out_proj(yn_b)


def fused_bidi_scan_packed(
    layer: nn.Module, x_norm: Tensor, cu_seqlens_aug: Tensor
) -> tuple[Tensor, Tensor]:
    """v5p: both directions on the PACKED augmented stream (no pad rows).

    cuBLAS projections and the fused gated norm run on the packed rows; the
    scan kernel addresses each track's segment via cu_seqlens_aug and flips
    the backward direction within the segment in-kernel — no gathers, no
    padded intermediates, ~31% less work everywhere vs the padded-static
    layout (head-to-head measured, see OPTIMIZATION_LOG Night 2).
    """
    from track_regression.ops.ssd_short_triton import (
        gated_rmsnorm,
        ssd_short_fwd_packed,
    )

    fm, bm = layer.forward_mamba, layer.backward_mamba
    H, P, N = fm.nheads, fm.headdim, fm.d_state
    dproj = fm.in_proj.weight.shape[0]

    rows = x_norm[0] if x_norm.dim() == 3 else x_norm  # (T_aug, D)
    T = rows.shape[0]
    zx_f = fm.in_proj(rows).contiguous()
    zx_b = bm.in_proj(rows).contiguous()
    y_f = ssd_short_fwd_packed(
        zx_f, fm.conv1d.weight, fm.conv1d.bias, fm.dt_bias, fm.A_log, fm.D,
        cu_seqlens_aug, H, P, N, False,
    )
    y_b = ssd_short_fwd_packed(
        zx_b, bm.conv1d.weight, bm.conv1d.bias, bm.dt_bias, bm.A_log, bm.D,
        cu_seqlens_aug, H, P, N, True,
    )
    yn_f = gated_rmsnorm(y_f, zx_f.view(T, dproj), fm.norm.weight, fm.norm.eps)
    yn_b = gated_rmsnorm(y_b, zx_b.view(T, dproj), bm.norm.weight, bm.norm.eps)
    return (
        fm.out_proj(yn_f).unsqueeze(0),
        bm.out_proj(yn_b).unsqueeze(0),
    )


def _register_fused_weights(layer: nn.Module) -> None:
    fm, bm = layer.forward_mamba, layer.backward_mamba
    with torch.no_grad():
        layer._fused_in_w = torch.cat([fm.in_proj.weight, bm.in_proj.weight], 0).contiguous()
        layer._fused_out_w = torch.stack(
            [fm.out_proj.weight.t().contiguous(), bm.out_proj.weight.t().contiguous()]
        ).contiguous()


# ---------------------------------------------------------------------------
# Variant plumbing (used by scripts/perf/bench_variant.py and tests)
# ---------------------------------------------------------------------------

_VARIANTS = ("v0", "v2p", "v3", "v3c", "v4", "v5", "v5p", "v5pc", "auto")


def _iter_bidi_layers(encoder: nn.Module):
    for layer in list(getattr(encoder, "layers", [])) + [encoder.final_layer]:
        yield layer


def _swap_mamba(
    layer_attr_owner: nn.Module, attr: str, impl: str, cls: type | None = None
) -> None:
    stock = getattr(layer_attr_owner, attr)
    short = (cls or Mamba2Short)(
        d_model=stock.d_model,
        d_state=stock.d_state,
        d_conv=stock.d_conv,
        expand=stock.expand,
        headdim=stock.headdim,
        ngroups=stock.ngroups,
        chunk_size=stock.chunk_size,
    )
    short.load_state_dict(stock.state_dict(), strict=True)
    short.impl = impl
    short.to(next(stock.parameters()).device, next(stock.parameters()).dtype)
    setattr(layer_attr_owner, attr, short)


def apply_variant(model: nn.Module, variant: str) -> nn.Module:
    """Mutate ``model`` (a TrackParameterRegressor or a bare encoder) in place.

    v0   -- stock packed configuration (no-op).
    v2p  -- padded-static layout on the STOCK kernel (parity bridge);
            single chunk via chunk_size=32.
    v3   -- padded-static + Mamba2Short (pure-torch quadratic dual).
    v3c  -- v3 + torch.compile of the static core.
    v4   -- v3 with the fused Triton kernel implementation (impl="triton").
    """
    if variant not in _VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; expected one of {_VARIANTS}")
    if variant == "v0":
        return model

    encoder = getattr(model, "encoder", model)

    if type(encoder).__name__ == "BidirectionalMambaEncoder":
        # State-ejection encoder (pool=ssm_state, padded-native). Supported
        # variants: v3 (eager quadratic) / v3c (+torch.compile). The final
        # state layer gets the closed-form terminal state
        # (Mamba2ShortWithState); architecture and weights untouched.
        if variant not in ("v3", "v3c"):
            raise NotImplementedError(
                f"state-eject encoder supports v3/v3c only, got {variant!r}"
            )
        for layer in encoder.layers:
            _swap_mamba(layer, "forward_mamba", "torch")
            _swap_mamba(layer, "backward_mamba", "torch")
        _swap_mamba(encoder.final_layer, "forward_mamba", "torch", Mamba2ShortWithState)
        _swap_mamba(encoder.final_layer, "backward_mamba", "torch", Mamba2ShortWithState)
        if variant == "v3c":
            compiled = torch.compile(encoder, dynamic=False)
            if model is not encoder:
                model.encoder = compiled
                return model
            return compiled
        return model

    if not hasattr(encoder, "enable_static_mode"):
        raise NotImplementedError(
            "encoder has no padded-static path (mamba_cls.py hook missing)"
        )

    if variant == "v2p":
        for layer in _iter_bidi_layers(encoder):
            layer.forward_mamba.chunk_size = 32
            layer.backward_mamba.chunk_size = 32
    else:
        if variant == "auto":
            # Config default (user 2026-07-09): v3c semantics while
            # model.training (compiled static core, pure-torch autograd) and
            # v5pc semantics in eval/val/test/predict (fused packed kernels).
            for layer in _iter_bidi_layers(encoder):
                _swap_mamba(layer, "forward_mamba", "torch")
                _swap_mamba(layer, "backward_mamba", "torch")
                layer._packed_fused = True   # eval path (gated on not training)
            encoder.enable_static_mode(compile_core=True)   # training path
            encoder.enable_packed_compile()                  # eval path
            encoder._auto_kernel = True
            return model

        impl = "triton" if variant in ("v4", "v5", "v5p", "v5pc") else "torch"
        if impl == "triton":
            # Import check up-front so a missing kernel fails loudly here.
            from track_regression.ops import ssd_short_triton  # noqa: F401
        for layer in _iter_bidi_layers(encoder):
            _swap_mamba(layer, "forward_mamba", impl)
            _swap_mamba(layer, "backward_mamba", impl)
            if impl == "triton":
                _register_fused_weights(layer)
                # v5p(+c): fused packed-stream path (no static conversion).
                layer._packed_fused = variant in ("v5p", "v5pc")
                # v5: kernel 2 — one program per track, heads looped
                # in-kernel (B/C conv + Gram matrix computed once).
                layer.forward_mamba._kernel2 = variant == "v5"
                layer.backward_mamba._kernel2 = variant == "v5"

    # v3c/v4 compile the static core; for v4 the fused Triton op is opaque to
    # Inductor (torch.library.custom_op) and the gate/norm fuse around it.
    if variant in ("v5p", "v5pc"):
        # v5p keeps the production packed path (_forward_packed); the fused
        # packed branch inside each layer takes over via _packed_fused.
        if variant == "v5pc":
            encoder.enable_packed_compile()
    else:
        encoder.enable_static_mode(compile_core=(variant in ("v3c", "v4", "v5")))
    return model
