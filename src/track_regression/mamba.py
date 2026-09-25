"""Bidirectional Mamba-2 encoder with learned CLS tokens, for tracks of <= 20 hits.

Encoder
-------
Every track is augmented to ``(cls_bwd, h_0, ..., h_{L-1}, cls_fwd)``.  Each
of the ``num_layers`` bidirectional layers normalises its input, runs a
forward and a reverse Mamba-2 block over the track, merges them with a
learned sigmoid gate and adds the residual.  The final layer's outputs at
``cls_fwd`` (the last token of the forward scan) and ``cls_bwd`` (the last
token of the reverse scan) are RMS-normalised and concatenated to the pooled
output ``(B, 2 * dim)``.

Mamba-2 block
-------------
:class:`Mamba2Short` has the parameters of ``mamba_ssm.Mamba2`` (same names
and shapes) but evaluates the selective scan as its single-chunk SSD quadratic
dual: at L <= 22 the recurrence is one dense lower-triangular L x L matrix
product per (track, head).

Kernels
-------
* Training and the reference inference path: the packed batch is scattered
  into static rows ``[cls_bwd, hits, cls_fwd, pad...]`` of length 22 (pads
  strictly trailing in both scan directions, so no masking is needed inside
  the blocks) and the layer stack is ``torch.compile``d with static shapes.
* Fused inference path (default at inference, see
  :mod:`track_regression.layout`): the layers run on the packed augmented
  stream; projections are cuBLAS GEMMs on the real rows, the scan and the
  gated norm are the Triton kernels of :mod:`track_regression.ops.ssd_short_triton`.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from track_regression.layout import fused_kernels_enabled

# 20 hits + 2 CLS tokens.
STATIC_LEN = 22


def _compute_dtype(t: Tensor) -> torch.dtype:
    """softplus / cumsum / exp / norms run in fp32 (fp64 for fp64 inputs)."""
    return torch.float64 if t.dtype == torch.float64 else torch.float32


# ---------------------------------------------------------------------------
# Mamba-2 block
# ---------------------------------------------------------------------------


class Mamba2Short(nn.Module):
    """Mamba-2 block with a chunk-free single-chunk SSD forward for L <= 22.

    The forward takes padded ``(B, L, D)`` input with pads strictly trailing:
    every op is positionwise or causal, so trailing pads cannot influence the
    valid outputs.  Parity-critical details of ``mamba_ssm.Mamba2`` mirrored
    here: in_proj split ``[z, x, B, C, dt]`` without bias; depthwise causal
    conv (with bias) + SiLU over the joint xBC; ``dt = softplus(dt + dt_bias)``;
    ``A = -exp(A_log)``; D-skip on the post-conv x; gated RMSNorm
    ``RMSNorm(y * silu(z)) * w`` (eps 1e-5); B and C shared across heads.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        d_conv: int = 1,
        expand: int = 2,
        headdim: int = 32,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.headdim = headdim
        self.d_inner = self.d_ssm = expand * d_model
        assert self.d_ssm % headdim == 0
        self.nheads = self.d_ssm // headdim

        self.in_proj = nn.Linear(d_model, 2 * self.d_inner + 2 * d_state + self.nheads, bias=False)
        self.conv_dim = self.d_ssm + 2 * d_state
        self.conv1d = nn.Conv1d(self.conv_dim, self.conv_dim, bias=True, kernel_size=d_conv,
                                groups=self.conv_dim, padding=d_conv - 1)
        # initialisation as in mamba_ssm.Mamba2
        dt = torch.exp(torch.rand(self.nheads) * (math.log(0.1) - math.log(0.001))
                       + math.log(0.001)).clamp(min=1e-4)
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        self.A_log = nn.Parameter(torch.log(torch.empty(self.nheads).uniform_(1, 16)))
        self.D = nn.Parameter(torch.ones(self.nheads))
        for p in (self.dt_bias, self.A_log, self.D):
            p._no_weight_decay = True
        # state-dict compatible with mamba_ssm's RMSNormGated (``norm.weight``)
        self.norm = nn.Module()
        self.norm.weight = nn.Parameter(torch.ones(self.d_ssm))
        self.norm.eps = 1e-5
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def _causal_conv(self, xBC: Tensor) -> Tensor:
        """Depthwise causal conv as shifted multiply-adds (exact IEEE fp32, no cuDNN), then SiLU."""
        w = self.conv1d.weight.squeeze(1)  # (conv_dim, d_conv)
        L = xBC.shape[1]
        xp = F.pad(xBC, (0, 0, self.d_conv - 1, 0))
        out = xp[:, 0:L, :] * w[:, 0]
        for k in range(1, self.d_conv):
            out = out + xp[:, k : k + L, :] * w[:, k]
        return F.silu(out + self.conv1d.bias)

    def _gated_rmsnorm(self, y: Tensor, z: Tensor) -> Tensor:
        cd = _compute_dtype(y)
        g = y.to(cd) * F.silu(z.to(cd))
        rstd = torch.rsqrt(g.square().mean(dim=-1, keepdim=True) + self.norm.eps)
        return (g * rstd * self.norm.weight.to(cd)).to(y.dtype)

    def _ssd_quadratic(self, x: Tensor, dt: Tensor, B: Tensor, C: Tensor) -> Tensor:
        """Single-chunk SSD dual ``Y = (L o (C B^T)) @ (x dt) + D x`` (fp32, fp64 for fp64 inputs).

        x: (Bt, L, H, P); dt: (Bt, L, H) post-softplus; B, C: (Bt, L, N).
        """
        Bt, L, H, P = x.shape
        cd = _compute_dtype(x)
        A = -torch.exp(self.A_log.to(cd))
        dt_c = dt.to(cd)
        cumA = torch.cumsum(dt_c * A, dim=1)                        # (Bt, L, H)
        # decay matrix from differences before exp; upper triangle -> exactly 0
        diff = cumA.unsqueeze(2) - cumA.unsqueeze(1)                # (Bt, L, L, H)
        tril = torch.ones(L, L, dtype=torch.bool, device=x.device).tril()
        Lmat = torch.exp(diff.masked_fill(~tril.unsqueeze(0).unsqueeze(-1), float("-inf")))
        G = torch.matmul(C.to(cd), B.to(cd).transpose(1, 2))      # (Bt, L, L)
        M = Lmat * G.unsqueeze(-1)
        xdt = x.to(cd) * dt_c.unsqueeze(-1)
        Y = torch.matmul(M.permute(0, 3, 1, 2), xdt.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)
        Y = Y + self.D.to(cd).view(1, 1, H, 1) * x.to(cd)
        return Y.to(x.dtype)

    def forward(self, u: Tensor) -> Tensor:
        Bt, L, _ = u.shape
        H, P, N = self.nheads, self.headdim, self.d_state
        z, xBC, dt_raw = torch.split(self.in_proj(u), [self.d_ssm, self.conv_dim, self.nheads], dim=-1)
        x, B, C = torch.split(self._causal_conv(xBC), [self.d_ssm, N, N], dim=-1)
        cd = _compute_dtype(u)
        dt = F.softplus(dt_raw.to(cd) + self.dt_bias.to(cd))
        Y = self._ssd_quadratic(x.view(Bt, L, H, P), dt.to(u.dtype), B, C)
        return self.out_proj(self._gated_rmsnorm(Y.reshape(Bt, L, self.d_ssm), z))


def mamba2_block_ref(module: nn.Module, u: Tensor) -> Tensor:
    """Independent einsum reference of a :class:`Mamba2Short` forward (tests;
    runs in the dtype of ``u``, e.g. fp64 with a ``.double()`` module)."""
    d_ssm, nheads, headdim, d_state = module.d_ssm, module.nheads, module.headdim, module.d_state
    conv_dim = d_ssm + 2 * d_state
    z, xBC, dt_raw = torch.split(F.linear(u, module.in_proj.weight), [d_ssm, conv_dim, nheads], dim=-1)
    conv = F.conv1d(xBC.transpose(1, 2), module.conv1d.weight.to(u.dtype), module.conv1d.bias.to(u.dtype),
                    padding=module.d_conv - 1, groups=conv_dim)[..., : u.shape[1]]
    x, B, C = torch.split(F.silu(conv.transpose(1, 2)), [d_ssm, d_state, d_state], dim=-1)
    dt = F.softplus(dt_raw + module.dt_bias.to(u.dtype))
    A = -torch.exp(module.A_log.to(u.dtype))
    X = x.view(*x.shape[:2], nheads, headdim)
    Adt = (dt * A).permute(0, 2, 1)
    T = Adt.shape[-1]
    seg = Adt.unsqueeze(-1).expand(*Adt.shape, T).masked_fill(
        ~torch.tril(torch.ones(T, T, device=u.device, dtype=torch.bool), -1), 0).cumsum(-2)
    Lmat = torch.exp(seg.masked_fill(~torch.tril(torch.ones(T, T, device=u.device, dtype=torch.bool)), -torch.inf))
    Bm = B.unsqueeze(2).expand(-1, -1, nheads, -1)
    Cm = C.unsqueeze(2).expand(-1, -1, nheads, -1)
    Y = torch.einsum("blhn,bshn,bhls,bshp->blhp", Cm, Bm, Lmat, X * dt.unsqueeze(-1))
    Y = Y + module.D.to(u.dtype).view(1, 1, nheads, 1) * X
    g = Y.reshape(*Y.shape[:2], d_ssm) * F.silu(z)
    y = g * torch.rsqrt(g.square().mean(dim=-1, keepdim=True) + module.norm.eps) * module.norm.weight.to(u.dtype)
    return F.linear(y, module.out_proj.weight)


# ---------------------------------------------------------------------------
# packed <-> static layout (arithmetic only, compile friendly)
# ---------------------------------------------------------------------------


def build_static_aux(cu_seqlens: Tensor, static_len: int = STATIC_LEN) -> dict:
    """Index tensors of the static layout ``[cls_bwd, h_0..h_{Lr-1}, cls_fwd, PAD..]``."""
    cu = cu_seqlens.long()
    lengths = cu[1:] - cu[:-1]
    if int(lengths.max()) + 2 > static_len:
        raise ValueError(f"track with {int(lengths.max())} hits does not fit static_len={static_len}")
    p = torch.arange(static_len, device=cu.device).unsqueeze(0)
    lr = lengths.unsqueeze(1)
    return {
        "valid": p <= lr + 1,                            # cls_bwd + hits + cls_fwd
        "flip_idx": torch.where(p <= lr + 1, lr + 1 - p, p),   # valid-prefix flip, pads fixed
        "cls_fwd_pos": lengths + 1,
        "static_len": static_len,
        "batch": lengths.shape[0],
    }


def packed_to_padded_static(x: Tensor, cu_seqlens: Tensor, aux: dict) -> tuple[Tensor, Tensor, Tensor]:
    """(1, T, D) packed hits -> (B, S, D) static rows, hits at columns 1..Lr."""
    cu = cu_seqlens.long()
    token = torch.arange(x.shape[1], device=x.device)
    row = torch.bucketize(token, cu[1:], right=True)
    col = token - cu[row] + 1
    x_pad = x.new_zeros(aux["batch"], aux["static_len"], x.shape[-1]).index_put((row, col), x[0])
    return x_pad, row, col


def _segment_flip_indices(cu_seqlens: Tensor, total_len: int) -> Tensor:
    """Gather index reversing every segment of a packed stream (its own inverse)."""
    cs = cu_seqlens.to(torch.long)
    arange = torch.arange(total_len, device=cs.device, dtype=torch.long)
    seg = torch.bucketize(arange, cs[1:], right=True)
    return cs[seg] + cs[seg + 1] - 1 - arange


# ---------------------------------------------------------------------------
# fused packed path (inference)
# ---------------------------------------------------------------------------


def fused_bidi_scan_packed(layer: nn.Module, x_norm: Tensor, cu_seqlens_aug: Tensor) -> tuple[Tensor, Tensor]:
    """Both scan directions of one layer on the packed augmented stream."""
    from track_regression.ops.ssd_short_triton import gated_rmsnorm, ssd_short_fwd_packed

    fm, bm = layer.forward_mamba, layer.backward_mamba
    H, P, N = fm.nheads, fm.headdim, fm.d_state
    dproj = fm.in_proj.weight.shape[0]
    rows = x_norm[0]
    T = rows.shape[0]
    zx_f = fm.in_proj(rows).contiguous()
    zx_b = bm.in_proj(rows).contiguous()
    y_f = ssd_short_fwd_packed(zx_f, fm.conv1d.weight, fm.conv1d.bias, fm.dt_bias, fm.A_log, fm.D,
                               cu_seqlens_aug, H, P, N, False)
    y_b = ssd_short_fwd_packed(zx_b, bm.conv1d.weight, bm.conv1d.bias, bm.dt_bias, bm.A_log, bm.D,
                               cu_seqlens_aug, H, P, N, True)
    yn_f = gated_rmsnorm(y_f, zx_f.view(T, dproj), fm.norm.weight, fm.norm.eps)
    yn_b = gated_rmsnorm(y_b, zx_b.view(T, dproj), bm.norm.weight, bm.norm.eps)
    return fm.out_proj(yn_f).unsqueeze(0), bm.out_proj(yn_b).unsqueeze(0)


# ---------------------------------------------------------------------------
# layers and encoder
# ---------------------------------------------------------------------------


class BidirectionalMambaLayer(nn.Module):
    """``x + gate * fwd(norm(x)) + (1 - gate) * bwd(norm(x))``, gate = sigmoid(Linear(norm(x)))."""

    def __init__(self, dim: int, d_state: int = 64, d_conv: int = 1, expand: int = 2, headdim: int = 32):
        super().__init__()
        self.norm = nn.RMSNorm(dim)
        kw = dict(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand, headdim=headdim)
        self.forward_mamba = Mamba2Short(**kw)
        self.backward_mamba = Mamba2Short(**kw)
        self.gate = nn.Linear(dim, dim)

    def forward(self, x: Tensor, flip_indices: Tensor | None = None,
                cu_seqlens: Tensor | None = None) -> Tensor:
        """Static rows ``(B, S, D)`` with a per-row ``flip_indices`` (B, S), or the
        packed augmented stream ``(1, T_aug, D)`` with its ``cu_seqlens`` (fused)."""
        x_norm = self.norm(x).contiguous()
        if cu_seqlens is not None:
            x_fwd, x_bwd = fused_bidi_scan_packed(self, x_norm, cu_seqlens)
        else:
            gather_idx = flip_indices.unsqueeze(-1).expand_as(x_norm)
            x_fwd = self.forward_mamba(x_norm)
            x_bwd = torch.gather(self.backward_mamba(torch.gather(x_norm, 1, gather_idx).contiguous()),
                                 1, gather_idx)
        gate = torch.sigmoid(self.gate(x_norm))
        return x + (gate * x_fwd + (1 - gate) * x_bwd)


class BidirectionalMambaCLSEncoder(nn.Module):
    """Bidirectional Mamba-2 encoder; returns ``(hit_output, pooled (B, 2 * dim))``.

    ``residual_depth_init`` rescales every ``out_proj.weight`` by
    ``1 / sqrt(2 * num_layers)`` (one factor per residual-stream write).
    """

    def __init__(self, num_layers: int, dim: int, d_state: int = 64, d_conv: int = 1,
                 expand: int = 2, headdim: int = 32, cls_init_scale: float = 0.02,
                 residual_depth_init: bool = True):
        super().__init__()
        self.num_layers = num_layers
        self.dim = dim
        self.cls_fwd = nn.Parameter(torch.randn(1, 1, dim) * cls_init_scale)
        self.cls_bwd = nn.Parameter(torch.randn(1, 1, dim) * cls_init_scale)
        kw = dict(dim=dim, d_state=d_state, d_conv=d_conv, expand=expand, headdim=headdim)
        self.layers = nn.ModuleList([BidirectionalMambaLayer(**kw) for _ in range(num_layers - 1)])
        self.final_layer = BidirectionalMambaLayer(**kw)
        self.final_norm = nn.RMSNorm(dim)
        self.cls_norm = nn.RMSNorm(dim)
        if residual_depth_init:
            scale = 1.0 / math.sqrt(2.0 * num_layers)
            with torch.no_grad():
                for layer in [*self.layers, self.final_layer]:
                    layer.forward_mamba.out_proj.weight.mul_(scale)
                    layer.backward_mamba.out_proj.weight.mul_(scale)
        self._static_core_fn = torch.compile(self._static_core, dynamic=False)
        self._packed_core_fn = torch.compile(self._packed_core, dynamic=True)

    @property
    def pool_dim(self) -> int:
        return 2 * self.dim

    def forward(self, x: Tensor, cu_seqlens: Tensor, seq_idx: Tensor | None = None):
        if self.training or not x.is_cuda or not fused_kernels_enabled():
            return self._forward_static(x, cu_seqlens)
        return self._forward_packed(x, seq_idx, cu_seqlens)

    # -- training / reference path: static rows ------------------------------

    def _static_core(self, x_aug: Tensor, valid: Tensor, flip_idx: Tensor, cls_fwd_pos: Tensor):
        vm = valid.unsqueeze(-1).to(x_aug.dtype)
        for layer in self.layers:
            x_aug = layer(x_aug, flip_indices=flip_idx) * vm      # keep pad rows at zero
        x_aug = self.final_layer(x_aug, flip_indices=flip_idx)
        rows = torch.arange(x_aug.shape[0], device=x_aug.device)
        cls_fwd_out = self.cls_norm(x_aug[rows, cls_fwd_pos, :])
        cls_bwd_out = self.cls_norm(x_aug[:, 0, :])
        return self.final_norm(x_aug) * vm, cls_fwd_out, cls_bwd_out

    def _forward_static(self, x: Tensor, cu_seqlens: Tensor):
        aux = build_static_aux(cu_seqlens)
        x_pad, row, col = packed_to_padded_static(x, cu_seqlens, aux)
        S = x_pad.shape[1]
        # CLS tokens placed with a one-hot add (their slots are zero), not an
        # in-place write, so their gradients are kept
        p = torch.arange(S, device=x_pad.device)
        onehot_bwd = (p == 0).to(x_pad.dtype).view(1, S, 1)
        onehot_fwd = (p.unsqueeze(0) == aux["cls_fwd_pos"].unsqueeze(1)).to(x_pad.dtype).unsqueeze(-1)
        x_aug = x_pad + onehot_bwd * self.cls_bwd.to(x_pad.dtype) + onehot_fwd * self.cls_fwd.to(x_pad.dtype)
        x_aug, cls_fwd_out, cls_bwd_out = self._static_core_fn(
            x_aug, aux["valid"], aux["flip_idx"], aux["cls_fwd_pos"])
        x_hits = x_aug[row, col].unsqueeze(0)
        pooled = torch.cat([cls_fwd_out, cls_bwd_out], dim=-1)
        if self.training:
            # keep every parameter in the autograd graph (DDP), numerically a no-op
            pooled = pooled + 0.0 * x_hits.float().sum()
        return x_hits, pooled

    # -- fused inference path: packed augmented stream -----------------------

    def _packed_core(self, x_aug: Tensor, aug_cu: Tensor, cls_fwd_positions: Tensor,
                     cls_bwd_positions: Tensor):
        for layer in self.layers:
            x_aug = layer(x_aug, cu_seqlens=aug_cu)
        x_aug = self.final_layer(x_aug, cu_seqlens=aug_cu)
        cls_fwd_out = self.cls_norm(x_aug[0, cls_fwd_positions, :])
        cls_bwd_out = self.cls_norm(x_aug[0, cls_bwd_positions, :])
        return self.final_norm(x_aug), cls_fwd_out, cls_bwd_out

    def _forward_packed(self, x: Tensor, seq_idx: Tensor | None, cu_seqlens: Tensor):
        cu = cu_seqlens.to(torch.long)
        B = cu.shape[0] - 1
        total_L, D = x.shape[1], x.shape[2]
        device = x.device
        seg_arange = torch.arange(B, device=device, dtype=torch.long)
        arange_total = torch.arange(total_L, device=device, dtype=torch.long)
        seq_idx_flat = (torch.bucketize(arange_total, cu[1:], right=True) if seq_idx is None
                        else seq_idx[0].to(torch.long))
        # augmented positions: each segment gains cls_bwd at its start and cls_fwd at its end
        cls_bwd_positions = cu[:-1] + 2 * seg_arange
        cls_fwd_positions = cu[1:] + 2 * seg_arange + 1
        aug_token_positions = arange_total + 2 * seq_idx_flat + 1
        all_values = torch.cat([x[0], self.cls_bwd[0, 0].to(x.dtype).unsqueeze(0).expand(B, D),
                                self.cls_fwd[0, 0].to(x.dtype).unsqueeze(0).expand(B, D)], dim=0)
        inv_perm = torch.argsort(torch.cat([aug_token_positions, cls_bwd_positions, cls_fwd_positions]))
        x_aug = all_values[inv_perm].unsqueeze(0).contiguous()
        aug_cu = F.pad(torch.cumsum((cu[1:] - cu[:-1]) + 2, dim=0), (1, 0))
        x_aug, cls_fwd_out, cls_bwd_out = self._packed_core_fn(
            x_aug, aug_cu, cls_fwd_positions, cls_bwd_positions)
        return x_aug[0:1, aug_token_positions, :], torch.cat([cls_fwd_out, cls_bwd_out], dim=-1)
