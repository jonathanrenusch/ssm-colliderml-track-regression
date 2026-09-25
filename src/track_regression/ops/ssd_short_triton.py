"""Fused Triton kernels for the short-sequence (L <= 22) Mamba-2 block on the packed stream.

The stock ``mamba_ssm`` chunked scan is built for sequences of thousands of
tokens.  For a track of at most 20 hits plus two CLS tokens the SSD
recurrence is a single chunk, i.e. one dense lower-triangular L x L matrix
product per (track, head).  :func:`ssd_short_fwd_packed` evaluates, for every
track (one program per track, heads looped in-kernel): depthwise causal conv
+ SiLU -> dt softplus -> decay matrix in registers -> ``M = decay o (C B^T)``
-> ``Y = M @ (x dt) + D x``, reading its segment of the packed in-projection
through ``cu_seqlens``; the backward direction is a within-segment flip done
in-kernel.  Tracks with at most 16 augmented tokens run in a BL = 16 launch,
the rest in BL = 32.  :func:`gated_rmsnorm` fuses ``RMSNorm(y * silu(z)) * w``.

Numerics: every load is converted to fp32 in registers (the projections may
be fp16 under autocast), both ``tl.dot`` calls use IEEE fp32, the store casts
back to the storage dtype.  The ops are registered with
``torch.library.custom_op`` so they are opaque to ``torch.compile``.
``_packed_scan_torch_ref`` is the pure-PyTorch reference used by the tests.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # HPP = heads per program: smaller HPP -> fewer registers/program ->
        # higher occupancy, at the cost of recomputing conv'd B/C + G per
        # head-group. The autotuner decides per shape.
        triton.Config({"HPP": 0}, num_warps=4),
        triton.Config({"HPP": 0}, num_warps=8),
        triton.Config({"HPP": 0}, num_warps=4, maxnreg=128),
        triton.Config({"HPP": 0}, num_warps=8, maxnreg=96),
        triton.Config({"HPP": 0}, num_warps=8, maxnreg=128),
        triton.Config({"HPP": 4}, num_warps=2),
        triton.Config({"HPP": 4}, num_warps=4),
        triton.Config({"HPP": 4}, num_warps=4, maxnreg=128),
        triton.Config({"HPP": 2}, num_warps=2),
        triton.Config({"HPP": 2}, num_warps=4),
        triton.Config({"HPP": 2}, num_warps=2, maxnreg=168),
    ],
    key=["H", "P", "N", "BL", "REVERSE", "USE_IDX"],
)
@triton.jit
def _ssd_packed_kernel(
    zxbcdt_ptr,  # (T_aug, DPROJ) packed rows
    convw_ptr, convb_ptr, dtb_ptr, alog_ptr, d_ptr,
    cu_ptr,      # (B+1,) int64/int32 augmented cumulative segment ends
    out_ptr,     # (T_aug, HP)
    idx_ptr,     # (n_programs,) int32 track ids when USE_IDX (length-bucketed launches); ignored otherwise
    DPROJ_TOT,
    H: tl.constexpr,
    P: tl.constexpr,
    N: tl.constexpr,
    DCONV: tl.constexpr,
    BL: tl.constexpr,
    REVERSE: tl.constexpr,
    HPP: tl.constexpr,  # heads per program; 0 means all H in one program
    USE_IDX: tl.constexpr,   # 1: program -> track via idx_ptr (length-bucketed launch); 0: program == track
):
    pid_t = tl.program_id(0)
    if USE_IDX:
        pid_t = tl.load(idx_ptr + pid_t).to(tl.int32)
    HREAL: tl.constexpr = H if HPP == 0 else HPP
    h0 = tl.program_id(1) * HREAL

    HP: tl.constexpr = H * P
    XBC_OFF: tl.constexpr = HP
    DT_OFF: tl.constexpr = 2 * HP + 2 * N

    base = tl.load(cu_ptr + pid_t).to(tl.int64)
    nxt = tl.load(cu_ptr + pid_t + 1).to(tl.int64)
    Lt = (nxt - base).to(tl.int32)  # segment length (Lr + 2), <= BL

    offs_l = tl.arange(0, BL)
    lmask = offs_l < Lt
    last = Lt - 1

    offs_p = tl.arange(0, P)
    offs_n = tl.arange(0, N)

    # ---- shared across heads: conv'd B, C and G, computed once -------------
    # rows may be fp32, fp16 or bf16: every load is converted to fp32 in registers
    b_acc = tl.zeros((BL, N), dtype=tl.float32)
    c_acc = tl.zeros((BL, N), dtype=tl.float32)
    for k in tl.static_range(DCONV):
        row = offs_l - (DCONV - 1) + k          # logical scan-order index
        rmask = (row >= 0) & lmask
        if REVERSE:
            row = tl.where(row <= last, last - row, row)
        roff = (base + row.to(tl.int64))[:, None] * DPROJ_TOT
        wb = tl.load(convw_ptr + (HP + offs_n) * DCONV + k)
        wc = tl.load(convw_ptr + (HP + N + offs_n) * DCONV + k)
        b_acc += wb[None, :] * tl.load(zxbcdt_ptr + roff + (XBC_OFF + HP + offs_n)[None, :],
                                       mask=rmask[:, None], other=0.0).to(tl.float32)
        c_acc += wc[None, :] * tl.load(zxbcdt_ptr + roff + (XBC_OFF + HP + N + offs_n)[None, :],
                                       mask=rmask[:, None], other=0.0).to(tl.float32)
    b_acc += tl.load(convb_ptr + HP + offs_n)[None, :]
    c_acc += tl.load(convb_ptr + HP + N + offs_n)[None, :]
    Bm = b_acc * tl.sigmoid(b_acc)
    Cm = c_acc * tl.sigmoid(c_acc)
    G = tl.dot(Cm, tl.trans(Bm), input_precision="ieee")

    causal = (offs_l[:, None] >= offs_l[None, :]) & lmask[:, None] & lmask[None, :]

    dt_row = offs_l
    if REVERSE:
        dt_row = tl.where(offs_l <= last, last - offs_l, offs_l)
    out_row = out_ptr + (base + dt_row.to(tl.int64))[:, None] * HP

    for hh in tl.static_range(HREAL):
        h = h0 + hh
        x_acc = tl.zeros((BL, P), dtype=tl.float32)
        for k in tl.static_range(DCONV):
            row = offs_l - (DCONV - 1) + k
            rmask = (row >= 0) & lmask
            if REVERSE:
                row = tl.where(row <= last, last - row, row)
            roff = (base + row.to(tl.int64))[:, None] * DPROJ_TOT
            wx = tl.load(convw_ptr + (h * P + offs_p) * DCONV + k)
            x_acc += wx[None, :] * tl.load(
                zxbcdt_ptr + roff + (XBC_OFF + h * P + offs_p)[None, :],
                mask=rmask[:, None], other=0.0).to(tl.float32)
        x_acc += tl.load(convb_ptr + h * P + offs_p)[None, :]
        x = x_acc * tl.sigmoid(x_acc)

        dt_raw = tl.load(zxbcdt_ptr + (base + dt_row.to(tl.int64)) * DPROJ_TOT + DT_OFF + h,
                         mask=lmask, other=0.0).to(tl.float32)
        v = dt_raw + tl.load(dtb_ptr + h)
        dt = tl.where(v <= 20.0, tl.log(1.0 + tl.exp(v)), v)
        dt = tl.where(lmask, dt, 0.0)

        A = -tl.exp(tl.load(alog_ptr + h))
        cumA = tl.cumsum(dt * A, 0)
        seg = cumA[:, None] - cumA[None, :]
        Lmat = tl.where(causal, tl.exp(seg), 0.0)
        M = Lmat * G

        xdt = x * dt[:, None]
        Y = tl.dot(M, xdt, input_precision="ieee")
        Y += tl.load(d_ptr + h) * x
        tl.store(out_row + h * P + offs_p[None, :], Y, mask=lmask[:, None])


_BUCKET_CACHE: dict = {}


def _length_buckets(cu: torch.Tensor):
    """(idx16, idx32) int32 track ids for the two BL launches; cached per cu tensor
    so the data-dependent split (one device sync) happens once per forward."""
    key = (cu.data_ptr(), cu.shape[0])
    hit = _BUCKET_CACHE.get(key)
    if hit is not None and hit[0] is cu:
        return hit[1], hit[2]
    lens = cu[1:] - cu[:-1]
    idx16 = torch.nonzero(lens <= 16, as_tuple=False).squeeze(1).to(torch.int32)
    idx32 = torch.nonzero(lens > 16, as_tuple=False).squeeze(1).to(torch.int32)
    if len(_BUCKET_CACHE) > 64:
        _BUCKET_CACHE.clear()
    _BUCKET_CACHE[key] = (cu, idx16, idx32)
    return idx16, idx32


@torch.library.custom_op("track_regression::ssd_short_fwd_packed", mutates_args=())
def ssd_short_fwd_packed(
    zxbcdt_rows: torch.Tensor,   # (T_aug, d_in_proj) packed rows
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor,
    dt_bias: torch.Tensor,
    A_log: torch.Tensor,
    D: torch.Tensor,
    cu_seqlens_aug: torch.Tensor,  # (B+1,)
    nheads: int,
    headdim: int,
    d_state: int,
    reverse: bool = False,
) -> torch.Tensor:
    """Packed-stream single-chunk SSD scan (pre-norm Y rows, (T_aug, H*P)).

    ``zxbcdt_rows`` may be fp32, fp16 or bf16; the output follows its dtype
    while the kernel computes in fp32 regardless (loads are converted in
    registers).  The weights stay fp32 (they are the module's parameters).
    """
    T, dproj = zxbcdt_rows.shape
    H, P, N = nheads, headdim, d_state
    assert zxbcdt_rows.dtype in (torch.float32, torch.float16, torch.bfloat16), zxbcdt_rows.dtype
    w = conv_weight.reshape(conv_weight.shape[0], -1).contiguous()
    zx = zxbcdt_rows.contiguous()
    cu = cu_seqlens_aug.to(device=zx.device, dtype=torch.int64).contiguous()
    out = torch.empty(T, H * P, device=zx.device, dtype=zx.dtype)
    args = (zx, w, conv_bias.contiguous(), dt_bias.contiguous(), A_log.contiguous(), D.contiguous(), cu, out)
    idx16, idx32 = _length_buckets(cu)
    for idx, bl in ((idx16, 16), (idx32, 32)):
        n = int(idx.shape[0])
        if n == 0:
            continue
        grid = lambda META, n=n: (n, 1 if META["HPP"] == 0 else H // META["HPP"])  # noqa: E731
        _ssd_packed_kernel[grid](
            *args, idx, dproj, H=H, P=P, N=N, DCONV=w.shape[-1], BL=bl, REVERSE=reverse,
            USE_IDX=1,
        )
    return out


@ssd_short_fwd_packed.register_fake
def _(zxbcdt_rows, conv_weight, conv_bias, dt_bias, A_log, D, cu_seqlens_aug,
      nheads, headdim, d_state, reverse=False):
    T, _ = zxbcdt_rows.shape
    return zxbcdt_rows.new_empty(T, nheads * headdim)


# ---------------------------------------------------------------------------
# Fused gated RMSNorm: out = RMSNorm(y * silu(z)) * w   (norm_before_gate=False)
# One row per program, single pass.
# ---------------------------------------------------------------------------


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
    ],
    key=["DSSM"],
)
@triton.jit
def _gated_rmsnorm_kernel(
    y_ptr,      # (R, DSSM) fp32/fp16/bf16 rows (the scan output)
    z_ptr,      # (R, Z_STRIDE) same dtype as the projection — z slice starts the row
    w_ptr,      # (DSSM,) fp32 parameter
    out_ptr,    # (R, DSSM) same dtype as y
    Z_STRIDE,   # row stride of the z tensor (d_in_proj when z is a view)
    EPS,
    DSSM: tl.constexpr,
    DBLK: tl.constexpr,  # next power of two >= DSSM (tl.arange needs pow2)
):
    row = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, DBLK)
    cmask = offs < DSSM
    # Convert on load: Triton's exp/sqrt are fp32/fp64-only, and the gate,
    # the mean square and the rsqrt must not run in half precision anyway.
    y = tl.load(y_ptr + row * DSSM + offs, mask=cmask, other=0.0).to(tl.float32)
    z = tl.load(z_ptr + row * Z_STRIDE + offs, mask=cmask, other=0.0).to(tl.float32)
    g = y * (z * tl.sigmoid(z))
    ms = tl.sum(g * g, 0) / DSSM
    rstd = 1.0 / tl.sqrt(ms + EPS)
    w = tl.load(w_ptr + offs, mask=cmask, other=0.0).to(tl.float32)
    tl.store(out_ptr + row * DSSM + offs, g * rstd * w, mask=cmask)


@torch.library.custom_op("track_regression::gated_rmsnorm", mutates_args=())
def gated_rmsnorm(y: torch.Tensor, z_rows: torch.Tensor, weight: torch.Tensor,
                  eps: float) -> torch.Tensor:
    """out = RMSNorm(y * silu(z)) * weight, rowwise over the last dim.

    ``z_rows`` is a (R, K) tensor whose first DSSM columns are z -- pass
    ``zxbcdt.view(R, d_in_proj)`` directly (z is its leading slice), so no
    slice copy is materialised.  The maths is fp32 in-kernel and the output
    follows ``y``'s dtype.
    """
    d = y.shape[-1]
    assert y.is_contiguous() and z_rows.stride(-1) == 1
    assert y.dtype in (torch.float32, torch.float64, torch.float16, torch.bfloat16), y.dtype
    y2 = y.view(-1, d)
    assert z_rows.shape[0] == y2.shape[0]
    out = torch.empty_like(y2)
    _gated_rmsnorm_kernel[(y2.shape[0],)](
        y2, z_rows, weight.contiguous(), out,
        z_rows.stride(0), eps, DSSM=d, DBLK=triton.next_power_of_2(d),
    )
    return out.view_as(y)


@gated_rmsnorm.register_fake
def _(y, z, weight, eps):
    return torch.empty_like(y)


def _packed_scan_torch_ref(zx_rows, conv_w, conv_b, dt_bias, A_log, Dp,
                           cu, H, P, N, reverse):
    """Pure-PyTorch reference of :func:`ssd_short_fwd_packed` (tests)."""
    import torch.nn.functional as F

    HP = H * P
    dproj = zx_rows.shape[1]
    dconv = conv_w.reshape(conv_w.shape[0], -1).shape[-1]
    w2 = conv_w.reshape(conv_w.shape[0], -1)  # (HP+2N, K)

    lengths = cu[1:] - cu[:-1]                # (B,) segment lengths Lt
    Bt = lengths.shape[0]
    S = 32
    device = zx_rows.device
    tok = torch.arange(zx_rows.shape[0], device=device)
    row = torch.bucketize(tok, cu[1:], right=True)
    col = tok - cu[row]
    pad = zx_rows.new_zeros(Bt, S, dproj).index_put((row, col), zx_rows)

    p = torch.arange(S, device=device).unsqueeze(0)
    valid = p < lengths.unsqueeze(1)
    if reverse:
        flip = torch.where(p < lengths.unsqueeze(1), lengths.unsqueeze(1) - 1 - p, p)
        pad = torch.gather(pad, 1, flip.unsqueeze(-1).expand_as(pad))

    xBC_raw = pad[..., HP:HP + HP + 2 * N]
    dt_raw = pad[..., HP + HP + 2 * N:]

    xp = F.pad(xBC_raw, (0, 0, dconv - 1, 0))
    conv = xp[:, 0:S, :] * w2[:, 0]
    for k in range(1, dconv):
        conv = conv + xp[:, k:k + S, :] * w2[:, k]
    conv = F.silu(conv + conv_b)
    x, Bm, Cm = torch.split(conv, [HP, N, N], dim=-1)

    dt = F.softplus(dt_raw.float() + dt_bias.float())          # (Bt,S,H)
    dt = torch.where(valid.unsqueeze(-1), dt, torch.zeros_like(dt))
    A = -torch.exp(A_log.float())
    cumA = torch.cumsum(dt * A, dim=1)
    diff = cumA.unsqueeze(2) - cumA.unsqueeze(1)               # (Bt,S,S,H)
    tril = torch.ones(S, S, dtype=torch.bool, device=device).tril()
    m = tril.unsqueeze(0).unsqueeze(-1) & (valid.unsqueeze(2) & valid.unsqueeze(1)).unsqueeze(-1)
    Lmat = torch.where(m, torch.exp(diff), torch.zeros_like(diff))
    G = torch.matmul(Cm.float(), Bm.float().transpose(1, 2))
    M = Lmat * G.unsqueeze(-1)

    xh = x.view(Bt, S, H, P).float()
    xdt = xh * dt.unsqueeze(-1)
    Y = torch.matmul(M.permute(0, 3, 1, 2), xdt.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)
    Y = Y + Dp.float().view(1, 1, H, 1) * xh
    Y = Y.reshape(Bt, S, HP).to(zx_rows.dtype)

    if reverse:
        Y = torch.gather(Y, 1, flip.unsqueeze(-1).expand_as(Y))
    return Y[row, col]
