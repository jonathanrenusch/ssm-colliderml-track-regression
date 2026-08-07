"""V4: fused Triton kernel for the short-sequence (L<=32) Mamba2 block core.

One kernel launch per (layer, direction) evaluates, for every (track, head)
program: depthwise causal conv (width d_conv) + SiLU -> dt softplus (threshold
20) -> single-chunk SSD decay matrix in registers -> M = decay o (C B^T) via
IEEE tl.dot -> Y = M @ (x*dt) + D o x -> one write of Y to HBM.  None of the
(B, H, L, L) intermediates that dominate the pure-torch V3 profile ever touch
global memory.

Grid: (B, H) — batch on axis 0 (limit 2^31-1), so the stock kernel's
batch*nchunks <= 65535 ceiling is gone by construction.  All pointer
arithmetic is explicitly int64 (at 1M tracks the zxbcdt tensor exceeds
int32 element indexing).

Numerics: IEEE fp32 throughout (`input_precision="ieee"` on both tl.dot
calls); softplus mirrors the stock kernel's threshold-20 linearisation;
exp/cumsum in fp32.  The cross-head gated RMSNorm and the two projections
deliberately stay OUTSIDE (cuBLAS + torch.compile fuse them) — the norm
needs all H*P channels, which would force cross-program reduction.

The op is registered via torch.library.custom_op (with a fake impl) so a
torch.compile'd caller treats it as an opaque node and fuses the
surrounding gate/norm/projection elementwise work around it.

Padding contract (Scheme A): pads strictly trail.  The kernel additionally
masks rows >= L_valid via `lmask`, and the decay matrix is masked to the
valid causal triangle, so pad rows write zeros and contribute nothing.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
        triton.Config({}, num_warps=2, num_stages=2),
        triton.Config({}, num_warps=4, num_stages=2),
    ],
    key=["H", "P", "N", "BL", "REVERSE"],
)
@triton.jit
def _ssd_short_fwd_kernel(
    zxbcdt_ptr,  # (B, L, DPROJ_TOT) fp32, contiguous; this direction's slice
                 # starts at ZX_OFF within the last dim
    convw_ptr,   # (HP + 2N, DCONV) fp32
    convb_ptr,   # (HP + 2N,) fp32
    dtb_ptr,     # (H,) fp32
    alog_ptr,    # (H,) fp32
    d_ptr,       # (H,) fp32
    lens_ptr,    # (B,) int32 — per-row last valid index (Lr + 1); only read
                 # when REVERSE=1
    out_ptr,     # (B, L, HP) fp32
    L,           # runtime static sequence length (<= BL)
    DPROJ_TOT,   # row stride of zxbcdt (= DPROJ for a plain call; 2*DPROJ
                 # when both directions share one fused in_proj output)
    ZX_OFF,      # channel offset of this direction's zxbcdt slice
    H: tl.constexpr,
    P: tl.constexpr,
    N: tl.constexpr,
    DCONV: tl.constexpr,
    BL: tl.constexpr,
    REVERSE: tl.constexpr,
):
    pid_b = tl.program_id(0).to(tl.int64)
    pid_h = tl.program_id(1)

    HP: tl.constexpr = H * P
    XBC_OFF: tl.constexpr = HP        # xBC starts right after z
    DT_OFF: tl.constexpr = 2 * HP + 2 * N

    offs_l = tl.arange(0, BL)
    lmask = offs_l < L
    # pid_b is int64; (L * DPROJ_TOT) stays int32 — the product promotes to
    # int64 (at 1M tracks the element offset exceeds int32).
    row_base = zxbcdt_ptr + pid_b * (L * DPROJ_TOT) + ZX_OFF

    # Valid-prefix flip, in-kernel: logical position l reads physical row
    # last-l for l <= last (pads keep identity — their outputs are garbage
    # that trails and is re-zeroed by the caller, same as the forward pass).
    if REVERSE:
        last = tl.load(lens_ptr + pid_b)
    else:
        last = 0  # unused

    # ---- depthwise causal conv + SiLU, in registers ------------------------
    # x channels of this head: XBC_OFF + pid_h*P + [0, P)
    offs_p = tl.arange(0, P)
    offs_n = tl.arange(0, N)
    xch = XBC_OFF + pid_h * P + offs_p            # (P,)
    bch = XBC_OFF + HP + offs_n                   # (N,) B channels
    cch = XBC_OFF + HP + N + offs_n               # (N,) C channels

    x_acc = tl.zeros((BL, P), dtype=tl.float32)
    b_acc = tl.zeros((BL, N), dtype=tl.float32)
    c_acc = tl.zeros((BL, N), dtype=tl.float32)
    for k in tl.static_range(DCONV):
        row = offs_l - (DCONV - 1) + k            # logical (scan-order) index
        rmask = (row >= 0) & lmask
        if REVERSE:
            row = tl.where(row <= last, last - row, row)
        roff = row.to(tl.int64)[:, None] * DPROJ_TOT
        wx = tl.load(convw_ptr + (xch - XBC_OFF) * DCONV + k)
        wb = tl.load(convw_ptr + (bch - XBC_OFF) * DCONV + k)
        wc = tl.load(convw_ptr + (cch - XBC_OFF) * DCONV + k)
        x_acc += wx[None, :] * tl.load(row_base + roff + xch[None, :],
                                       mask=rmask[:, None], other=0.0)
        b_acc += wb[None, :] * tl.load(row_base + roff + bch[None, :],
                                       mask=rmask[:, None], other=0.0)
        c_acc += wc[None, :] * tl.load(row_base + roff + cch[None, :],
                                       mask=rmask[:, None], other=0.0)
    x_acc += tl.load(convb_ptr + (xch - XBC_OFF))[None, :]
    b_acc += tl.load(convb_ptr + (bch - XBC_OFF))[None, :]
    c_acc += tl.load(convb_ptr + (cch - XBC_OFF))[None, :]
    x = x_acc * tl.sigmoid(x_acc)  # SiLU
    Bm = b_acc * tl.sigmoid(b_acc)
    Cm = c_acc * tl.sigmoid(c_acc)

    # ---- dt = softplus(dt_raw + dt_bias), threshold-20 like the stock kernel
    dt_row = offs_l
    if REVERSE:
        dt_row = tl.where(offs_l <= last, last - offs_l, offs_l)
    dt_raw = tl.load(row_base + dt_row.to(tl.int64) * DPROJ_TOT + DT_OFF + pid_h,
                     mask=lmask, other=0.0)
    dt_bias = tl.load(dtb_ptr + pid_h)
    v = dt_raw + dt_bias
    dt = tl.where(v <= 20.0, tl.log(1.0 + tl.exp(v)), v)
    dt = tl.where(lmask, dt, 0.0)  # keep cumsum finite past L

    # ---- single-chunk SSD decay matrix, in registers ------------------------
    A = -tl.exp(tl.load(alog_ptr + pid_h))
    cumA = tl.cumsum(dt * A, 0)                       # (BL,)
    seg = cumA[:, None] - cumA[None, :]               # (BL, BL)
    causal = (offs_l[:, None] >= offs_l[None, :]) & lmask[:, None] & lmask[None, :]
    Lmat = tl.where(causal, tl.exp(seg), 0.0)

    # G = <C_l, B_s> shared across heads (ngroups=1); exact IEEE fp32
    # (a TF32 tl.dot variant was measured night 2: slower AND noisier).
    G = tl.dot(Cm, tl.trans(Bm), input_precision="ieee")
    M = Lmat * G

    # ---- Y = M @ (x*dt) + D o x --------------------------------------------
    xdt = x * dt[:, None]
    Y = tl.dot(M, xdt, input_precision="ieee")
    Dh = tl.load(d_ptr + pid_h)
    Y += Dh * x

    # ---- one masked write (REVERSE un-flips: logical row l lands at its
    # physical position, so the caller never gathers) -------------------------
    st_row = offs_l
    if REVERSE:
        st_row = tl.where(offs_l <= last, last - offs_l, offs_l)
    out_row = out_ptr + pid_b * (L * HP) + st_row.to(tl.int64)[:, None] * HP
    tl.store(out_row + pid_h * P + offs_p[None, :], Y, mask=lmask[:, None])


@torch.library.custom_op("track_regression::ssd_short_fwd", mutates_args=())
def ssd_short_fwd(
    zxbcdt: torch.Tensor,
    conv_weight: torch.Tensor,  # (conv_dim, 1, DCONV) or (conv_dim, DCONV)
    conv_bias: torch.Tensor,
    dt_bias: torch.Tensor,
    A_log: torch.Tensor,
    D: torch.Tensor,
    nheads: int,
    headdim: int,
    d_state: int,
    lens: torch.Tensor | None = None,
    reverse: bool = False,
    zx_offset: int = 0,
    zx_row_stride: int = 0,
    kernel2: bool = False,
) -> torch.Tensor:
    """Fused conv+SiLU+dt+SSD-dual+D-skip. Returns pre-norm Y (B, L, H*P).

    reverse=True evaluates the backward scan direction entirely in-kernel
    (per-row valid-prefix flip using ``lens`` = Lr+1), writing outputs back
    at their physical positions — no gather passes.  ``zx_offset`` /
    ``zx_row_stride`` allow both directions to slice a single fused
    in_proj output (B, L, 2*DPROJ).
    """
    B, L, _ = zxbcdt.shape
    H, P, N = nheads, headdim, d_state
    assert L <= 32, f"kernel is specialised for L<=32, got {L}"
    w = conv_weight.reshape(conv_weight.shape[0], -1).contiguous()
    zx = zxbcdt.contiguous()
    row_stride = zx_row_stride or zx.shape[-1]
    if reverse:
        assert lens is not None
        lens_t = lens.to(device=zx.device, dtype=torch.int32).contiguous()
    else:
        lens_t = torch.empty(0, device=zx.device, dtype=torch.int32)
    out = torch.empty(B, L, H * P, device=zx.device, dtype=zx.dtype)
    if kernel2:
        # One program per track, heads looped in-kernel (B/C + G once).
        _ssd_short_fwd_kernel2[(B,)](
            zx, w, conv_bias.contiguous(), dt_bias.contiguous(),
            A_log.contiguous(), D.contiguous(), lens_t, out,
            L, row_stride, zx_offset,
            H=H, P=P, N=N, DCONV=w.shape[-1], BL=32, REVERSE=reverse,
        )
        return out
    grid = (B, H)
    _ssd_short_fwd_kernel[grid](
        zx, w, conv_bias.contiguous(), dt_bias.contiguous(),
        A_log.contiguous(), D.contiguous(), lens_t, out,
        L, row_stride, zx_offset,
        H=H, P=P, N=N, DCONV=w.shape[-1], BL=32, REVERSE=reverse,
    )
    return out


@ssd_short_fwd.register_fake
def _(zxbcdt, conv_weight, conv_bias, dt_bias, A_log, D, nheads, headdim,
      d_state, lens=None, reverse=False, zx_offset=0, zx_row_stride=0,
      kernel2=False):
    B, L, _ = zxbcdt.shape
    return zxbcdt.new_empty(B, L, nheads * headdim)


# ---------------------------------------------------------------------------
# Kernel 2 ("v5"): one program per TRACK, loop over heads in-kernel.
# Motivation (ncu, night 1): kernel 1's (B, H) grid re-reads the shared B/C
# channels once per head (8x) and re-derives G per head; with L1/TEX at 95%
# throughput that redundancy IS the bottleneck.  Here B/C are conv'd and
# G = C B^T computed ONCE per track; each head's output goes to its own
# disjoint out columns (no cross-head reduction needed pre-norm).
# ---------------------------------------------------------------------------


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
        triton.Config({}, num_warps=4, num_stages=2),
        triton.Config({}, num_warps=4, num_stages=1),
        triton.Config({}, num_warps=4, maxnreg=128),
        triton.Config({}, num_warps=8, maxnreg=96),
        triton.Config({}, num_warps=8, maxnreg=128),
    ],
    key=["H", "P", "N", "BL", "REVERSE"],
)
@triton.jit
def _ssd_short_fwd_kernel2(
    zxbcdt_ptr, convw_ptr, convb_ptr, dtb_ptr, alog_ptr, d_ptr, lens_ptr,
    out_ptr,
    L, DPROJ_TOT, ZX_OFF,
    H: tl.constexpr,
    P: tl.constexpr,
    N: tl.constexpr,
    DCONV: tl.constexpr,
    BL: tl.constexpr,
    REVERSE: tl.constexpr,
):
    pid_b = tl.program_id(0).to(tl.int64)

    HP: tl.constexpr = H * P
    XBC_OFF: tl.constexpr = HP
    DT_OFF: tl.constexpr = 2 * HP + 2 * N

    offs_l = tl.arange(0, BL)
    lmask = offs_l < L
    row_base = zxbcdt_ptr + pid_b * (L * DPROJ_TOT) + ZX_OFF

    if REVERSE:
        last = tl.load(lens_ptr + pid_b)
    else:
        last = 0

    offs_p = tl.arange(0, P)
    offs_n = tl.arange(0, N)

    # ---- shared across heads: conv'd B, C and the Gram matrix G -------------
    b_acc = tl.zeros((BL, N), dtype=tl.float32)
    c_acc = tl.zeros((BL, N), dtype=tl.float32)
    for k in tl.static_range(DCONV):
        row = offs_l - (DCONV - 1) + k
        rmask = (row >= 0) & lmask
        if REVERSE:
            row = tl.where(row <= last, last - row, row)
        roff = row.to(tl.int64)[:, None] * DPROJ_TOT
        wb = tl.load(convw_ptr + (HP + offs_n) * DCONV + k)
        wc = tl.load(convw_ptr + (HP + N + offs_n) * DCONV + k)
        b_acc += wb[None, :] * tl.load(row_base + roff + (XBC_OFF + HP + offs_n)[None, :],
                                       mask=rmask[:, None], other=0.0)
        c_acc += wc[None, :] * tl.load(row_base + roff + (XBC_OFF + HP + N + offs_n)[None, :],
                                       mask=rmask[:, None], other=0.0)
    b_acc += tl.load(convb_ptr + HP + offs_n)[None, :]
    c_acc += tl.load(convb_ptr + HP + N + offs_n)[None, :]
    Bm = b_acc * tl.sigmoid(b_acc)
    Cm = c_acc * tl.sigmoid(c_acc)
    G = tl.dot(Cm, tl.trans(Bm), input_precision="ieee")  # (BL, BL), once

    causal = (offs_l[:, None] >= offs_l[None, :]) & lmask[:, None] & lmask[None, :]

    dt_row = offs_l
    if REVERSE:
        dt_row = tl.where(offs_l <= last, last - offs_l, offs_l)
    st_row = dt_row  # same permutation un-flips the store

    out_row = out_ptr + pid_b * (L * HP) + st_row.to(tl.int64)[:, None] * HP

    # ---- per-head: conv x, dt, decay, dual, D-skip, store -------------------
    for h in tl.static_range(H):
        x_acc = tl.zeros((BL, P), dtype=tl.float32)
        for k in tl.static_range(DCONV):
            row = offs_l - (DCONV - 1) + k
            rmask = (row >= 0) & lmask
            if REVERSE:
                row = tl.where(row <= last, last - row, row)
            roff = row.to(tl.int64)[:, None] * DPROJ_TOT
            wx = tl.load(convw_ptr + (h * P + offs_p) * DCONV + k)
            x_acc += wx[None, :] * tl.load(
                row_base + roff + (XBC_OFF + h * P + offs_p)[None, :],
                mask=rmask[:, None], other=0.0)
        x_acc += tl.load(convb_ptr + h * P + offs_p)[None, :]
        x = x_acc * tl.sigmoid(x_acc)

        dt_raw = tl.load(row_base + dt_row.to(tl.int64) * DPROJ_TOT + DT_OFF + h,
                         mask=lmask, other=0.0)
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


# ---------------------------------------------------------------------------
# Kernel 2p ("v5p"): kernel 2 with PACKED row addressing.  The batch is one
# concatenated stream (T_aug rows, no pad slots anywhere); program t reads its
# segment rows [cu[t], cu[t+1]) directly.  cuBLAS projections and the norm
# run on the same packed rows, so the ~31% pad work of the padded-static
# layout disappears from every stage.  REVERSE flips within the segment.
# ---------------------------------------------------------------------------


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
    key=["H", "P", "N", "BL", "REVERSE"],
)
@triton.jit
def _ssd_short_fwd_kernel2p(
    zxbcdt_ptr,  # (T_aug, DPROJ) packed rows
    convw_ptr, convb_ptr, dtb_ptr, alog_ptr, d_ptr,
    cu_ptr,      # (B+1,) int64/int32 augmented cumulative segment ends
    out_ptr,     # (T_aug, HP)
    DPROJ_TOT,
    H: tl.constexpr,
    P: tl.constexpr,
    N: tl.constexpr,
    DCONV: tl.constexpr,
    BL: tl.constexpr,
    REVERSE: tl.constexpr,
    HPP: tl.constexpr,  # heads per program; 0 means all H in one program
):
    pid_t = tl.program_id(0)
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

    # NOTE (measured, night 2): a shift-matrix tl.dot conv (load each tile
    # once, convolve in registers) was tried here and REGRESSED 4× — the
    # extra per-dot SMEM staging recreates kernel3's pathology in miniature.
    # Strided masked loads win; keep them.

    # ---- shared across heads: conv'd B, C and G, computed once -------------
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
                                       mask=rmask[:, None], other=0.0)
        c_acc += wc[None, :] * tl.load(zxbcdt_ptr + roff + (XBC_OFF + HP + N + offs_n)[None, :],
                                       mask=rmask[:, None], other=0.0)
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
                mask=rmask[:, None], other=0.0)
        x_acc += tl.load(convb_ptr + h * P + offs_p)[None, :]
        x = x_acc * tl.sigmoid(x_acc)

        dt_raw = tl.load(zxbcdt_ptr + (base + dt_row.to(tl.int64)) * DPROJ_TOT + DT_OFF + h,
                         mask=lmask, other=0.0)
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
    """Packed-stream single-chunk SSD scan (pre-norm Y rows, (T_aug, H*P))."""
    T, dproj = zxbcdt_rows.shape
    H, P, N = nheads, headdim, d_state
    w = conv_weight.reshape(conv_weight.shape[0], -1).contiguous()
    zx = zxbcdt_rows.contiguous()
    cu = cu_seqlens_aug.to(device=zx.device, dtype=torch.int64).contiguous()
    B = cu.shape[0] - 1
    out = torch.empty(T, H * P, device=zx.device, dtype=zx.dtype)
    grid = lambda META: (B, 1 if META["HPP"] == 0 else H // META["HPP"])  # noqa: E731
    _ssd_short_fwd_kernel2p[grid](
        zx, w, conv_bias.contiguous(), dt_bias.contiguous(),
        A_log.contiguous(), D.contiguous(), cu, out,
        dproj, H=H, P=P, N=N, DCONV=w.shape[-1], BL=32, REVERSE=reverse,
    )
    return out


@ssd_short_fwd_packed.register_fake
def _(zxbcdt_rows, conv_weight, conv_bias, dt_bias, A_log, D, cu_seqlens_aug,
      nheads, headdim, d_state, reverse=False):
    T, _ = zxbcdt_rows.shape
    return zxbcdt_rows.new_empty(T, nheads * headdim)


# ---------------------------------------------------------------------------
# Fused gated RMSNorm: out = RMSNorm(y * silu(z)) * w   (norm_before_gate=False)
# One row per program, single pass — Inductor's generated reduction for this
# pattern measured ~0.58 TB/s; this trivial kernel is purely load-bound.
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
    y_ptr,      # (R, DSSM) fp32 rows
    z_ptr,      # (R, Z_STRIDE) fp32 — z slice starts the row
    w_ptr,      # (DSSM,)
    out_ptr,    # (R, DSSM)
    Z_STRIDE,   # row stride of the z tensor (d_in_proj when z is a view)
    EPS,
    DSSM: tl.constexpr,
    DBLK: tl.constexpr,  # next power of two >= DSSM (tl.arange needs pow2)
):
    row = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, DBLK)
    cmask = offs < DSSM
    y = tl.load(y_ptr + row * DSSM + offs, mask=cmask, other=0.0)
    z = tl.load(z_ptr + row * Z_STRIDE + offs, mask=cmask, other=0.0)
    g = y * (z * tl.sigmoid(z))
    ms = tl.sum(g * g, 0) / DSSM
    rstd = 1.0 / tl.sqrt(ms + EPS)
    w = tl.load(w_ptr + offs, mask=cmask, other=0.0)
    tl.store(out_ptr + row * DSSM + offs, g * rstd * w, mask=cmask)


@torch.library.custom_op("track_regression::gated_rmsnorm", mutates_args=())
def gated_rmsnorm(y: torch.Tensor, z_rows: torch.Tensor, weight: torch.Tensor,
                  eps: float) -> torch.Tensor:
    """out = RMSNorm(y * silu(z)) * weight, rowwise over the last dim.

    ``z_rows`` is a CONTIGUOUS (R, K) tensor whose first DSSM columns are z
    — pass ``zxbcdt.view(R, d_in_proj)`` directly (z is its leading slice),
    so no slice copy is ever materialised.
    """
    d = y.shape[-1]
    assert y.is_contiguous() and z_rows.is_contiguous()
    y2 = y.view(-1, d)
    assert z_rows.shape[0] == y2.shape[0]
    out = torch.empty_like(y2)
    _gated_rmsnorm_kernel[(y2.shape[0],)](
        y2, z_rows, weight.contiguous(), out,
        z_rows.shape[1], eps, DSSM=d, DBLK=triton.next_power_of_2(d),
    )
    return out.view_as(y)


@gated_rmsnorm.register_fake
def _(y, z, weight, eps):
    return torch.empty_like(y)


# ---------------------------------------------------------------------------
# Training support: backward passes for the packed scan + gated norm ops.
#
# Strategy (Mamba's own recompute philosophy, adapted): the backward
# re-evaluates the identical algebra in differentiable pure torch on the
# saved inputs and routes gradients with torch.autograd.grad. Exactly the
# same math as the kernel (oracle chain), so gradients are exact up to
# fp reordering; forward speed keeps the Triton kernel, backward costs one
# torch recompute (v3-class speed) — measured in the training benches.
# ---------------------------------------------------------------------------


def _packed_scan_torch_ref(zx_rows, conv_w, conv_b, dt_bias, A_log, Dp,
                           cu, H, P, N, reverse):
    """Differentiable reference of _ssd_short_fwd_kernel2p (packed rows)."""
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


def _ssd_packed_setup_ctx(ctx, inputs, output):
    (zx, conv_w, conv_b, dt_bias, A_log, Dp, cu, H, P, N, reverse) = inputs
    ctx.save_for_backward(zx, conv_w, conv_b, dt_bias, A_log, Dp, cu)
    ctx.dims = (H, P, N, reverse)


def _ssd_packed_backward(ctx, grad_out):
    zx, conv_w, conv_b, dt_bias, A_log, Dp, cu = ctx.saved_tensors
    H, P, N, reverse = ctx.dims
    with torch.enable_grad():
        leaves = [t.detach().requires_grad_(t.is_floating_point())
                  for t in (zx, conv_w, conv_b, dt_bias, A_log, Dp)]
        y = _packed_scan_torch_ref(*leaves, cu, H, P, N, reverse)
        grads = torch.autograd.grad(y, leaves, grad_out)
    return (*grads, None, None, None, None, None)


ssd_short_fwd_packed.register_autograd(
    _ssd_packed_backward, setup_context=_ssd_packed_setup_ctx
)


def _gated_norm_setup_ctx(ctx, inputs, output):
    y, z_rows, weight, eps = inputs
    ctx.save_for_backward(y, z_rows, weight)
    ctx.eps = eps


def _gated_norm_backward(ctx, grad_out):
    import torch.nn.functional as F

    y, z_rows, weight, eps = *ctx.saved_tensors, ctx.eps
    d = y.shape[-1]
    with torch.enable_grad():
        yl = y.detach().requires_grad_(True)
        zl = z_rows.detach().requires_grad_(True)
        wl = weight.detach().requires_grad_(True)
        z = zl.view(-1, zl.shape[-1])[:, :d].view_as(yl)
        g = yl.float() * F.silu(z.float())
        rstd = torch.rsqrt(g.square().mean(dim=-1, keepdim=True) + eps)
        out = (g * rstd * wl.float()).to(yl.dtype)
        gy, gz, gw = torch.autograd.grad(out, (yl, zl, wl), grad_out)
    return gy, gz, gw, None


gated_rmsnorm.register_autograd(
    _gated_norm_backward, setup_context=_gated_norm_setup_ctx
)
