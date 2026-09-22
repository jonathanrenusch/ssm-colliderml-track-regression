"""One-program-per-track attention for short packed sequences (Triton).

Why a kernel of our own.  A charged-particle track is 6-20 hits plus two class
tokens, and a batch is ~10^5 of them.  Every stock attention primitive is built
for the opposite shape (few, long sequences): dense SDPA needs the tracks
padded to the longest one and a (B, L, L) boolean mask materialised;
FlashAttention's tiling never amortises over 22 tokens and its varlen entry
point does not even launch at B = 131k (grid limit, measured
``scripts/attn_strategy_study.py``).  Here one program owns one track: it
loads the track's q/k/v rows straight from the packed ``(T, 3D)`` projection
output using ``cu_seqlens``, applies the per-vector RMSNorm the model puts on
q, k and v (which needs the whole 128-wide row and so cannot be split per
head), runs the four 22x22 attention problems in registers, and writes the
``(T, D)`` output in place.  No padding, no mask tensor, no scatter/gather
around the attention step, no shared memory beyond what ``tl.dot`` needs
internally.

Conventions follow ``mingru_short_triton`` / ``ssd_short_triton``:
``@torch.library.custom_op`` (opaque to Inductor), packed layout with
``cu_seqlens``, output rows written exactly once, device pinned at launch,
strict IEEE fp32 dots for fp32 inputs (the Mamba campaign measured TF32 inside
tiny in-kernel dots as slower); fp16/bf16 inputs use the tensor cores with fp32
accumulation and fp32 softmax.
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
        triton.Config({}, num_warps=4, num_stages=2),
    ],
    key=["D", "NH", "BL", "DOT_F16", "NORM"],
)
@triton.jit
def _attn_track_kernel(
    qkv_ptr,     # (T, 3D) packed, [q | k | v], fp32 / fp16 / bf16
    cu_ptr,      # (B+1,) int32 cumulative segment boundaries (CLS rows included)
    wq_ptr, wk_ptr, wv_ptr,   # (D,) RMSNorm weights for q, k, v (read iff NORM)
    out_ptr,     # (T, D) same dtype as qkv
    eps,         # RMSNorm epsilon (fp32)
    sm_scale,    # 1/sqrt(DH)
    D: tl.constexpr,
    NH: tl.constexpr,
    DH: tl.constexpr,
    BL: tl.constexpr,        # static bound on the track length (pow2 >= 22)
    DOT_F16: tl.constexpr,   # 1: fp16 tensor-core dots; 0: IEEE fp32 dots
    NORM: tl.constexpr,      # 1: apply the q/k/v RMSNorm in-kernel
):
    pid = tl.program_id(0).to(tl.int64)
    start = tl.load(cu_ptr + pid).to(tl.int64)
    end = tl.load(cu_ptr + pid + 1).to(tl.int64)
    L = end - start

    offs_l = tl.arange(0, BL)
    lmask = offs_l < L
    rows = start + offs_l                        # (BL,) int64
    row_off = rows[:, None] * (3 * D)            # (BL, 1)
    offs_d = tl.arange(0, DH)

    # ---- pass 1: per-row sum of squares over the FULL D-wide q, k, v -------
    # (the model's qkv_norm is RMSNorm(D), i.e. across all heads, so the row
    # statistic has to be gathered before any head can be normalised)
    rq = tl.zeros((BL,), dtype=tl.float32)
    rk = tl.zeros((BL,), dtype=tl.float32)
    rv = tl.zeros((BL,), dtype=tl.float32)
    if NORM:
        for h in tl.static_range(NH):
            cols = h * DH + offs_d
            q = tl.load(qkv_ptr + row_off + cols[None, :],
                        mask=lmask[:, None], other=0.0).to(tl.float32)
            k = tl.load(qkv_ptr + row_off + D + cols[None, :],
                        mask=lmask[:, None], other=0.0).to(tl.float32)
            v = tl.load(qkv_ptr + row_off + 2 * D + cols[None, :],
                        mask=lmask[:, None], other=0.0).to(tl.float32)
            rq += tl.sum(q * q, 1)
            rk += tl.sum(k * k, 1)
            rv += tl.sum(v * v, 1)
        rq = 1.0 / tl.sqrt(rq / D + eps)
        rk = 1.0 / tl.sqrt(rk / D + eps)
        rv = 1.0 / tl.sqrt(rv / D + eps)

    # ---- pass 2: per head, normalise, attend, store -----------------------
    for h in tl.static_range(NH):
        cols = h * DH + offs_d
        q = tl.load(qkv_ptr + row_off + cols[None, :],
                    mask=lmask[:, None], other=0.0).to(tl.float32)
        k = tl.load(qkv_ptr + row_off + D + cols[None, :],
                    mask=lmask[:, None], other=0.0).to(tl.float32)
        v = tl.load(qkv_ptr + row_off + 2 * D + cols[None, :],
                    mask=lmask[:, None], other=0.0).to(tl.float32)
        if NORM:
            q = q * rq[:, None] * tl.load(wq_ptr + cols).to(tl.float32)[None, :]
            k = k * rk[:, None] * tl.load(wk_ptr + cols).to(tl.float32)[None, :]
            v = v * rv[:, None] * tl.load(wv_ptr + cols).to(tl.float32)[None, :]
        if DOT_F16:
            s = tl.dot(q.to(tl.float16), tl.trans(k.to(tl.float16)))
        else:
            s = tl.dot(q, tl.trans(k), input_precision="ieee")
        s = s * sm_scale
        s = tl.where(lmask[None, :], s, float("-inf"))     # pad keys out
        m = tl.max(s, 1)
        m = tl.maximum(m, -1e30)                           # pad query rows: no NaN
        p = tl.exp(s - m[:, None])
        l = tl.sum(p, 1)
        p = p / tl.maximum(l, 1e-30)[:, None]
        if DOT_F16:
            o = tl.dot(p.to(tl.float16), v.to(tl.float16))
        else:
            o = tl.dot(p, v, input_precision="ieee")
        tl.store(out_ptr + rows[:, None] * D + cols[None, :],
                 o.to(out_ptr.dtype.element_ty), mask=lmask[:, None])


@torch.library.custom_op("track_regression::attn_packed_tracks", mutates_args=())
def attn_packed_tracks(qkv: torch.Tensor, cu_seqlens: torch.Tensor,
                       wq: torch.Tensor, wk: torch.Tensor, wv: torch.Tensor,
                       num_heads: int, eps: float, norm: bool,
                       max_len: int) -> torch.Tensor:
    """Block-diagonal (per-track) softmax attention on a packed stream.

    ``qkv``: (T, 3D) contiguous, ``[q | k | v]`` in the layout
    ``F._in_projection_packed`` produces; ``cu_seqlens``: (B+1,) segment
    boundaries of the stream (class tokens already included in the segments).
    With ``norm`` the model's ``RMSNorm(D)`` on q, k and v is applied
    in-kernel with weights ``wq``/``wk``/``wv`` and ``eps``.  Returns (T, D) in
    ``qkv``'s dtype -- exactly what ``recombine_heads(SDPA(...))`` returns on
    the padded path, gathered back to the packed rows.
    """
    assert qkv.is_cuda and qkv.is_contiguous() and qkv.dim() == 2
    T, three_d = qkv.shape
    D = three_d // 3
    assert 3 * D == three_d, qkv.shape
    NH = int(num_heads)
    DH = D // NH
    assert DH * NH == D and DH >= 16 and (DH & (DH - 1)) == 0, (D, NH)
    BL = 16
    while BL < int(max_len):
        BL *= 2
    B = cu_seqlens.numel() - 1
    out = torch.empty(T, D, device=qkv.device, dtype=qkv.dtype)
    dot_f16 = 1 if qkv.dtype in (torch.float16, torch.bfloat16) else 0
    if not norm:      # dummy pointers, never read
        wq = wk = wv = out
    with torch.cuda.device(qkv.device):
        _attn_track_kernel[(B,)](
            qkv, cu_seqlens.to(torch.int32), wq, wk, wv, out,
            float(eps), float(DH) ** -0.5,
            D=D, NH=NH, DH=DH, BL=BL, DOT_F16=dot_f16, NORM=1 if norm else 0,
        )
    return out


@attn_packed_tracks.register_fake
def _(qkv, cu_seqlens, wq, wk, wv, num_heads, eps, norm, max_len):
    return qkv.new_empty(qkv.shape[0], qkv.shape[1] // 3)


def attn_packed_tracks_reference(qkv: torch.Tensor, cu_seqlens: torch.Tensor,
                                 wq, wk, wv, num_heads: int, eps: float,
                                 norm: bool) -> torch.Tensor:
    """Pure-torch oracle for the kernel (any dtype, incl. float64)."""
    T, three_d = qkv.shape
    D = three_d // 3
    q, k, v = qkv.split(D, dim=-1)
    if norm:
        q = torch.nn.functional.rms_norm(q, (D,), wq.to(q.dtype), eps)
        k = torch.nn.functional.rms_norm(k, (D,), wk.to(k.dtype), eps)
        v = torch.nn.functional.rms_norm(v, (D,), wv.to(v.dtype), eps)
    cu = cu_seqlens.tolist()
    out = torch.empty(T, D, dtype=qkv.dtype, device=qkv.device)
    DH = D // num_heads
    for s in range(len(cu) - 1):
        a, b = cu[s], cu[s + 1]
        qs = q[a:b].view(-1, num_heads, DH).transpose(0, 1)   # (H, L, DH)
        ks = k[a:b].view(-1, num_heads, DH).transpose(0, 1)
        vs = v[a:b].view(-1, num_heads, DH).transpose(0, 1)
        sc = (qs @ ks.transpose(-1, -2)) * DH ** -0.5
        o = torch.softmax(sc, -1) @ vs                        # (H, L, DH)
        out[a:b] = o.transpose(0, 1).reshape(b - a, D)
    return out


# ---------------------------------------------------------------------------
# Fused residual add + RMSNorm for the packed encoder glue.
#
# Inductor's generated reduction for ``rmsnorm(x + g * o)`` over 128-wide rows
# runs at a fraction of HBM bandwidth (the same pathology the SSD campaign
# measured for its gated RMSNorm, ~0.58 TB/s).  One program per block of rows,
# single pass, writes both the new residual stream ``y`` and the normalised
# GEMM input ``h`` -- purely load-bound.
# ---------------------------------------------------------------------------


@triton.autotune(
    configs=[
        triton.Config({"BR": 4}, num_warps=2),
        triton.Config({"BR": 8}, num_warps=4),
        triton.Config({"BR": 16}, num_warps=4),
        triton.Config({"BR": 16}, num_warps=8),
        triton.Config({"BR": 32}, num_warps=8),
    ],
    key=["D", "HAS_RES", "WANT_H"],
)
@triton.jit
def _add_rmsnorm_kernel(
    x_ptr, o_ptr, g_ptr, w_ptr, y_ptr, h_ptr, T, eps,
    D: tl.constexpr, BR: tl.constexpr, HAS_RES: tl.constexpr, WANT_H: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    rows = pid * BR + tl.arange(0, BR)
    rmask = rows < T
    cols = tl.arange(0, D)
    off = rows[:, None] * D + cols[None, :]
    m = rmask[:, None]
    x = tl.load(x_ptr + off, mask=m, other=0.0).to(tl.float32)
    if HAS_RES:
        o = tl.load(o_ptr + off, mask=m, other=0.0).to(tl.float32)
        g = tl.load(g_ptr + cols).to(tl.float32)
        x = x + g[None, :] * o
        tl.store(y_ptr + off, x.to(y_ptr.dtype.element_ty), mask=m)
    if WANT_H:
        ms = tl.sum(x * x, 1) / D
        r = 1.0 / tl.sqrt(ms + eps)
        w = tl.load(w_ptr + cols).to(tl.float32)
        h = x * r[:, None] * w[None, :]
        tl.store(h_ptr + off, h.to(h_ptr.dtype.element_ty), mask=m)


@torch.library.custom_op("track_regression::add_rmsnorm_packed", mutates_args=())
def add_rmsnorm_packed(x: torch.Tensor, o: torch.Tensor, gamma: torch.Tensor,
                       w: torch.Tensor, eps: float, has_res: bool, want_h: bool,
                       h_dtype: torch.dtype) -> list[torch.Tensor]:
    """``y = x + gamma * o`` (fp32 math, stored in ``x.dtype``) and
    ``h = RMSNorm(y) * w`` stored in ``h_dtype``.  ``has_res=False`` skips the
    add (``y`` is returned empty, the caller keeps ``x``); ``want_h=False``
    skips the norm (``h`` is empty)."""
    assert x.is_cuda and x.is_contiguous() and x.dim() == 2
    T, D = x.shape
    assert (D & (D - 1)) == 0, D
    h = torch.empty(T if want_h else 0, D, device=x.device, dtype=h_dtype)
    if has_res:
        assert o.shape == x.shape and o.is_contiguous()
        y = torch.empty_like(x)
    else:
        # a custom op may not return one of its inputs, so ``y`` is an empty
        # placeholder here and the caller keeps using ``x``
        y, o, gamma = x.new_empty(0, D), x, w
    grid = lambda meta: (triton.cdiv(T, meta["BR"]),)  # noqa: E731
    with torch.cuda.device(x.device):
        _add_rmsnorm_kernel[grid](x, o, gamma, w, y, h, T, float(eps),
                                  D=D, HAS_RES=1 if has_res else 0,
                                  WANT_H=1 if want_h else 0)
    return [y, h]


@add_rmsnorm_packed.register_fake
def _(x, o, gamma, w, eps, has_res, want_h, h_dtype):
    return [x.new_empty(x.shape) if has_res else x.new_empty((0, x.shape[1])),
            x.new_empty((x.shape[0] if want_h else 0, x.shape[1]), dtype=h_dtype)]


# ---------------------------------------------------------------------------
# fp16 GEMM with the encoder's epilogues fused in (Triton, tensor cores).
#
#   EPI = 0   : out = acc + bias                          (plain, fp16)
#   EPI = 1   : out = silu(acc + bias)                    (feed-forward up-proj)
#   EPI = 2   : y = x_res + gamma * (acc + bias)          (LayerScale residual)
#               h = RMSNorm(y) * w                        (next pre-norm)
#               -- requires BLOCK_N == N: one program owns whole rows, so the
#               row statistic of the residual stream is available in-tile and
#               the norm costs no second pass over the activations.
#
# Why: on the packed stream every layer is dominated by memory traffic, not
# FLOPs.  With cuBLAS the bias, the SiLU, the residual add and the RMSNorm each
# cost a full extra read+write of the activations; fused into the GEMM epilogue
# they cost nothing.  fp16 only -- strict fp32 keeps the cuBLAS path.
# ---------------------------------------------------------------------------


def _gemm_configs():
    cfgs = []
    for bm, bk, w, st in [(64, 32, 4, 3), (128, 32, 4, 3), (128, 64, 4, 3), (128, 64, 8, 3),
                          (64, 64, 4, 4), (128, 32, 8, 4), (256, 64, 8, 3), (64, 128, 4, 3)]:
        cfgs.append(triton.Config({"BM": bm, "BK": bk}, num_warps=w, num_stages=st))
    return cfgs


@triton.autotune(configs=_gemm_configs(), key=["N", "K", "EPI", "WANT_H"])
@triton.jit
def _gemm_epi_kernel(
    a_ptr, w_ptr, b_ptr, x_ptr, g_ptr, nw_ptr, out_ptr, y_ptr, h_ptr,
    M, eps,
    N: tl.constexpr, K: tl.constexpr, BN: tl.constexpr,
    BM: tl.constexpr, BK: tl.constexpr,
    EPI: tl.constexpr, WANT_H: tl.constexpr,
):
    pid_m = tl.program_id(0).to(tl.int64)
    pid_n = tl.program_id(1)
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    mmask = rm < M
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in tl.range(0, K, BK):
        rk = k0 + tl.arange(0, BK)
        a = tl.load(a_ptr + rm[:, None] * K + rk[None, :], mask=mmask[:, None], other=0.0)
        # W is (N, K) row-major (nn.Linear layout); load its (BK, BN) transpose tile
        wt = tl.load(w_ptr + rn[None, :] * K + rk[:, None])
        acc = tl.dot(a, wt, acc)
    acc = acc + tl.load(b_ptr + rn).to(tl.float32)[None, :]
    off = rm[:, None] * N + rn[None, :]
    if EPI == 0:
        tl.store(out_ptr + off, acc.to(out_ptr.dtype.element_ty), mask=mmask[:, None])
    elif EPI == 1:
        acc = acc * tl.sigmoid(acc)
        tl.store(out_ptr + off, acc.to(out_ptr.dtype.element_ty), mask=mmask[:, None])
    else:
        g = tl.load(g_ptr + rn).to(tl.float32)
        x = tl.load(x_ptr + off, mask=mmask[:, None], other=0.0).to(tl.float32)
        y = x + g[None, :] * acc
        tl.store(y_ptr + off, y.to(y_ptr.dtype.element_ty), mask=mmask[:, None])
        if WANT_H:
            ms = tl.sum(y * y, 1) / N
            r = 1.0 / tl.sqrt(ms + eps)
            nw = tl.load(nw_ptr + rn).to(tl.float32)
            h = y * r[:, None] * nw[None, :]
            tl.store(h_ptr + off, h.to(h_ptr.dtype.element_ty), mask=mmask[:, None])


@torch.library.custom_op("track_regression::gemm_epilogue_fp16", mutates_args=())
def gemm_epilogue_fp16(a: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor,
                       x_res: torch.Tensor, gamma: torch.Tensor, norm_w: torch.Tensor,
                       eps: float, epi: int, want_h: bool) -> list[torch.Tensor]:
    """``a`` (M, K) fp16, ``weight`` (N, K) fp16, ``bias`` (N,) fp16/fp32.
    epi 0/1: returns ``[out (M, N) fp16, empty, empty]``.
    epi 2: returns ``[empty, y (M, N) in x_res.dtype, h (M, N) fp16 or empty]``
    with ``y = x_res + gamma * (a @ W^T + b)`` and ``h = RMSNorm(y) * norm_w``."""
    assert a.is_cuda and a.dtype == torch.float16 and a.is_contiguous()
    assert weight.dtype == torch.float16 and weight.is_contiguous()
    M, K = a.shape
    N = weight.shape[0]
    assert weight.shape[1] == K and K % 128 == 0, (N, K)   # every BK in the configs divides K
    if epi in (0, 1):
        assert N % 128 == 0, N                       # column blocks of 128
    else:
        assert (N & (N - 1)) == 0, N                 # one program owns a full row
    dev = a.device
    empty = a.new_empty(0, N)
    if epi in (0, 1):
        out = torch.empty(M, N, device=dev, dtype=torch.float16)
        y, h = empty, empty
        x_res, gamma, norm_w = out, bias, bias
        BN = min(N, 128)
    else:
        assert x_res.shape == (M, N) and x_res.is_contiguous()
        out = empty
        y = torch.empty(M, N, device=dev, dtype=x_res.dtype)
        h = torch.empty(M if want_h else 0, N, device=dev, dtype=torch.float16)
        BN = N            # one program owns full rows -> row norm in-tile
    grid = lambda meta: (triton.cdiv(M, meta["BM"]), triton.cdiv(N, BN))  # noqa: E731
    with torch.cuda.device(dev):
        _gemm_epi_kernel[grid](
            a, weight, bias, x_res, gamma, norm_w, out, y, h, M, float(eps),
            N=N, K=K, BN=BN, EPI=int(epi), WANT_H=1 if want_h else 0,
        )
    return [out, y, h]


@gemm_epilogue_fp16.register_fake
def _(a, weight, bias, x_res, gamma, norm_w, eps, epi, want_h):
    M = a.shape[0]; N = weight.shape[0]
    if epi in (0, 1):
        return [a.new_empty(M, N), a.new_empty(0, N), a.new_empty(0, N)]
    return [a.new_empty(0, N), x_res.new_empty(M, N), a.new_empty(M if want_h else 0, N)]
