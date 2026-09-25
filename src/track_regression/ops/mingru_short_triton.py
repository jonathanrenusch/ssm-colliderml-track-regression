"""Fused bidirectional minGRU scan on the packed stream (Triton).

One program owns one (track, channel block, direction).  It walks the
track's segment of the packed ``(T, 4H)`` in-projection ``[z_fwd | n_fwd |
z_bwd | n_bwd]`` -- forward from its first hit, reverse from its last -- and
runs the recurrence ``h = h + sigmoid(z) * (n - h)`` as fused multiply-adds on
a register vector, writing ``[h_fwd | h_bwd]`` at the physical row.  No
padding, no flip gathers, no shared memory: the recurrence is elementwise, so
there is no L x L matrix and no ``tl.dot``.  The loop exits at the track's own
length.  The input may be fp32 / fp16 / bf16 (converted on load); the state
always accumulates in fp32.

Only plain loads/stores, elementwise FMA and ``tl.sigmoid`` are used (no TMA,
warp specialisation or fp8), so the kernel also runs on Ada (sm_89) GPUs.
The op is registered with ``torch.library.custom_op`` so it stays opaque to
``torch.compile``.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({"BD": 64}, num_warps=2),
        triton.Config({"BD": 64}, num_warps=4),
        triton.Config({"BD": 128}, num_warps=4),
        triton.Config({"BD": 128}, num_warps=8),
        triton.Config({"BD": 256}, num_warps=8),
        triton.Config({"BD": 128}, num_warps=4, num_stages=2),
        triton.Config({"BD": 256}, num_warps=4, num_stages=2),
    ],
    key=["H", "MAXL"],
)
@triton.jit
def _mingru_bidi_packed_kernel(
    zn_ptr,      # (T, 4H) fp32/fp16/bf16 packed — [z_fwd | n_fwd | z_bwd | n_bwd]
    cu_ptr,      # (B+1,) int32 cumulative segment boundaries
    out_ptr,     # (T, 2H) same dtype as zn — [h_fwd | h_bwd], written in place
    H: tl.constexpr,
    MAXL: tl.constexpr,   # static upper bound on track length
    BD: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_d = tl.program_id(1)
    pid_dir = tl.program_id(2)

    offs_d = pid_d * BD + tl.arange(0, BD)
    dmask = offs_d < H

    start = tl.load(cu_ptr + pid_b).to(tl.int64)
    end = tl.load(cu_ptr + pid_b + 1).to(tl.int64)
    Lr = end - start

    z_off = pid_dir * 2 * H
    n_off = z_off + H
    o_off = pid_dir * H

    h = tl.zeros((BD,), dtype=tl.float32)
    # a program owns ONE track, so the loop exits at its length with no divergence
    t = 0
    while t < Lr:
        # forward reads the segment in order, backward from its end; both
        # write at the PHYSICAL row, so the output needs no un-flip.
        phys = start + t
        if pid_dir == 1:
            phys = end - 1 - t

        base = zn_ptr + phys * (4 * H) + offs_d
        # convert on load: a half-precision `zn` halves the traffic, while the
        # state (a product of up to MAXL gates) stays in fp32
        zt = tl.sigmoid(tl.load(base + z_off, mask=dmask,
                                other=0.0).to(tl.float32))
        nt = tl.load(base + n_off, mask=dmask, other=0.0).to(tl.float32)
        h = h + zt * (nt - h)

        tl.store(out_ptr + phys * (2 * H) + o_off + offs_d, h, mask=dmask)
        t += 1


@torch.library.custom_op("track_regression::mingru_bidi_packed", mutates_args=())
def mingru_bidi_packed(zn: torch.Tensor, cu_seqlens: torch.Tensor,
                       hidden: int, max_len: int) -> torch.Tensor:
    """``zn``: (T, 4H) contiguous packed; returns (T, 2H) = [h_fwd | h_bwd].

    The output follows the dtype of ``zn``.  Every row is written exactly once
    per direction, so the output is allocated with ``empty``.
    """
    assert zn.is_cuda and zn.is_contiguous()
    assert zn.dtype in (torch.float32, torch.float16, torch.bfloat16), zn.dtype
    T, four_h = zn.shape
    H = int(hidden)
    assert four_h == 4 * H, (four_h, H)
    B = cu_seqlens.numel() - 1
    out = torch.empty(T, 2 * H, device=zn.device, dtype=zn.dtype)
    cu32 = cu_seqlens.to(torch.int32)
    # Triton launches on the current device: pin it to the tensor's
    grid = lambda meta: (B, triton.cdiv(H, meta["BD"]), 2)  # noqa: E731
    with torch.cuda.device(zn.device):
        _mingru_bidi_packed_kernel[grid](
            zn, cu32, out, H=H, MAXL=int(max_len),
        )
    return out


@mingru_bidi_packed.register_fake
def _(zn, cu_seqlens, hidden, max_len):
    return zn.new_empty(zn.shape[0], 2 * hidden)
