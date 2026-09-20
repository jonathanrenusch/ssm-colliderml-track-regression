"""Fused short-sequence bidirectional minGRU scan (Triton).

Conventions deliberately mirror ``ssd_short_triton`` (the Mamba-2 kernel
campaign, branch ``opt_kernel``):

* padded-static layout ``(B, S, ...)`` with pads strictly trailing, so the
  causal recurrence needs no mask;
* the backward direction is a **valid-prefix flip performed in-kernel**
  (``REVERSE`` constexpr + per-row ``lens``), so the caller never gathers and
  the store un-flips — no flip kernels, no extra traffic;
* the direction's slice is addressed by a **channel offset** into one shared
  in-projection output instead of a Python-level slice (night-1 finding:
  Inductor materialises a copy for every sliced opaque-op input);
* ``@torch.library.custom_op`` so the op stays opaque to Inductor;
* strict IEEE fp32 (no TF32 anywhere; the Mamba campaign measured TF32 inside
  the scan as both slower and noisier).

Portability (deployment target is an RTX 5000 Ada, sm_89 — not only Hopper)
--------------------------------------------------------------------------
The kernel deliberately uses only plain loads/stores, elementwise FMA and
``tl.sigmoid``: **no ``tl.dot``, no TMA, no warp specialisation, no clusters,
no fp8, no ``maxnreg``**, and it allocates **zero shared memory** (state lives
in registers), against Ada's 101,376 B/SM.  Every autotune config is
ahead-of-time compiled for ``sm_89`` and checked for Hopper-only PTX by
``tests/test_mingru.py::test_kernels_compile_for_ada_sm89``, which runs on any
machine — so a Hopper-only construct cannot reach the Ada deployment
unnoticed.  ``num_warps`` stays <= 8 and the grid is 3-D, both within sm_89
limits; Triton re-autotunes per architecture on first launch.

Why this is so much cheaper than the SSD kernel: minGRU's recurrence is
**elementwise**, so there is no L x L decay matrix, no Gram matrix and no
``tl.dot`` at all.  Each program owns one (track, channel-block) and runs the
L steps as fused multiply-adds on a register vector.  The work is
embarrassingly parallel over B x D/BD; the only serial chain is L <= 20 FMAs.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BD": 32}, num_warps=1),
        triton.Config({"BD": 64}, num_warps=2),
        triton.Config({"BD": 64}, num_warps=4),
        triton.Config({"BD": 128}, num_warps=4),
        triton.Config({"BD": 128}, num_warps=8),
        triton.Config({"BD": 256}, num_warps=8),
        triton.Config({"BD": 64}, num_warps=2, num_stages=2),
        triton.Config({"BD": 128}, num_warps=4, num_stages=2),
    ],
    key=["H", "BL"],
)
@triton.jit
def _mingru_bidi_kernel(
    zn_ptr,      # (B, S, 4H) fp32 — [z_fwd | n_fwd | z_bwd | n_bwd]
    lens_ptr,    # (B,) int32 — per-row valid length
    outf_ptr,    # (B, S, H) fp32 — forward hidden states
    outb_ptr,    # (B, S, H) fp32 — backward hidden states (already un-flipped)
    S,           # padded sequence length (runtime)
    H: tl.constexpr,
    BL: tl.constexpr,     # static bound on S (<= 32 here)
    BD: tl.constexpr,     # channel block
):
    pid_b = tl.program_id(0).to(tl.int64)
    pid_d = tl.program_id(1)
    pid_dir = tl.program_id(2)          # 0 = forward, 1 = backward

    offs_d = pid_d * BD + tl.arange(0, BD)
    dmask = offs_d < H

    last = tl.load(lens_ptr + pid_b) - 1          # last valid index

    # Channel offset of this direction's [z | n] pair inside the shared
    # in-projection output.  Forward: [0, H) and [H, 2H).  Backward:
    # [2H, 3H) and [3H, 4H).
    z_off = pid_dir * 2 * H
    n_off = z_off + H

    row_base = zn_ptr + pid_b * (S * 4 * H)
    h = tl.zeros((BD,), dtype=tl.float32)

    for t in tl.range(0, BL):
        valid = t <= last
        # Logical step t reads physical row t (forward) or last - t (backward).
        phys = t
        if pid_dir == 1:
            phys = last - t
        phys = tl.where(valid, phys, 0)
        off = row_base + phys.to(tl.int64) * (4 * H) + offs_d

        zt = tl.load(off + z_off, mask=dmask & valid, other=0.0)
        nt = tl.load(off + n_off, mask=dmask & valid, other=0.0)
        zt = tl.sigmoid(zt)
        # h_t = (1 - z) * h_{t-1} + z * n   — one FMA chain, h stays in regs.
        h_new = h + zt * (nt - h)
        h = tl.where(valid, h_new, h)

        # Store at the PHYSICAL row, so the backward pass is written already
        # un-flipped and the caller never gathers.
        out_ptr = outf_ptr if pid_dir == 0 else outb_ptr
        tl.store(out_ptr + pid_b * (S * H) + phys.to(tl.int64) * H + offs_d,
                 h, mask=dmask & valid)


@torch.library.custom_op("track_regression::mingru_bidi_fused", mutates_args=())
def mingru_bidi_fused(zn: torch.Tensor, lens: torch.Tensor,
                      hidden: int) -> list[torch.Tensor]:
    """``zn``: (B, S, 4H) fp32 contiguous; returns ``[h_fwd, h_bwd]`` (B, S, H)."""
    assert zn.is_cuda and zn.dtype == torch.float32 and zn.is_contiguous()
    B, S, four_h = zn.shape
    H = int(hidden)
    assert four_h == 4 * H, (four_h, H)
    BL = 1
    while BL < S:
        BL *= 2
    outf = torch.zeros(B, S, H, device=zn.device, dtype=zn.dtype)
    outb = torch.zeros(B, S, H, device=zn.device, dtype=zn.dtype)
    grid = lambda meta: (B, triton.cdiv(H, meta["BD"]), 2)  # noqa: E731
    _mingru_bidi_kernel[grid](
        zn, lens.to(torch.int32), outf, outb, S, H=H, BL=BL,
    )
    return [outf, outb]


@mingru_bidi_fused.register_fake
def _(zn, lens, hidden):
    B, S, _ = zn.shape
    return [zn.new_empty(B, S, hidden), zn.new_empty(B, S, hidden)]


# ---------------------------------------------------------------------------
# packed-stream variant — no pad rows anywhere (the analogue of the SSD v5p
# path).  A third of the padded path's projection FLOPs are spent on pad slots
# (mean 13.3 hits vs a 20-slot pad), so the packed stream is ~1.5x cheaper
# before any kernel tuning.
# ---------------------------------------------------------------------------


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
    for t in tl.range(0, MAXL):
        valid = t < Lr
        # forward reads the segment in order, backward from its end; both
        # write at the PHYSICAL row, so the output needs no un-flip.
        phys = start + t
        if pid_dir == 1:
            phys = end - 1 - t
        phys = tl.where(valid, phys, start)

        base = zn_ptr + phys * (4 * H) + offs_d
        # `.to(tl.float32)` is a REGISTER convert on a load that has to happen
        # anyway -- free -- so a half-precision `zn` halves the global traffic of
        # the largest intermediate while the recurrence still accumulates in
        # fp32.  A product of up to MAXL gates in fp16 would lose mantissa and
        # can underflow the 6e-8 subnormal floor, so the state never leaves fp32.
        zt = tl.sigmoid(tl.load(base + z_off, mask=dmask & valid,
                                other=0.0).to(tl.float32))
        nt = tl.load(base + n_off, mask=dmask & valid, other=0.0).to(tl.float32)
        h = tl.where(valid, h + zt * (nt - h), h)

        tl.store(out_ptr + phys * (2 * H) + o_off + offs_d, h,
                 mask=dmask & valid)


@torch.library.custom_op("track_regression::mingru_bidi_packed", mutates_args=())
def mingru_bidi_packed(zn: torch.Tensor, cu_seqlens: torch.Tensor,
                       hidden: int, max_len: int) -> torch.Tensor:
    """``zn``: (T, 4H) contiguous packed; returns (T, 2H) = [h_fwd|h_bwd].

    ``zn`` may be fp32, fp16 or bf16 and the output follows its dtype; the scan
    itself always accumulates in fp32 inside the kernel.  Accepting reduced
    precision here is what makes an fp16 encoder worthwhile: otherwise the
    caller has to insert a full (T, 4H) cast kernel at the boundary, which on
    this model costs more than the faster GEMMs save.

    Every packed row is written exactly once by each direction, so the output
    may be allocated with ``empty`` — no zero-fill pass.
    """
    assert zn.is_cuda and zn.is_contiguous()
    assert zn.dtype in (torch.float32, torch.float16, torch.bfloat16), zn.dtype
    T, four_h = zn.shape
    H = int(hidden)
    assert four_h == 4 * H, (four_h, H)
    B = cu_seqlens.numel() - 1
    out = torch.empty(T, 2 * H, device=zn.device, dtype=zn.dtype)
    grid = lambda meta: (B, triton.cdiv(H, meta["BD"]), 2)  # noqa: E731
    _mingru_bidi_packed_kernel[grid](
        zn, cu_seqlens.to(torch.int32), out, H=H, MAXL=int(max_len),
    )
    return out


@mingru_bidi_packed.register_fake
def _(zn, cu_seqlens, hidden, max_len):
    return zn.new_empty(zn.shape[0], 2 * hidden)
