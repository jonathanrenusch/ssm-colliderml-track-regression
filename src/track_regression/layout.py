"""Packed batch layout and the inference-kernel switch.

Batches are *packed*: the hits of all tracks in a batch form one
``(1, total_hits, D)`` stream with ``cu_seqlens`` marking the track
boundaries.  :func:`packed_to_padded` converts to the padded ``(B, L_max, D)``
layout used by the training-path kernels.

At inference every encoder runs its fused short-sequence path by default: the
packed Triton scan for the minGRU, the fused packed SSD kernels for Mamba-2,
and the packed attention / fused-epilogue kernels for the Transformer.
``TRK_REFERENCE_KERNELS=1`` makes inference use the exact code path each
encoder trains with instead (padded layout, compiled pure-PyTorch recurrence
or attention) -- the "default kernel" baseline of the throughput comparison.
Training always uses the training path.
"""

from __future__ import annotations

import os

import torch
from torch import Tensor


def fused_kernels_enabled() -> bool:
    return os.environ.get("TRK_REFERENCE_KERNELS", "0") != "1"


def packed_to_padded(x: Tensor, cu_seqlens: Tensor):
    """``(1, T, D)`` packed -> ``(B, L_max, D)`` zero-padded (pads trailing).

    Returns ``(x_pad, row, pos, lens)``; the inverse gather is ``x_pad[row, pos]``.
    """
    cu = cu_seqlens.to(torch.long)
    B = cu.numel() - 1
    total_L, D = x.shape[1], x.shape[2]
    lens = cu[1:] - cu[:-1]
    max_len = int(lens.max().item())
    arange = torch.arange(total_L, device=x.device, dtype=torch.long)
    row = torch.bucketize(arange, cu[1:], right=True)
    pos = arange - cu[row]
    x_pad = x.new_zeros(B, max_len, D)
    x_pad[row, pos] = x[0]
    return x_pad, row, pos, lens
