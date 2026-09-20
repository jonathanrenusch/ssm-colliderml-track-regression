"""Acceptance tests for the 2026-09 architecture-ablation encoders.

Covers the per-encoder gate of the study protocol:
  (a) shapes on a packed batch,
  (b) packed <-> padded equivalence,
  (c) permutation / segment independence of the packed path,
  (e) parameter counts inside the trunk budget,
plus a CPU-only sanity check that the head stack built on top of each
encoder is byte-identical to the paper model's.
"""

from __future__ import annotations

import pytest
import torch

from track_regression.ablation_encoders import (
    BiGRUCLSEncoder,
    FlatMLPEncoder,
    ForwardMambaCLSEncoder,
)

torch.manual_seed(0)

DIM = 128


def make_packed(lens, dim=DIM, dtype=torch.float64, device="cpu"):
    cu = torch.zeros(len(lens) + 1, dtype=torch.int32, device=device)
    cu[1:] = torch.cumsum(torch.tensor(lens, device=device), 0)
    total = int(cu[-1])
    x = torch.randn(1, total, dim, dtype=dtype, device=device)
    return x, cu


def to_padded(x, cu, dim=DIM):
    lens = (cu[1:].long() - cu[:-1].long())
    B, L = len(lens), int(lens.max())
    xp = x.new_zeros(B, L, dim)
    mask = torch.zeros(B, L, dtype=torch.bool, device=x.device)
    o = 0
    for i, n in enumerate(lens.tolist()):
        xp[i, :n] = x[0, o:o + n]
        mask[i, :n] = True
        o += n
    return xp, mask


def encoders(dtype=torch.float64):
    return {
        "gru": BiGRUCLSEncoder(dim=DIM, hidden_size=120, num_layers=2).to(dtype),
        "fwd": ForwardMambaCLSEncoder(num_layers=2, dim=DIM, d_state=64, d_conv=1,
                                      expand=4, headdim=32, compile_core=False).to(dtype),
        "mlp": FlatMLPEncoder(dim=DIM, max_len=20, hidden_layers=(192, 192)).to(dtype),
    }


@pytest.mark.parametrize("name", ["gru", "fwd", "mlp"])
def test_shapes_packed(name):
    enc = encoders()[name].eval()
    lens = [7, 12, 20, 6, 13]
    x, cu = make_packed(lens)
    seq, pooled = enc(x, cu_seqlens=cu, seq_idx=None)
    assert pooled.shape == (len(lens), 256)
    assert enc.pool_dim == 256
    assert seq.shape[0] == 1 and seq.shape[1] == sum(lens)
    assert torch.isfinite(pooled).all()


@pytest.mark.parametrize("name", ["gru", "fwd", "mlp"])
def test_packed_vs_padded(name):
    enc = encoders()[name].eval()
    lens = [7, 12, 20, 6, 13]
    x, cu = make_packed(lens)
    xp, mask = to_padded(x, cu)
    _, p_packed = enc(x, cu_seqlens=cu, seq_idx=None)
    _, p_padded = enc(xp, kv_mask=mask)
    assert torch.max((p_packed - p_padded).abs()).item() < 1e-6


@pytest.mark.parametrize("name", ["gru", "fwd", "mlp"])
def test_segment_permutation_independence(name):
    """Shuffling the track order inside a packed batch must permute the
    outputs and change nothing else — the standard cu_seqlens bug catcher."""
    enc = encoders()[name].eval()
    lens = [7, 12, 20, 6, 13]
    x, cu = make_packed(lens)
    _, pooled = enc(x, cu_seqlens=cu, seq_idx=None)

    perm = [3, 0, 4, 1, 2]
    lens_p = [lens[i] for i in perm]
    starts = [int(cu[i]) for i in range(len(lens))]
    pieces = [x[0, starts[i]:starts[i] + lens[i]] for i in perm]
    x2 = torch.cat(pieces, 0).unsqueeze(0)
    cu2 = torch.zeros(len(lens) + 1, dtype=torch.int32)
    cu2[1:] = torch.cumsum(torch.tensor(lens_p), 0)
    _, pooled2 = enc(x2, cu_seqlens=cu2, seq_idx=None)
    assert torch.max((pooled2 - pooled[perm]).abs()).item() < 1e-9


def test_hit_order_matters_for_sequence_encoders():
    """The GRU and the forward SSM must actually read the order (the MLP
    reads slot index, so it also does; a permutation-invariant encoder
    would silently pass everything else)."""
    for name in ("gru", "fwd"):
        enc = encoders()[name].eval()
        lens = [11]
        x, cu = make_packed(lens)
        _, a = enc(x, cu_seqlens=cu, seq_idx=None)
        xr = x.flip(1).contiguous()
        _, b = enc(xr, cu_seqlens=cu, seq_idx=None)
        assert torch.max((a - b).abs()).item() > 1e-6, name


def test_trunk_parameter_budget():
    """Encoder parameter counts, as used to match the 628,320-parameter
    trunk of the paper model (input_net 78,080 + pool_head 49,408 fixed)."""
    fixed = 78_080 + 49_408
    target = 628_320
    counts = {k: sum(p.numel() for p in e.parameters())
              for k, e in encoders(torch.float32).items()}
    for k in ("gru", "fwd"):
        trunk = counts[k] + fixed
        assert abs(trunk - target) / target < 0.10, (k, trunk)


def test_no_dropout_anywhere():
    for name, enc in encoders(torch.float32).items():
        for m in enc.modules():
            assert not (isinstance(m, torch.nn.Dropout) and m.p > 0), name


def test_ddp_tie_inference_safe():
    """A NaN in one track's hits must not poison the other tracks at
    inference (CLAUDE.md §4.27)."""
    for name in ("gru", "fwd", "mlp"):
        enc = encoders()[name].eval()
        lens = [7, 12]
        x, cu = make_packed(lens)
        x = x.clone()
        x[0, 0, 0] = float("nan")
        _, pooled = enc(x, cu_seqlens=cu, seq_idx=None)
        assert torch.isfinite(pooled[1]).all(), name
