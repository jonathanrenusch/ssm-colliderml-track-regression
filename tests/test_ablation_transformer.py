"""Acceptance gate for the transformer baseline of the 2026-09 ablations.

``EncoderWithCLS`` is pre-existing production code; these tests only verify
that the *packed* path it will be driven with in this study is sound
(shapes, packed<->padded equivalence, segment independence) and record the
one behavioural difference from the SSM encoder: its DDP unused-parameter
tie is not gated on ``self.training``.
"""

from __future__ import annotations

import torch

from track_regression.transformer_encoder import EncoderWithCLS

torch.manual_seed(0)
DIM = 128


def build(dtype=torch.float64):
    return EncoderWithCLS(
        dim=DIM, num_cls_tokens=2, num_layers=3, attn_type="torch",
        norm="RMSNorm", value_residual=False, qkv_norm=True, layer_scale=1.0e-5,
        attn_kwargs={"num_heads": 4}, dense_kwargs={"hidden_dim_scale": 3},
        posenc_fourier_scales=[-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
        posenc_fourier_base=2, posenc_time_scale=1.0,
    ).to(dtype).eval()


def make_packed(lens, dtype=torch.float64):
    cu = torch.zeros(len(lens) + 1, dtype=torch.int32)
    cu[1:] = torch.cumsum(torch.tensor(lens), 0)
    total = int(cu[-1])
    x = torch.randn(1, total, DIM, dtype=dtype)
    t = torch.rand(1, total, dtype=dtype) * 10.0
    seq_idx = torch.repeat_interleave(
        torch.arange(len(lens), dtype=torch.int32), torch.tensor(lens)
    ).unsqueeze(0)
    return x, t, seq_idx, cu


def to_padded(x, t, cu):
    lens = (cu[1:].long() - cu[:-1].long())
    B, L = len(lens), int(lens.max())
    xp = x.new_zeros(B, L, DIM)
    tp = t.new_zeros(B, L)
    mask = torch.zeros(B, L, dtype=torch.bool)
    o = 0
    for i, n in enumerate(lens.tolist()):
        xp[i, :n] = x[0, o:o + n]
        tp[i, :n] = t[0, o:o + n]
        mask[i, :n] = True
        o += n
    return xp, tp, mask


def test_shapes_and_pool_dim():
    enc = build()
    lens = [7, 12, 20, 6, 13]
    x, t, si, cu = make_packed(lens)
    seq, pooled = enc(x, x_sort_value=t, seq_idx=si, cu_seqlens=cu)
    assert enc.pool_dim == 256
    assert pooled.shape == (len(lens), 256)
    assert seq.shape == (1, sum(lens), DIM)
    assert torch.isfinite(pooled).all()


def test_packed_vs_padded():
    """Padded mode sorts by ``x_sort_value``; the stores are already in that
    order, so feed a monotone key and the two paths must agree."""
    enc = build()
    lens = [7, 12, 20, 6, 13]
    x, t, si, cu = make_packed(lens)
    xp, tp, mask = to_padded(x, t, cu)
    # Padded mode argsorts the whole padded row and pad slots carry key 0, so
    # a like-for-like comparison of the two layouts is only well defined with
    # the posenc/sort key off.  That is the claim under test: the packed
    # block-diagonal attention reproduces the padded masked attention.
    enc.posenc_proj = None
    _, a = enc(x, x_sort_value=None, seq_idx=si, cu_seqlens=cu)
    _, b = enc(xp, x_sort_value=None, kv_mask=mask)
    assert torch.max((a - b).abs()).item() < 1e-8


def test_segment_permutation_independence():
    enc = build()
    lens = [7, 12, 20, 6, 13]
    x, t, si, cu = make_packed(lens)
    _, pooled = enc(x, x_sort_value=t, seq_idx=si, cu_seqlens=cu)

    perm = [3, 0, 4, 1, 2]
    lens_p = [lens[i] for i in perm]
    starts = [int(cu[i]) for i in range(len(lens))]
    xs = torch.cat([x[0, starts[i]:starts[i] + lens[i]] for i in perm], 0).unsqueeze(0)
    ts = torch.cat([t[0, starts[i]:starts[i] + lens[i]] for i in perm], 0).unsqueeze(0)
    cu2 = torch.zeros(len(lens) + 1, dtype=torch.int32)
    cu2[1:] = torch.cumsum(torch.tensor(lens_p), 0)
    si2 = torch.repeat_interleave(
        torch.arange(len(lens), dtype=torch.int32), torch.tensor(lens_p)
    ).unsqueeze(0)
    _, pooled2 = enc(xs, x_sort_value=ts, seq_idx=si2, cu_seqlens=cu2)
    assert torch.max((pooled2 - pooled[perm]).abs()).item() < 1e-9


def test_no_cross_segment_attention():
    """Changing one track's hits must not change any other track's pooled
    vector (the block-diagonal attention mask)."""
    enc = build()
    lens = [7, 12, 9]
    x, t, si, cu = make_packed(lens)
    _, a = enc(x, x_sort_value=t, seq_idx=si, cu_seqlens=cu)
    x2 = x.clone()
    x2[0, int(cu[0]):int(cu[1])] = torch.randn_like(x2[0, int(cu[0]):int(cu[1])])
    _, b = enc(x2, x_sort_value=t, seq_idx=si, cu_seqlens=cu)
    assert torch.max((a[1:] - b[1:]).abs()).item() < 1e-10
    assert torch.max((a[0] - b[0]).abs()).item() > 1e-6


def test_posenc_is_the_only_order_signal():
    """With the posenc off the transformer is permutation-invariant in the
    hits; with it on it is not.  Recorded because the posenc is derived from
    the digitised hit time, an input the SSM does not receive."""
    enc = build()
    lens = [11]
    x, t, si, cu = make_packed(lens)
    _, a = enc(x, x_sort_value=t, seq_idx=si, cu_seqlens=cu)
    _, b = enc(x.flip(1).contiguous(), x_sort_value=t.flip(1).contiguous(),
               seq_idx=si, cu_seqlens=cu)
    assert torch.max((a - b).abs()).item() < 1e-9      # posenc travels with the token

    _, c = enc(x, x_sort_value=t.flip(1).contiguous(), seq_idx=si, cu_seqlens=cu)
    assert torch.max((a - c).abs()).item() > 1e-6      # order enters only via the key


# ---------------------------------------------------------------------------
# index-posenc variant (the transformer actually used in the study)
# ---------------------------------------------------------------------------

from track_regression.ablation_encoders import (  # noqa: E402
    IndexPosEncTransformerCLS,
    TimePosEncTransformerCLS,
)


def build_idx(dtype=torch.float64, scales=(-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5)):
    return IndexPosEncTransformerCLS(
        dim=DIM, num_cls_tokens=2, num_layers=3, attn_type="torch",
        norm="RMSNorm", value_residual=False, qkv_norm=True, layer_scale=1.0e-5,
        attn_kwargs={"num_heads": 4}, dense_kwargs={"hidden_dim_scale": 3},
        posenc_fourier_scales=list(scales), posenc_fourier_base=2,
        posenc_time_scale=1.0,
    ).to(dtype).eval()


def test_index_posenc_same_param_count():
    a = sum(p.numel() for p in build().parameters())
    b = sum(p.numel() for p in build_idx().parameters())
    assert a == b


def test_index_posenc_breaks_permutation_invariance():
    """With hit_time identically zero (v3) the stock posenc is constant and
    the encoder is order-blind; the index posenc must not be."""
    lens = [11]
    x, _, si, cu = make_packed(lens)
    zeros = torch.zeros(1, sum(lens), dtype=torch.float64)

    stock = build()
    _, a = stock(x, x_sort_value=zeros, seq_idx=si, cu_seqlens=cu)
    _, b = stock(x.flip(1).contiguous(), x_sort_value=zeros, seq_idx=si, cu_seqlens=cu)
    assert torch.max((a - b).abs()).item() < 1e-9          # order-blind

    idx = build_idx()
    _, c = idx(x, x_sort_value=zeros, seq_idx=si, cu_seqlens=cu)
    _, d = idx(x.flip(1).contiguous(), x_sort_value=zeros, seq_idx=si, cu_seqlens=cu)
    assert torch.max((c - d).abs()).item() > 1e-6          # order-aware


def test_index_posenc_segment_independence():
    enc = build_idx()
    lens = [7, 12, 20, 6, 13]
    x, t, si, cu = make_packed(lens)
    _, pooled = enc(x, x_sort_value=t, seq_idx=si, cu_seqlens=cu)
    perm = [3, 0, 4, 1, 2]
    lens_p = [lens[i] for i in perm]
    starts = [int(cu[i]) for i in range(len(lens))]
    xs = torch.cat([x[0, starts[i]:starts[i] + lens[i]] for i in perm], 0).unsqueeze(0)
    cu2 = torch.zeros(len(lens) + 1, dtype=torch.int32)
    cu2[1:] = torch.cumsum(torch.tensor(lens_p), 0)
    si2 = torch.repeat_interleave(
        torch.arange(len(lens), dtype=torch.int32), torch.tensor(lens_p)
    ).unsqueeze(0)
    _, pooled2 = enc(xs, x_sort_value=None, seq_idx=si2, cu_seqlens=cu2)
    assert torch.max((pooled2 - pooled[perm]).abs()).item() < 1e-9


def test_index_posenc_packed_vs_padded():
    enc = build_idx()
    lens = [7, 12, 20, 6, 13]
    x, t, si, cu = make_packed(lens)
    xp, tp, mask = to_padded(x, t, cu)
    _, a = enc(x, seq_idx=si, cu_seqlens=cu)
    _, b = enc(xp, kv_mask=mask)
    assert torch.max((a - b).abs()).item() < 1e-6


def test_padded_routing_matches_stock_packed_attention():
    """The study transformer routes packed batches through the padded
    attention path; that must equal the stock block-diagonal packed path
    (same function, O(B L^2) instead of O((B L)^2))."""
    lens = [7, 12, 20, 6, 13]
    x, t, si, cu = make_packed(lens)
    stock = build()
    stock.posenc_proj = None
    _, a = stock(x, x_sort_value=None, seq_idx=si, cu_seqlens=cu)

    routed = TimePosEncTransformerCLS(
        dim=DIM, num_cls_tokens=2, num_layers=3, attn_type="torch",
        norm="RMSNorm", value_residual=False, qkv_norm=True, layer_scale=1.0e-5,
        attn_kwargs={"num_heads": 4}, dense_kwargs={"hidden_dim_scale": 3},
        posenc_fourier_scales=[], posenc_fourier_base=2, posenc_time_scale=1.0,
    ).to(torch.float64).eval()
    routed.load_state_dict(stock.state_dict(), strict=False)
    _, b = routed(x, x_sort_value=None, seq_idx=si, cu_seqlens=cu)
    assert torch.max((a - b).abs()).item() < 1e-9


def test_time_posenc_variant_is_order_blind_on_v3():
    """v3 hit_time is identically zero -> the literal-protocol arm cannot see
    the hit order at all.  This is the diagnostic the report quotes."""
    lens = [11]
    x, _, si, cu = make_packed(lens)
    zeros = torch.zeros(1, sum(lens), dtype=torch.float64)
    enc = TimePosEncTransformerCLS(
        dim=DIM, num_cls_tokens=2, num_layers=3, attn_type="torch",
        norm="RMSNorm", value_residual=False, qkv_norm=True, layer_scale=1.0e-5,
        attn_kwargs={"num_heads": 4}, dense_kwargs={"hidden_dim_scale": 3},
        posenc_fourier_scales=[-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
        posenc_fourier_base=2, posenc_time_scale=1.0,
    ).to(torch.float64).eval()
    _, a = enc(x, x_sort_value=zeros, seq_idx=si, cu_seqlens=cu)
    _, b = enc(x.flip(1).contiguous(), x_sort_value=zeros, seq_idx=si, cu_seqlens=cu)
    assert torch.max((a - b).abs()).item() < 1e-9
