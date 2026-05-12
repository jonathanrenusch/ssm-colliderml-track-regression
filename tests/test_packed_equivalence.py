"""Unit tests for the packed-batch path of the SSM-CLS encoder.

Two layers of testing:

1. **Pure-torch helpers** (CPU, no Mamba dependency) — verify that
   :func:`_segment_flip_indices` correctly reverses each segment, that
   :func:`collate_tracks_packed` produces the right shapes / seq_idx /
   cu_seqlens, and that round-tripping a pad ↔ pack conversion of a
   batch is exact.

2. **End-to-end equivalence** (GPU, marked ``gpu``) — feed identical
   per-track inputs through :class:`BidirectionalMambaCLSEncoder` once
   in padded mode and once in packed mode; assert the pooled CLS
   outputs match to within fp32 round-off **when all tracks have the
   same length** (no padding tokens in the padded path → exact match
   modulo SSD-kernel rounding).

   For variable-length tracks the two paths diverge by a small but
   non-zero amount because padded mode lets zero-padded tokens
   contribute non-zero state to the bidirectional Mamba scan, while
   packed mode resets state cleanly at each segment boundary. The
   packed result is the physically correct one. Variable-length
   testing here is therefore a smoke test (correct shapes, finite
   outputs, gradient flow), not an equivalence assertion.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from track_regression.data import (
    collate_tracks,
    collate_tracks_packed,
)
from track_regression.mamba_cls import (
    _segment_flip_indices,
)


# ---------------------------------------------------------------------------
# Pure-torch helpers (run on CPU)
# ---------------------------------------------------------------------------


class TestSegmentFlipIndices:
    def test_basic_two_segments(self):
        """[0, 3, 5] → segments of length 3 and 2; reverse each in place."""
        cu_seqlens = torch.tensor([0, 3, 5], dtype=torch.long)
        idx = _segment_flip_indices(cu_seqlens, total_len=5)
        # Segment 0: positions 0,1,2 → reversed 2,1,0
        # Segment 1: positions 3,4   → reversed 4,3
        expected = torch.tensor([2, 1, 0, 4, 3], dtype=torch.long)
        assert torch.equal(idx, expected)

    def test_three_uneven_segments(self):
        cu_seqlens = torch.tensor([0, 4, 6, 9], dtype=torch.long)
        idx = _segment_flip_indices(cu_seqlens, total_len=9)
        # Seg 0 (len 4): 0,1,2,3 → 3,2,1,0
        # Seg 1 (len 2): 4,5     → 5,4
        # Seg 2 (len 3): 6,7,8   → 8,7,6
        expected = torch.tensor([3, 2, 1, 0, 5, 4, 8, 7, 6], dtype=torch.long)
        assert torch.equal(idx, expected)

    def test_singleton_segment(self):
        """A 1-token segment is its own reverse."""
        cu_seqlens = torch.tensor([0, 1, 4], dtype=torch.long)
        idx = _segment_flip_indices(cu_seqlens, total_len=4)
        expected = torch.tensor([0, 3, 2, 1], dtype=torch.long)
        assert torch.equal(idx, expected)

    def test_int32_cu_seqlens_accepted(self):
        """``cu_seqlens`` is often int32 (from collate); helper should cast."""
        cu_seqlens = torch.tensor([0, 3, 5], dtype=torch.int32)
        idx = _segment_flip_indices(cu_seqlens, total_len=5)
        expected = torch.tensor([2, 1, 0, 4, 3], dtype=torch.long)
        assert torch.equal(idx, expected)

    def test_double_application_recovers_identity(self):
        """Segment-wise reverse is its own inverse — gather twice → original."""
        torch.manual_seed(0)
        cu_seqlens = torch.tensor([0, 5, 9, 12], dtype=torch.long)
        total_len = 12
        D = 7
        x = torch.randn(1, total_len, D)
        idx = _segment_flip_indices(cu_seqlens, total_len)

        gather_idx = idx.unsqueeze(0).unsqueeze(-1).expand_as(x)
        x_flipped = torch.gather(x, 1, gather_idx)
        x_back = torch.gather(x_flipped, 1, gather_idx)
        torch.testing.assert_close(x_back, x)

    def test_flip_actually_reverses_each_segment(self):
        """Verify the flipped tensor matches a per-segment torch.flip."""
        torch.manual_seed(1)
        cu_seqlens = torch.tensor([0, 4, 7, 10], dtype=torch.long)
        total_len = 10
        D = 3
        x = torch.randn(1, total_len, D)
        idx = _segment_flip_indices(cu_seqlens, total_len)
        gather_idx = idx.unsqueeze(0).unsqueeze(-1).expand_as(x)
        x_flipped = torch.gather(x, 1, gather_idx)

        # Independently compute by reversing each segment with torch.flip.
        expected = torch.empty_like(x)
        for s in range(len(cu_seqlens) - 1):
            a, b = int(cu_seqlens[s]), int(cu_seqlens[s + 1])
            expected[0, a:b] = torch.flip(x[0, a:b], dims=[0])
        torch.testing.assert_close(x_flipped, expected)


# ---------------------------------------------------------------------------
# Collate function
# ---------------------------------------------------------------------------


def _make_track(L: int, D: int = 12, seed: int = 0) -> dict[str, np.ndarray]:
    """Build a synthetic preprocessed-track dict matching ``_load_track``."""
    rng = np.random.default_rng(seed)
    feats = rng.standard_normal((L, D)).astype(np.float32)
    # Column 6 is the input feature ``s``; populate plausibly even though the
    # sort key is ``hit_time``.
    feats[:, 6] = np.sort(rng.uniform(50.0, 1500.0, size=L)).astype(np.float32)
    # Truth time — strictly increasing within a track (matches the on-disk
    # time-sorted layout produced by preprocess_colliderml_compact v2).
    hit_time = np.sort(rng.uniform(0.0, 5.0, size=L)).astype(np.float32)
    return {
        "hit_features": feats,
        "hit_s": feats[:, 6].copy(),
        "hit_time": hit_time,
        "targets": rng.standard_normal(5).astype(np.float32),
        "length": L,
    }


class TestCollatePacked:
    def test_shapes_and_layout(self):
        batch = [_make_track(L, seed=i) for i, L in enumerate([5, 7, 3, 6])]
        inputs, targets = collate_tracks_packed(batch)

        total_L = 5 + 7 + 3 + 6
        assert inputs["hit_features"].shape == (1, total_L, 12)
        assert inputs["hit_s"].shape == (1, total_L)
        assert inputs["seq_idx"].shape == (1, total_L)
        assert inputs["cu_seqlens"].shape == (5,)
        assert inputs["cu_seqlens"].tolist() == [0, 5, 12, 15, 21]
        assert inputs["track_lengths"].tolist() == [5, 7, 3, 6]

    def test_seq_idx_matches_cu_seqlens(self):
        batch = [_make_track(L, seed=i) for i, L in enumerate([4, 2, 8])]
        inputs, _ = collate_tracks_packed(batch)
        seq = inputs["seq_idx"][0].tolist()
        # Expect [0]*4 + [1]*2 + [2]*8
        assert seq == [0] * 4 + [1] * 2 + [2] * 8

    def test_concatenation_preserves_per_track_features(self):
        """Each track's hit_features should land in the right packed slice."""
        batch = [_make_track(L, seed=i) for i, L in enumerate([3, 5])]
        inputs, _ = collate_tracks_packed(batch)
        feats = inputs["hit_features"][0]  # (8, 12)
        np.testing.assert_array_equal(feats[:3].numpy(), batch[0]["hit_features"])
        np.testing.assert_array_equal(feats[3:8].numpy(), batch[1]["hit_features"])

    def test_innermost_anchor_matches_padded(self):
        """innermost_phi/theta must agree between padded and packed collates."""
        batch = [_make_track(L, seed=i) for i, L in enumerate([4, 6, 5])]
        padded_inputs, padded_targets = collate_tracks(batch)
        packed_inputs, packed_targets = collate_tracks_packed(batch)
        torch.testing.assert_close(
            packed_targets["innermost_phi"], padded_targets["innermost_phi"],
        )
        torch.testing.assert_close(
            packed_targets["innermost_theta"], padded_targets["innermost_theta"],
        )

    def test_targets_match_padded(self):
        batch = [_make_track(L, seed=i) for i, L in enumerate([3, 7])]
        _, padded_targets = collate_tracks(batch)
        _, packed_targets = collate_tracks_packed(batch)
        for k in ["d0", "z0", "phi", "theta", "qop"]:
            torch.testing.assert_close(packed_targets[k], padded_targets[k])


# ---------------------------------------------------------------------------
# End-to-end encoder equivalence (GPU + Mamba required)
# ---------------------------------------------------------------------------


def _mamba_available() -> bool:
    try:
        import mamba_ssm  # noqa: F401
        return torch.cuda.is_available()
    except ImportError:
        return False


_REQUIRES_MAMBA_GPU = pytest.mark.skipif(
    not _mamba_available(),
    reason="requires CUDA and mamba_ssm",
)


def _build_encoder(num_layers: int = 2, dim: int = 64, chunk_size: int = 16):
    from track_regression.mamba_cls import (
        BidirectionalMambaCLSEncoder,
    )

    return BidirectionalMambaCLSEncoder(
        num_layers=num_layers,
        dim=dim,
        d_state=32,
        d_conv=4,
        expand=2,
        headdim=32,
        ngroups=1,
        chunk_size=chunk_size,
        norm="RMSNorm",
        dropout=0.0,
        cls_init_scale=0.02,
        residual_depth_init=False,
    )


def _build_padded_and_packed_inputs(track_lengths: list[int], dim: int):
    """Build padded ``(B, max_L, D)`` and packed ``(1, total_L, D)`` inputs
    that contain the *same* per-track tokens in the same intra-track order.
    """
    torch.manual_seed(123)
    B = len(track_lengths)
    max_L = max(track_lengths)
    total_L = sum(track_lengths)

    # Random per-track tokens (after the input embedding — random tensor).
    track_tokens = [torch.randn(L, dim) for L in track_lengths]

    padded = torch.zeros(B, max_L, dim)
    for i, t in enumerate(track_tokens):
        padded[i, : t.shape[0]] = t

    packed = torch.cat(track_tokens, dim=0).unsqueeze(0)  # (1, total_L, D)

    cu_seqlens = torch.zeros(B + 1, dtype=torch.int32)
    cu_seqlens[1:] = torch.cumsum(torch.tensor(track_lengths, dtype=torch.int32), 0)

    seq_idx = torch.empty(1, total_L, dtype=torch.int32)
    offset = 0
    for i, L in enumerate(track_lengths):
        seq_idx[0, offset:offset + L] = i
        offset += L

    return padded, packed, cu_seqlens, seq_idx


class TestPackedPaddedEquivalence:
    """End-to-end equivalence between the padded and packed encoder paths."""

    @_REQUIRES_MAMBA_GPU
    def test_same_length_tracks_match_exactly(self):
        """When every track has the same length there is no padding in the
        padded path either, so both layouts process identical token streams.
        Pooled CLS outputs must match to within fp32 round-off.
        """
        device = torch.device("cuda")
        dim = 64
        encoder = _build_encoder(num_layers=2, dim=dim).to(device).eval()

        track_lengths = [12, 12, 12, 12]
        padded, packed, cu_seqlens, seq_idx = _build_padded_and_packed_inputs(
            track_lengths, dim,
        )
        padded = padded.to(device)
        packed = packed.to(device)
        cu_seqlens = cu_seqlens.to(device)
        seq_idx = seq_idx.to(device)

        with torch.no_grad():
            _, padded_pool = encoder(padded, x_sort_value=None)
            _, packed_pool = encoder(packed, seq_idx=seq_idx, cu_seqlens=cu_seqlens)

        # Same-length tracks produce identical CLS-padded geometry; tolerance
        # absorbs SSD-kernel chunked-matmul reordering.
        torch.testing.assert_close(
            packed_pool, padded_pool, atol=1e-4, rtol=1e-4,
        )

    @_REQUIRES_MAMBA_GPU
    def test_variable_length_tracks_smoke(self):
        """Variable-length packed forward runs and produces correct shapes.

        We do *not* assert numerical equivalence with the padded path here:
        padded mode lets zero-padded tokens contribute non-zero state to
        the bidirectional Mamba scan, while packed mode resets state at
        every segment boundary. The packed output is the correct one.
        """
        device = torch.device("cuda")
        dim = 64
        encoder = _build_encoder(num_layers=2, dim=dim).to(device).eval()

        track_lengths = [8, 14, 6, 11]
        _, packed, cu_seqlens, seq_idx = _build_padded_and_packed_inputs(
            track_lengths, dim,
        )
        packed = packed.to(device)
        cu_seqlens = cu_seqlens.to(device)
        seq_idx = seq_idx.to(device)

        with torch.no_grad():
            seq_out, pool = encoder(packed, seq_idx=seq_idx, cu_seqlens=cu_seqlens)

        B = len(track_lengths)
        total_L = sum(track_lengths)
        assert seq_out.shape == (1, total_L, dim)
        assert pool.shape == (B, 2 * dim)
        assert torch.isfinite(pool).all()
        assert torch.isfinite(seq_out).all()

    @_REQUIRES_MAMBA_GPU
    def test_packed_gradients_flow_through_cls_tokens(self):
        """Training tie: cls_fwd / cls_bwd parameters must receive gradient
        in packed mode. A common bug here is building the augmented
        sequence via in-place writes to a torch.zeros tensor, which severs
        the autograd graph at the CLS positions. The argsort-based
        interleave avoids this — verify it.
        """
        device = torch.device("cuda")
        dim = 64
        encoder = _build_encoder(num_layers=2, dim=dim).to(device).train()

        track_lengths = [5, 9, 7]
        _, packed, cu_seqlens, seq_idx = _build_padded_and_packed_inputs(
            track_lengths, dim,
        )
        packed = packed.to(device).requires_grad_(True)
        cu_seqlens = cu_seqlens.to(device)
        seq_idx = seq_idx.to(device)

        _, pool = encoder(packed, seq_idx=seq_idx, cu_seqlens=cu_seqlens)
        loss = pool.float().sum()
        loss.backward()

        assert encoder.cls_fwd.grad is not None
        assert encoder.cls_bwd.grad is not None
        assert encoder.cls_fwd.grad.abs().sum().item() > 0.0
        assert encoder.cls_bwd.grad.abs().sum().item() > 0.0

    @_REQUIRES_MAMBA_GPU
    def test_packed_path_is_layout_invariant(self):
        """Reordering segments in the packed batch must permute the rows of
        the pooled output identically — i.e. segment 0's CLS readout
        depends only on its own tokens, not on which slot of the packed
        tensor it lives in. This catches bugs where a global op leaks
        across segments.
        """
        device = torch.device("cuda")
        dim = 64
        encoder = _build_encoder(num_layers=2, dim=dim).to(device).eval()

        track_lengths = [5, 9, 7]
        _, packed, cu_seqlens, seq_idx = _build_padded_and_packed_inputs(
            track_lengths, dim,
        )
        packed = packed.to(device)
        cu_seqlens = cu_seqlens.to(device)
        seq_idx = seq_idx.to(device)

        with torch.no_grad():
            _, pool_orig = encoder(packed, seq_idx=seq_idx, cu_seqlens=cu_seqlens)

        # Reverse segment order.
        new_order = [2, 1, 0]
        new_lengths = [track_lengths[i] for i in new_order]
        # Build reordered packed tensor by concatenating segment slices.
        seg_tensors = []
        for i, L in enumerate(track_lengths):
            a = int(cu_seqlens[i].item()); b = int(cu_seqlens[i + 1].item())
            seg_tensors.append(packed[0, a:b])
        reordered = torch.cat([seg_tensors[i] for i in new_order], dim=0).unsqueeze(0)
        new_cu = torch.zeros(len(new_lengths) + 1, dtype=torch.int32, device=device)
        new_cu[1:] = torch.cumsum(
            torch.tensor(new_lengths, dtype=torch.int32, device=device), 0,
        )
        new_seq_idx = torch.empty(1, sum(new_lengths), dtype=torch.int32, device=device)
        offset = 0
        for i, L in enumerate(new_lengths):
            new_seq_idx[0, offset:offset + L] = i
            offset += L

        with torch.no_grad():
            _, pool_reordered = encoder(
                reordered, seq_idx=new_seq_idx, cu_seqlens=new_cu,
            )

        # pool_reordered[i] should equal pool_orig[new_order[i]].
        for i, src in enumerate(new_order):
            torch.testing.assert_close(
                pool_reordered[i], pool_orig[src], atol=1e-4, rtol=1e-4,
            )
