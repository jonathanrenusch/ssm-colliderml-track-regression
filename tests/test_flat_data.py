"""Flat store: collate, gather, and the block sampler."""

from __future__ import annotations

import pickle

import numpy as np
import pytest
import torch

from track_regression.flat_data import BlockBatchSampler, FlatBlockTrackDataset, FlatTrackDataset, FlatTrackStore
from track_regression.seed import seed_from_csr


def test_packed_batch_layout_and_seed(synthetic_store):
    store = FlatTrackStore(synthetic_store / "test")
    inputs, tgt = FlatTrackDataset(store, seed_residual_features=True).__getitems__(list(range(40, 60)))
    cu, lens = inputs["cu_seqlens"], inputs["track_lengths"]
    assert inputs["hit_features"].shape == (1, int(lens.sum()), 15)
    assert torch.equal(cu[1:] - cu[:-1], lens)
    assert torch.equal(inputs["seq_idx"][0], torch.repeat_interleave(torch.arange(20, dtype=torch.int32), lens))
    h = store.open()[0]
    a, b = int(h["off"][40]), int(h["off"][60])
    assert np.array_equal(inputs["hit_features"][0, :, :12].numpy(), h["hits"][a:b])
    assert np.array_equal(tgt["d0"].numpy(), h["targets"][40:60, 0])
    seed = seed_from_csr(np.asarray(h["hits"][a:b]), lens.numpy()).astype(np.float32)
    assert np.array_equal(tgt["seed_qop"].numpy(), seed[:, 4])


def test_block_straddling_parts_equals_random_access(synthetic_store):
    store = FlatTrackStore(synthetic_store / "train")
    i0 = int(store.cum[1]) - 5                                  # 5 tracks from part 0, 5 from part 1
    blk, _ = FlatBlockTrackDataset(store)[(i0, i0 + 10)]
    rnd, _ = FlatTrackDataset(store).__getitems__(list(range(i0, i0 + 10)))
    assert torch.equal(blk["hit_features"], rnd["hit_features"])


def test_store_does_not_pickle_mmaps(synthetic_store):
    store = FlatTrackStore(synthetic_store / "train")
    store.open()
    assert pickle.loads(pickle.dumps(store))._h is None


def test_max_tracks_spreads_over_all_parts(synthetic_store):
    store = FlatTrackStore(synthetic_store / "val", max_tracks=50)
    assert store.n == 50 and (store.counts > 0).all()


@pytest.mark.parametrize("world", [1, 2, 3])
def test_block_sampler(world):
    n, bs = 10_000, 64
    per_rank = [list(BlockBatchSampler(n, bs, rank=r, world_size=world)) for r in range(world)]
    lens = {len(b) for b in per_rank}
    assert len(lens) == 1                                        # identical count on every rank
    starts = sorted(s for b in per_rank for s, _ in b)
    assert all(e - s == bs for b in per_rank for s, e in b)
    assert all(b - a >= bs for a, b in zip(starts, starts[1:]))  # no overlap
    s = BlockBatchSampler(n, bs)
    for epoch in range(5):                                       # constant length under the jitter
        s.set_epoch(epoch)
        assert len(list(s)) == len(s)
    s.set_epoch(1)
    assert list(s) != list(BlockBatchSampler(n, bs))             # layout changes between epochs
