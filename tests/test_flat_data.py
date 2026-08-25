"""Unit tests for the flat training store.

The flat path replaces the legacy per-shard streaming dataset for the
drift_beamspot muon set.  Two properties have to hold for it to be a safe
substitute, and both are cheap to assert on a synthetic store:

1. **Collate equivalence.**  The vectorised packed/padded builders in
   :mod:`track_regression.flat_data` must produce exactly what
   :func:`track_regression.data.collate_tracks_packed` /
   :func:`~track_regression.data.collate_tracks` produce for the same tracks.
   They skip the per-item Python loop and the per-segment ``argmin`` over hit
   time (the writer stores hits time-sorted, so the innermost hit is the first
   one), so the shortcut needs guarding.

2. **Sampler contract.**  Blocks must tile the split without overlap, differ
   between epochs, and hand every DDP rank an identical batch count — the last
   one is what removes the ``limit_train_batches`` override the streaming path
   needed.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from track_regression.data import collate_tracks, collate_tracks_packed
from track_regression.flat_data import (
    BlockBatchSampler,
    FlatBlockTrackDataset,
    FlatTrackDataset,
    FlatTrackStore,
)

N_FEAT = 12


@pytest.fixture(scope="module")
def flat_store(tmp_path_factory):
    """A two-part synthetic store with realistic ragged track lengths."""
    rng = np.random.default_rng(0)
    root = tmp_path_factory.mktemp("flat")
    parts = []
    for pi, n_tracks in enumerate((137, 91)):
        lens = rng.integers(6, 21, size=n_tracks).astype(np.int32)
        off = np.zeros(n_tracks + 1, np.int64)
        np.cumsum(lens, out=off[1:])
        n_hits = int(off[-1])
        hits = rng.normal(size=(n_hits, N_FEAT)).astype(np.float32)
        # hit_time must be non-decreasing within a track: the writer sorts by it
        times = np.concatenate([
            np.sort(rng.normal(size=int(L)).astype(np.float32)) for L in lens
        ])
        d = root / f"part_{pi:04d}"
        d.mkdir()
        np.save(d / "hits.npy", hits)
        np.save(d / "hit_times.npy", times)
        np.save(d / "offsets.npy", off)
        np.save(d / "lengths.npy", lens)
        np.save(d / "targets.npy", rng.normal(size=(n_tracks, 5)).astype(np.float32))
        np.save(d / "acts_reco.npy", rng.normal(size=(n_tracks, 5)).astype(np.float32))
        np.save(d / "acts_dm.npy", rng.random(n_tracks) > 0.5)
        parts.append({"name": d.name, "n_tracks": int(n_tracks), "n_hits": n_hits})
    (root / "manifest.json").write_text(json.dumps({
        "layout": "flat_csr", "version": 3, "n_feat": N_FEAT, "parts": parts,
        "n_tracks": sum(p["n_tracks"] for p in parts),
        "n_hits": sum(p["n_hits"] for p in parts),
    }))
    return root


def _reference_items(store, idx):
    """The legacy per-item dicts for a set of global indices."""
    h = store.open()
    out = []
    for i in idx:
        p = int(np.searchsorted(store.cum, i, side="right") - 1)
        loc = int(i - store.cum[p])
        e = h[p]
        a, b = int(e["off"][loc]), int(e["off"][loc + 1])
        hf = np.array(e["hits"][a:b])
        out.append({
            "hit_features": hf,
            "hit_s": hf[:, 6].copy(),
            "hit_time": np.array(e["times"][a:b]),
            "targets": np.array(e["targets"][loc]),
            "length": b - a,
        })
    return out


def _assert_same(ref, got, keys):
    for k in keys:
        assert k in got, f"missing key {k}"
        assert got[k].shape == ref[k].shape, f"{k}: {got[k].shape} != {ref[k].shape}"
        assert got[k].dtype == ref[k].dtype, f"{k}: {got[k].dtype} != {ref[k].dtype}"
        assert torch.equal(got[k], ref[k]), f"{k}: values differ"


@pytest.mark.parametrize("contiguous", [True, False])
def test_packed_matches_legacy_collate(flat_store, contiguous):
    """Vectorised packed build == collate_tracks_packed, bit for bit."""
    store = FlatTrackStore(flat_store)
    idx = (np.arange(40, 140) if contiguous
           else np.sort(np.random.default_rng(1).choice(store.n, 100, replace=False)))
    ref = collate_tracks_packed(_reference_items(store, idx))

    if contiguous:
        got = FlatBlockTrackDataset(FlatTrackStore(flat_store))[(int(idx[0]), int(idx[-1]) + 1)]
    else:
        got = FlatTrackDataset(FlatTrackStore(flat_store)).__getitems__(idx)

    _assert_same(ref[0], got[0],
                 ["hit_features", "hit_s", "hit_time", "seq_idx", "cu_seqlens", "track_lengths"])
    _assert_same(ref[1], got[1],
                 ["d0", "z0", "phi", "theta", "qop", "innermost_phi", "innermost_theta"])


def test_padded_matches_legacy_collate(flat_store):
    """Vectorised padded build == collate_tracks, bit for bit."""
    store = FlatTrackStore(flat_store)
    idx = np.arange(10, 90)
    ref = collate_tracks(_reference_items(store, idx))
    got = FlatTrackDataset(FlatTrackStore(flat_store), packed=False).__getitems__(idx)
    _assert_same(ref[0], got[0], ["hit_features", "hit_s", "hit_time", "hit_valid"])
    _assert_same(ref[1], got[1], ["d0", "z0", "phi", "theta", "qop"])


def test_block_straddling_a_part_boundary(flat_store):
    """A block crossing two parts must still match the per-item reference."""
    store = FlatTrackStore(flat_store)
    lo = int(store.cum[1]) - 20
    idx = np.arange(lo, lo + 40)
    assert idx[0] < store.cum[1] < idx[-1], "fixture no longer straddles a boundary"
    ref = collate_tracks_packed(_reference_items(store, idx))
    got = FlatBlockTrackDataset(FlatTrackStore(flat_store))[(int(idx[0]), int(idx[-1]) + 1)]
    _assert_same(ref[0], got[0], ["hit_features", "hit_time", "cu_seqlens", "track_lengths"])


def test_eval_dataset_preserves_on_disk_order(flat_store):
    """Sample i is track i, so the prediction writer's row order is stable."""
    store = FlatTrackStore(flat_store, load_acts=True)
    ds = FlatTrackDataset(store)
    _, tgt = ds.__getitems__(np.arange(50))
    on_disk = np.load(flat_store / "part_0000" / "targets.npy")[:50]
    assert np.allclose(tgt["d0"].numpy(), on_disk[:, 0])
    assert "acts_reco_d0" in tgt and "acts_dm_mask" in tgt


def test_blocks_tile_without_overlap(flat_store):
    n = FlatTrackStore(flat_store).n
    s = BlockBatchSampler(n, 32, seed=3)
    blocks = sorted(list(s))
    assert len(blocks) == len(s)
    for (a0, a1), (b0, _) in zip(blocks, blocks[1:]):
        assert a1 <= b0, "blocks overlap within an epoch"
    assert all(b - a == 32 for a, b in blocks)


def test_epoch_changes_the_block_layout(flat_store):
    n = FlatTrackStore(flat_store).n
    s = BlockBatchSampler(n, 32, seed=3)
    s.set_epoch(0)
    first = list(s)
    s.set_epoch(1)
    assert list(s) != first


@pytest.mark.parametrize("world_size", [1, 2, 3, 4, 8])
def test_every_rank_gets_the_same_batch_count(flat_store, world_size):
    """Exact DDP balance — this is what removes the limit_train_batches hack."""
    n = FlatTrackStore(flat_store).n
    counts = [len(BlockBatchSampler(n, 16, rank=r, world_size=world_size))
              for r in range(world_size)]
    assert len(set(counts)) == 1, counts
    assert counts[0] > 0


def test_store_does_not_pickle_mmaps(flat_store):
    """Workers must re-open their own handles after fork."""
    import pickle
    store = FlatTrackStore(flat_store)
    store.open()
    revived = pickle.loads(pickle.dumps(store))
    assert revived._h is None
    assert revived.n == store.n


def test_max_tracks_spreads_the_cap_over_all_parts(flat_store):
    """Capping val/test must sample every part, not truncate to a prefix of parts."""
    full = FlatTrackStore(flat_store)
    cap = FlatTrackStore(flat_store, max_tracks=100)
    assert cap.n == 100
    assert cap.full_n == full.n
    assert (cap.counts > 0).all(), "every part must still contribute"
    # indices must stay inside each part's own prefix
    ds = FlatTrackDataset(cap)
    _, tgt = ds.__getitems__(np.arange(cap.n))
    assert tgt["d0"].shape == (100,)


def test_max_tracks_above_the_split_size_is_a_no_op(flat_store):
    full = FlatTrackStore(flat_store)
    assert FlatTrackStore(flat_store, max_tracks=10**9).n == full.n
    assert FlatTrackStore(flat_store, max_tracks=None).n == full.n


def test_capped_store_rows_come_from_the_right_parts(flat_store):
    """Row i of the capped store is row i of that part, not of the full store."""
    cap = FlatTrackStore(flat_store, max_tracks=60)
    ds = FlatTrackDataset(cap)
    _, tgt = ds.__getitems__(np.arange(int(cap.counts[0])))
    on_disk = np.load(flat_store / "part_0000" / "targets.npy")[: int(cap.counts[0])]
    assert np.allclose(tgt["d0"].numpy(), on_disk[:, 0])
