"""Flat-store datasets for the muon-scale training set.

Why this exists alongside :mod:`track_regression.data`
------------------------------------------------------
The legacy streaming dataset was built for ttbar, where a batch drawn from one
event is strongly correlated (tracks share a vertex), so it shuffled a large
buffer of shards to decorrelate.  That buffer is what forces
``num_workers: 1``: each worker takes a disjoint shard subset, drops its own
``drop_last`` remainder, and the resulting per-worker batch counts disagree --
hence ``batches_per_epoch`` and the ``limit_train_batches`` override, and under
DDP an NCCL deadlock if they are wrong.

The drift_beamspot muon set has exactly one track per event, so tracks are
i.i.d. on disk and none of that is necessary.  Measured on the real data (per
batch target means against the CLT prediction, batch 2000): contiguous blocks
of the muon store score z = 0.93-1.09, indistinguishable from random batches;
ttbar in natural order scores 5.57.  Shuffling once at write time and reading
contiguous blocks is therefore a valid substitute for a global shuffle -- and
turns every batch into a sequential read.

Measured throughput at batch 10000 (packed), against 13.9 k tracks/s of
single-GPU compute:

    legacy streaming, num_workers=1        64 k tracks/s
    flat + per-item __getitem__            77 k
    flat + batched random-index fetch     849 k   (431 k cold cache)
    flat + contiguous blocks            1,408 k   (1,538 k cold cache)

The block reader needs no shuffle buffer: resident memory is one batch plus
mmap, flat in dataset size.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch import Tensor
from torch.utils.data import Dataset, Sampler

TARGET_NAMES = ["d0", "z0", "phi", "theta", "qop"]


# ---------------------------------------------------------------------------
# store
# ---------------------------------------------------------------------------

class FlatTrackStore:
    """mmap handles over the parts of one split, with an O(log P) global index.

    The legacy map-style dataset materialises a Python list of
    ``(shard, local)`` tuples -- 200 M tuples for this dataset, tens of GB per
    worker.  Here the global -> (part, local) map is one int64 array plus a
    ``searchsorted``, so it costs nothing and forks cheaply.
    """

    def __init__(self, root: str | Path, load_acts: bool = False,
                 max_tracks: int | None = None):
        self.root = Path(root)
        man_path = self.root / "manifest.json"
        if not man_path.exists():
            raise FileNotFoundError(
                f"{man_path} not found — expected a flat store written by "
                "scripts/preprocess_flat.py"
            )
        self.man = json.loads(man_path.read_text())
        if self.man.get("layout") != "flat_csr":
            raise ValueError(f"{man_path}: unsupported layout {self.man.get('layout')!r}")
        self.names = [p["name"] for p in self.man["parts"]]
        counts = np.asarray([p["n_tracks"] for p in self.man["parts"]], np.int64)
        self.full_n = int(counts.sum())
        if max_tracks is not None and max_tracks < self.full_n:
            counts = self._trim(counts, int(max_tracks))
        self.counts = counts
        self.cum = np.zeros(len(counts) + 1, np.int64)
        np.cumsum(counts, out=self.cum[1:])
        self.n = int(self.cum[-1])
        self.load_acts = load_acts
        self._h: list[dict] | None = None   # opened lazily, after fork

    @staticmethod
    def _trim(counts: np.ndarray, max_tracks: int) -> np.ndarray:
        """Take a prefix of every part, so the subsample spans all input shards.

        Each part is one input shard, internally shuffled at write time, so a
        prefix of a part is an unbiased sample of that shard.  Spreading the cap
        over all parts keeps every shard represented rather than dropping the
        later ones wholesale.
        """
        per = -(-max_tracks // len(counts))          # ceil
        out = np.minimum(counts, per)
        # give back any shortfall (parts smaller than `per`) to the larger parts
        while out.sum() < max_tracks:
            room = counts - out
            if not room.any():
                break
            take = min(max_tracks - int(out.sum()), int(room.max()))
            i = int(np.argmax(room))
            out[i] += take
        # and trim any overshoot from the end
        over = int(out.sum()) - max_tracks
        for i in range(len(out) - 1, -1, -1):
            if over <= 0:
                break
            d = min(over, int(out[i]))
            out[i] -= d
            over -= d
        return out

    def open(self) -> list[dict]:
        if self._h is None:
            self._h = []
            for nm in self.names:
                p = self.root / nm
                e = {
                    "hits": np.load(p / "hits.npy", mmap_mode="r"),
                    "times": np.load(p / "hit_times.npy", mmap_mode="r"),
                    # mmap these too: eager loads cost ~2.2 GB per worker on the
                    # 182-part muon store, and only a handful of entries are
                    # touched per batch.
                    "off": np.load(p / "offsets.npy", mmap_mode="r"),
                    "lens": np.load(p / "lengths.npy", mmap_mode="r"),
                    "targets": np.load(p / "targets.npy", mmap_mode="r"),
                }
                if self.load_acts and (p / "acts_reco.npy").exists():
                    e["acts"] = np.load(p / "acts_reco.npy", mmap_mode="r")
                    e["dm"] = np.load(p / "acts_dm.npy", mmap_mode="r")
                self._h.append(e)
        return self._h

    def __getstate__(self):
        s = self.__dict__.copy()
        s["_h"] = None          # never pickle mmaps into a worker
        return s

    def __len__(self) -> int:
        return self.n


# ---------------------------------------------------------------------------
# collate (vectorised; bit-identical to data.collate_tracks{,_packed})
# ---------------------------------------------------------------------------

def _pack(H, T, lens, targets, acts=None, dm=None, meta=None):
    """Packed batch, built without a Python loop over the batch."""
    B = len(lens)
    lens64 = lens.astype(np.int64, copy=False)
    cu = np.zeros(B + 1, np.int32)
    cu[1:] = np.cumsum(lens64)
    starts = cu[:-1].astype(np.int64)
    inputs = {
        "hit_features": torch.from_numpy(H).unsqueeze(0),
        "hit_s": torch.from_numpy(np.ascontiguousarray(H[:, 6])).unsqueeze(0),
        "hit_time": torch.from_numpy(T).unsqueeze(0),
        "seq_idx": torch.from_numpy(np.repeat(np.arange(B, dtype=np.int32), lens64)).unsqueeze(0),
        "cu_seqlens": torch.from_numpy(cu),
        "track_lengths": torch.from_numpy(lens.astype(np.int32, copy=False)),
    }
    # Hits are written time-sorted, so the innermost hit is a track's first hit;
    # the legacy collates' per-item argmin over the segment is redundant.
    tgt = {n: torch.from_numpy(np.ascontiguousarray(targets[:, i]))
           for i, n in enumerate(TARGET_NAMES)}
    tgt["track_valid"] = torch.ones(B, dtype=torch.bool)
    tgt["innermost_phi"] = torch.from_numpy(np.ascontiguousarray(H[starts, 4]))
    tgt["innermost_theta"] = torch.from_numpy(np.ascontiguousarray(H[starts, 5]))
    if acts is not None:
        for i, n in enumerate(TARGET_NAMES):
            tgt[f"acts_reco_{n}"] = torch.from_numpy(np.ascontiguousarray(acts[:, i]))
        tgt["acts_dm_mask"] = torch.from_numpy(dm.astype(bool, copy=False))
    if meta is not None:
        tgt["track_pt"] = torch.from_numpy(np.ascontiguousarray(meta[:, 0]))
        tgt["track_vertex_primary"] = torch.from_numpy(np.ascontiguousarray(meta[:, 1]))
    return inputs, tgt


def _pad(H, T, lens, targets, acts=None, dm=None, meta=None):
    """Padded batch ``(B, max_L, D)``, built with one scatter instead of a loop."""
    B = len(lens)
    lens64 = lens.astype(np.int64, copy=False)
    max_len = int(lens64.max())
    D = H.shape[1]
    csum = np.cumsum(lens64)
    pos = np.arange(len(H), dtype=np.int64) - np.repeat(csum - lens64, lens64)
    row = np.repeat(np.arange(B, dtype=np.int64), lens64)

    hf = torch.zeros(B, max_len, D, dtype=torch.float32)
    ht = torch.zeros(B, max_len, dtype=torch.float32)
    hv = torch.zeros(B, max_len, dtype=torch.bool)
    hf[row, pos] = torch.from_numpy(H)
    ht[row, pos] = torch.from_numpy(T)
    hv[row, pos] = True
    inputs = {"hit_features": hf, "hit_s": hf[..., 6].contiguous(),
              "hit_time": ht, "hit_valid": hv}
    _, tgt = _pack(H, T, lens, targets, acts, dm, meta)
    for k in ("cu_seqlens", "seq_idx", "track_lengths"):
        tgt.pop(k, None)
    return inputs, tgt


# ---------------------------------------------------------------------------
# gather helpers
# ---------------------------------------------------------------------------

def _gather_random(store: FlatTrackStore, idx: np.ndarray):
    """Fetch an arbitrary index set: one fancy-index gather per touched part."""
    h = store.open()
    parts = np.searchsorted(store.cum, idx, side="right") - 1
    o = np.argsort(parts, kind="stable")
    idx, parts = idx[o], parts[o]
    Hs, Ts, Ls, Gs, As, Ds, Ms = [], [], [], [], [], [], []
    for p in np.unique(parts):
        m = parts == p
        loc = idx[m] - store.cum[p]
        e = h[p]
        lens = e["lens"][loc].astype(np.int64)
        csum = np.cumsum(lens)
        pos = np.arange(int(csum[-1]), dtype=np.int64) - np.repeat(csum - lens, lens)
        g = np.repeat(e["off"][loc], lens) + pos
        Hs.append(e["hits"][g]); Ts.append(e["times"][g])
        Ls.append(lens); Gs.append(e["targets"][loc])
        if "acts" in e:
            As.append(e["acts"][loc]); Ds.append(e["dm"][loc])
    j = lambda xs: xs[0] if len(xs) == 1 else np.concatenate(xs)     # noqa: E731
    return (np.ascontiguousarray(j(Hs)), np.ascontiguousarray(j(Ts)), j(Ls),
            np.ascontiguousarray(j(Gs)),
            np.ascontiguousarray(j(As)) if As else None,
            j(Ds) if Ds else None, None)


def _gather_block(store: FlatTrackStore, i0: int, i1: int):
    """Fetch a contiguous range: a single slice, no gather at all."""
    h = store.open()
    p = int(np.searchsorted(store.cum, i0, side="right") - 1)
    if i1 > store.cum[p + 1]:                      # straddles a part boundary
        return _gather_random(store, np.arange(i0, i1, dtype=np.int64))
    e = h[p]
    lo, hi = i0 - int(store.cum[p]), i1 - int(store.cum[p])
    a, b = int(e["off"][lo]), int(e["off"][hi])
    return (np.array(e["hits"][a:b]), np.array(e["times"][a:b]),
            e["lens"][lo:hi].astype(np.int64), np.array(e["targets"][lo:hi]),
            np.array(e["acts"][lo:hi]) if "acts" in e else None,
            np.array(e["dm"][lo:hi]) if "dm" in e else None, None)


# ---------------------------------------------------------------------------
# datasets
# ---------------------------------------------------------------------------

class FlatBlockTrackDataset(Dataset):
    """Training dataset: one item *is* one batch, fetched as a contiguous slice.

    Pair with :class:`BlockBatchSampler` and ``batch_size=None``.
    """

    def __init__(self, store: FlatTrackStore, packed: bool = True):
        self.store = store
        self._build = _pack if packed else _pad

    def __len__(self) -> int:
        return self.store.n

    def __getitem__(self, block) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        i0, i1 = int(block[0]), int(block[1])
        return self._build(*_gather_block(self.store, i0, i1))


class FlatTrackDataset(Dataset):
    """Random-access dataset for validation, test and prediction.

    Drop-in for :class:`track_regression.data.ColliderMLTrackDataset`: sample
    ``i`` is always track ``i`` of the split, in on-disk order, so the
    prediction writer's row order is stable.  ``__getitems__`` lets the
    DataLoader fetch a whole batch in one vectorised call.
    """

    def __init__(self, store: FlatTrackStore, packed: bool = True):
        self.store = store
        self._build = _pack if packed else _pad

    def __len__(self) -> int:
        return self.store.n

    def __getitem__(self, i):
        return self.__getitems__([int(i)])

    def __getitems__(self, idx):
        return self._build(*_gather_random(self.store, np.asarray(idx, np.int64)))


# ---------------------------------------------------------------------------
# sampler
# ---------------------------------------------------------------------------

class BlockBatchSampler(Sampler):
    """Contiguous index blocks, in shuffled order, split evenly across ranks.

    Every rank gets exactly ``len(self)`` blocks, so there is no per-worker
    ``drop_last`` skew, no need to override ``limit_train_batches``, and no
    epoch-boundary NCCL mismatch.  ``jitter`` moves the block boundaries each
    epoch so a track's batch companions are not frozen for the whole run.
    """

    def __init__(self, n: int, batch_size: int, seed: int = 42,
                 jitter: bool = True, rank: int | None = None,
                 world_size: int | None = None):
        self.n, self.bs, self.seed, self.jitter = int(n), int(batch_size), int(seed), jitter
        if rank is None or world_size is None:
            if dist.is_available() and dist.is_initialized():
                rank, world_size = dist.get_rank(), dist.get_world_size()
            else:
                rank, world_size = 0, 1
        self.rank, self.world = rank, world_size
        self._epoch = 0
        self._per_rank = len(self._starts(0)) // self.world

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def _starts(self, epoch: int) -> np.ndarray:
        rng = np.random.default_rng(self.seed + epoch)
        off = int(rng.integers(0, self.bs)) if self.jitter else 0
        starts = np.arange(off, self.n - self.bs + 1, self.bs, dtype=np.int64)
        rng.shuffle(starts)
        return starts

    def __iter__(self):
        starts = self._starts(self._epoch)
        mine = starts[self.rank::self.world][: self._per_rank]
        for s in mine:
            yield (int(s), int(s) + self.bs)

    def __len__(self) -> int:
        return self._per_rank
