"""Flat track stores: datasets, collate and the block sampler.

A flat store (written by ``scripts/preprocess_flat.py``) holds one directory
per split, ``{train,val,test}/part_XXXX/``, each part a CSR layout of the hits
of many tracks (``hits.npy``, ``offsets.npy``, ``lengths.npy``,
``targets.npy``).  Tracks are shuffled once at write time, so for training a
batch is read as one *contiguous* block of tracks (a sequential read; the
blocks themselves are visited in shuffled order).  Validation and test read
tracks in on-disk order so predictions line up with the store.

Batches are always *packed*: the hits of all tracks are concatenated into one
``(1, total_hits, n_features)`` tensor, with ``cu_seqlens`` marking the track
boundaries.  No padding is materialised anywhere in the data path.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch import Tensor
from torch.utils.data import Dataset, Sampler

from track_regression.seed import compress_residuals, seed_perigee, seed_residuals

TARGET_NAMES = ["d0", "z0", "phi", "theta", "qop"]


# ---------------------------------------------------------------------------
# store
# ---------------------------------------------------------------------------

class FlatTrackStore:
    """mmap handles over the parts of one split, with an O(log P) global index."""

    def __init__(self, root: str | Path, max_tracks: int | None = None):
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
        self._h: list[dict] | None = None   # opened lazily, after fork

    @staticmethod
    def _trim(counts: np.ndarray, max_tracks: int) -> np.ndarray:
        """Take a prefix of every part, so a capped split spans all input shards."""
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
                self._h.append({
                    "hits": np.load(p / "hits.npy", mmap_mode="r"),
                    "off": np.load(p / "offsets.npy", mmap_mode="r"),
                    "lens": np.load(p / "lengths.npy", mmap_mode="r"),
                    "targets": np.load(p / "targets.npy", mmap_mode="r"),
                })
        return self._h

    def __getstate__(self):
        s = self.__dict__.copy()
        s["_h"] = None          # never pickle mmaps into a worker
        return s

    def __len__(self) -> int:
        return self.n


# ---------------------------------------------------------------------------
# collate
# ---------------------------------------------------------------------------

def _pack(H, lens, targets, seed_residual_features=False):
    """Build a packed batch from the hits ``H`` (n_hits, 12) of ``len(lens)`` tracks.

    The analytic three-hit seed (:mod:`track_regression.seed`) is computed per
    track and returned as the ``seed_<param>`` targets, which the loss uses as
    anchors.  With ``seed_residual_features`` the three per-hit residuals to the
    seed helix (asinh du, asinh dv, s_helix) are appended as features 12-14.
    At inference the model computes the same seed on the GPU instead (see
    ``TrackParameterRegressor.forward``), so evaluation batches carry only the
    12 raw features.
    """
    B = len(lens)
    lens64 = lens.astype(np.int64, copy=False)
    cu = np.zeros(B + 1, np.int32)
    cu[1:] = np.cumsum(lens64)
    starts = cu[:-1].astype(np.int64)
    max_len = int(lens64.max())
    pos = np.arange(len(H), dtype=np.int64) - np.repeat(starts, lens64)
    row = np.repeat(np.arange(B, dtype=np.int64), lens64)
    xyz = np.zeros((B, max_len, 3), np.float64); vol = np.zeros((B, max_len), np.float64)
    hv = np.zeros((B, max_len), bool)
    xyz[row, pos] = H[:, :3]; vol[row, pos] = H[:, 7]; hv[row, pos] = True
    seed64 = seed_perigee(xyz, hv, vol)
    seed = seed64.astype(np.float32)
    if seed_residual_features:
        res = compress_residuals(seed_residuals(H[:, :3], seed64, row)).astype(np.float32)
        H = np.concatenate([H, res], axis=1)
    inputs = {
        "hit_features": torch.from_numpy(np.ascontiguousarray(H)).unsqueeze(0),
        "seq_idx": torch.from_numpy(np.repeat(np.arange(B, dtype=np.int32), lens64)).unsqueeze(0),
        "cu_seqlens": torch.from_numpy(cu),
        "track_lengths": torch.from_numpy(lens.astype(np.int32, copy=False)),
    }
    tgt = {n: torch.from_numpy(np.ascontiguousarray(targets[:, i]))
           for i, n in enumerate(TARGET_NAMES)}
    tgt["track_valid"] = torch.ones(B, dtype=torch.bool)
    for i, n in enumerate(TARGET_NAMES):
        tgt[f"seed_{n}"] = torch.from_numpy(np.ascontiguousarray(seed[:, i]))
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
    Hs, Ls, Gs = [], [], []
    for p in np.unique(parts):
        loc = idx[parts == p] - store.cum[p]
        e = h[p]
        lens = e["lens"][loc].astype(np.int64)
        csum = np.cumsum(lens)
        pos = np.arange(int(csum[-1]), dtype=np.int64) - np.repeat(csum - lens, lens)
        g = np.repeat(e["off"][loc], lens) + pos
        Hs.append(e["hits"][g]); Ls.append(lens); Gs.append(e["targets"][loc])
    j = lambda xs: xs[0] if len(xs) == 1 else np.concatenate(xs)     # noqa: E731
    return np.ascontiguousarray(j(Hs)), j(Ls), np.ascontiguousarray(j(Gs))


def _gather_block(store: FlatTrackStore, i0: int, i1: int):
    """Fetch a contiguous range: a single slice, no gather at all."""
    h = store.open()
    p = int(np.searchsorted(store.cum, i0, side="right") - 1)
    if i1 > store.cum[p + 1]:                      # straddles a part boundary
        return _gather_random(store, np.arange(i0, i1, dtype=np.int64))
    e = h[p]
    lo, hi = i0 - int(store.cum[p]), i1 - int(store.cum[p])
    a, b = int(e["off"][lo]), int(e["off"][hi])
    return (np.array(e["hits"][a:b]), e["lens"][lo:hi].astype(np.int64),
            np.array(e["targets"][lo:hi]))


# ---------------------------------------------------------------------------
# datasets
# ---------------------------------------------------------------------------

class FlatBlockTrackDataset(Dataset):
    """Training dataset: one item *is* one batch, fetched as a contiguous slice.

    Pair with :class:`BlockBatchSampler` and ``batch_size=None``.
    """

    def __init__(self, store: FlatTrackStore, seed_residual_features: bool = False):
        self.store = store
        self.seed_residual_features = seed_residual_features

    def __len__(self) -> int:
        return self.store.n

    def __getitem__(self, block) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        i0, i1 = int(block[0]), int(block[1])
        return _pack(*_gather_block(self.store, i0, i1), self.seed_residual_features)


class FlatTrackDataset(Dataset):
    """Random-access dataset for validation, test and prediction.

    Sample ``i`` is always track ``i`` of the split, in on-disk order, so the
    prediction writer's row order matches the store.  ``__getitems__`` lets the
    DataLoader fetch a whole batch in one vectorised call.
    """

    def __init__(self, store: FlatTrackStore, seed_residual_features: bool = False):
        self.store = store
        self.seed_residual_features = seed_residual_features

    def __len__(self) -> int:
        return self.store.n

    def __getitem__(self, i):
        return self.__getitems__([int(i)])

    def __getitems__(self, idx):
        return _pack(*_gather_random(self.store, np.asarray(idx, np.int64)),
                     self.seed_residual_features)


# ---------------------------------------------------------------------------
# sampler
# ---------------------------------------------------------------------------

class BlockBatchSampler(Sampler):
    """Contiguous index blocks, in shuffled order, split evenly across ranks.

    Every rank gets exactly ``len(self)`` blocks in every epoch.  ``jitter``
    moves the block boundaries each epoch so a track's batch companions are not
    frozen for the whole run; the block count is that of the worst-case offset,
    so the epoch length does not change with the jitter (Lightning schedules
    validation at ``batch_idx + 1 == len(loader)``).
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
        self._n_blocks = (self.n - self.bs + 1) // self.bs if self.jitter else self.n // self.bs
        self._per_rank = self._n_blocks // self.world

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def _starts(self, epoch: int) -> np.ndarray:
        rng = np.random.default_rng(self.seed + epoch)
        off = int(rng.integers(0, self.bs)) if self.jitter else 0
        starts = np.arange(off, self.n - self.bs + 1, self.bs, dtype=np.int64)[: self._n_blocks]
        rng.shuffle(starts)
        return starts

    def __iter__(self):
        starts = self._starts(self._epoch)
        mine = starts[self.rank::self.world][: self._per_rank]
        for s in mine:
            yield (int(s), int(s) + self.bs)

    def __len__(self) -> int:
        return self._per_rank
