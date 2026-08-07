"""ColliderML track regression DataModule.

Reads the preprocessed memmap format produced by
:mod:`scripts.preprocess_colliderml` and provides PyTorch DataLoaders with:

- Track-level random access via CSR-indexed hit arrays
- Dynamic padding to batch-max length
- Deterministic train / val / test splits from a ``split.json`` file

All track selection (min/max hits, kinematics, perigee ranges) is applied
at preprocessing time.  The dataset trusts that every track stored in the
preprocessed shards passes selection and loads them unconditionally.
"""

from __future__ import annotations

import gc
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from lightning import LightningDataModule
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, IterableDataset, Sampler


def _resolve_data_path(p: str | Path) -> Path:
    """Expand ``~`` in a path. Configs ship with absolute paths, but
    ``expanduser`` is kept for the rare ``~/...`` override on the CLI.
    """
    return Path(os.path.expanduser(str(p)))


class _StratifiedTailSampler(Sampler[int]):
    """Two-group stratified sampler: a fixed fraction of each epoch's draws
    comes from the 'tail' index group, the rest from the 'core' group.

    Replacement for ``WeightedRandomSampler`` that avoids ``torch.multinomial``'s
    2^24 category limit.  Indices are sampled uniformly within each group
    using ``numpy.random.integers``, which has no such cap.  Draws are with
    replacement (matches WeightedRandomSampler semantics).

    Parameters
    ----------
    tail_indices, core_indices : np.ndarray[int64]
        Global indices into the underlying dataset.
    num_samples : int
        Total draws per epoch (typically ``len(dataset)``).
    target_tail_fraction : float
        Fraction of each draw coming from the tail group.  0.5 balances
        a heavily imbalanced dataset; values outside [0, 1] are clamped.
    seed : int
        Base seed; per-epoch state is seed + epoch_idx.
    """

    def __init__(
        self,
        tail_indices: np.ndarray,
        core_indices: np.ndarray,
        num_samples: int,
        target_tail_fraction: float,
        seed: int = 42,
    ):
        self.tail_indices = np.asarray(tail_indices, dtype=np.int64)
        self.core_indices = np.asarray(core_indices, dtype=np.int64)
        self.num_samples = int(num_samples)
        self.target_tail_fraction = float(max(0.0, min(1.0, target_tail_fraction)))
        self.seed = int(seed)
        self._epoch = 0

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self._epoch)
        self._epoch += 1
        n_tail = int(round(self.num_samples * self.target_tail_fraction))
        n_core = self.num_samples - n_tail
        parts = []
        if n_tail > 0 and len(self.tail_indices) > 0:
            parts.append(self.tail_indices[rng.integers(0, len(self.tail_indices), size=n_tail)])
        if n_core > 0 and len(self.core_indices) > 0:
            parts.append(self.core_indices[rng.integers(0, len(self.core_indices), size=n_core)])
        out = np.concatenate(parts) if parts else np.zeros(0, dtype=np.int64)
        rng.shuffle(out)
        return iter(out.tolist())

    def __len__(self):
        return self.num_samples


# ============================================================================
# Dataset
# ============================================================================


class ColliderMLTrackDataset(Dataset):
    """Memory-mapped dataset for track parameter regression.

    Each sample is a single selected track, yielding:
    - ``hit_features``: ``(L, 12)`` float32 — per-hit feature vectors
    - ``hit_s``: ``(L,)`` float32 — distance from IP (for sorting)
    - ``targets``: ``(5,)`` float32 — [d0, z0, phi, theta, qop]

    All track selection (min/max hits, kinematics, d0/z0 range) is applied
    at preprocessing time.  This dataset loads every track unconditionally.

    Parameters
    ----------
    preprocessed_dir : str | Path
        Root directory with ``shard_XXXX/`` subdirectories.
    shard_indices : list[int]
        Which shards to include in this dataset split.
    """

    def __init__(
        self,
        preprocessed_dir: str | Path,
        shard_indices: list[int],
        load_acts: bool = False,
    ):
        super().__init__()
        self.preprocessed_dir = _resolve_data_path(preprocessed_dir)
        self.load_acts = load_acts

        # Build global index: list of (shard_idx, local_track_idx)
        # and cache memmap references per shard
        self._shard_data: dict[int, dict[str, np.ndarray]] = {}
        self._global_index: list[tuple[int, int]] = []

        for si in sorted(shard_indices):
            shard_dir = self.preprocessed_dir / f"shard_{si:04d}"
            if not shard_dir.exists():
                continue

            shard_entry = _open_shard(self.preprocessed_dir, si, load_acts)
            self._shard_data[si] = shard_entry

            n_tracks = len(shard_entry["targets"])
            for t in range(n_tracks):
                self._global_index.append((si, t))

    def __len__(self) -> int:
        return len(self._global_index)

    def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
        shard_idx, local_idx = self._global_index[idx]
        return _load_track(self._shard_data[shard_idx], local_idx)


# ============================================================================
# Shared track-loading helper
# ============================================================================


def _load_track(data: dict[str, np.ndarray], local_idx: int) -> dict[str, np.ndarray]:
    """Load a single track from mmap'd shard arrays.

    Used by both :class:`ColliderMLTrackDataset` (map-style) and
    :class:`ColliderMLStreamingDataset` (iterable) to ensure identical output.
    """
    offsets = data["offsets"]
    start = int(offsets[local_idx])
    end = int(offsets[local_idx + 1])
    hit_idx = np.array(data["hit_indices"][start:end])

    # Gather hit features.
    # New format (12 cols): [x, y, z, r, phi_hit, theta_hit, s, volume_id,
    #                        layer_id, surface_id, detector, eta_hit]
    # Legacy format (11 cols): same without eta_hit — derive it here so old
    # shards (loose, core_kf_hits) continue to work.
    hit_feats = np.array(data["hits"][hit_idx])

    if hit_feats.shape[1] < 12:
        theta_hit = hit_feats[:, 5].copy()
        eta_hit = -np.log(np.tan(np.clip(theta_hit, 1e-8, np.pi - 1e-8) / 2.0))
        eta_hit = np.clip(eta_hit, -10.0, 10.0)
        hit_feats = np.concatenate([hit_feats, eta_hit[:, None]], axis=1).astype(np.float32)
    else:
        hit_feats = hit_feats.astype(np.float32, copy=False)

    hit_s = hit_feats[:, 6].copy()  # s column — kept as input feature 6
    # Truth-time sidecar (used as encoder sort key; on-disk hits are
    # already pre-sorted by this value, so it's also strictly
    # non-decreasing within each track).
    hit_time = np.array(data["hit_times"][hit_idx]).astype(np.float32, copy=False)
    targets = np.array(data["targets"][local_idx])  # (5,)

    result: dict[str, np.ndarray] = {
        "hit_features": hit_feats,
        "hit_s": hit_s,
        "hit_time": hit_time,
        "targets": targets,
        "length": len(hit_idx),
    }

    # ACTS reco data (when available)
    if "acts_reco" in data:
        result["acts_reco"] = np.array(data["acts_reco"][local_idx])  # (5,)
        result["acts_dm"] = bool(data["acts_dm_mask"][local_idx])

    # Track metadata (pt, vertex_primary) for tight selection filtering
    if "track_meta" in data:
        meta = np.array(data["track_meta"][local_idx])  # (2,): [pt, vertex_primary]
        result["track_pt"] = float(meta[0])
        result["track_vertex_primary"] = float(meta[1])

    return result


def _open_shard(preprocessed_dir: Path, si: int, load_acts: bool) -> dict[str, np.ndarray]:
    """Open mmap handles for a single shard."""
    shard_dir = preprocessed_dir / f"shard_{si:04d}"
    sel_dir = shard_dir / "selected_tracks"
    hit_times_path = shard_dir / "hit_times.npy"
    if not hit_times_path.exists():
        raise FileNotFoundError(
            f"hit_times.npy missing under {shard_dir}. The preprocessed shards "
            "must be regenerated with the v2 preprocessor (time-sort + sidecar). "
            "Re-run scripts.preprocess_colliderml_compact on this dataset."
        )
    entry: dict[str, np.ndarray] = {
        "hits": np.load(shard_dir / "hits.npy", mmap_mode="r"),
        "hit_times": np.load(hit_times_path, mmap_mode="r"),
        "targets": np.load(sel_dir / "track_targets.npy", mmap_mode="r"),
        "offsets": np.load(sel_dir / "track_hit_offsets.npy", mmap_mode="r"),
        "hit_indices": np.load(sel_dir / "track_hit_indices.npy", mmap_mode="r"),
    }
    if load_acts:
        acts_reco_path = sel_dir / "acts_reco.npy"
        acts_dm_path = sel_dir / "acts_dm_mask.npy"
        if acts_reco_path.exists() and acts_dm_path.exists():
            entry["acts_reco"] = np.load(acts_reco_path, mmap_mode="r")
            entry["acts_dm_mask"] = np.load(acts_dm_path, mmap_mode="r")
        meta_path = sel_dir / "track_meta.npy"
        if meta_path.exists():
            entry["track_meta"] = np.load(meta_path, mmap_mode="r")
    return entry


# ============================================================================
# Streaming (shard-shuffled) dataset for large-scale training
# ============================================================================


class ColliderMLStreamingDataset(IterableDataset):
    """Shard-shuffled streaming dataset that keeps only a small window of
    shards in memory at a time.

    Each epoch: shuffle shard order → iterate in chunks of
    ``shard_buffer_size`` → within each chunk, shuffle tracks locally →
    yield one track at a time → release chunk mmaps before loading next.

    DDP + multi-worker partitioning is handled internally: each
    (rank, worker) pair gets a disjoint subset of shards.

    Parameters
    ----------
    preprocessed_dir : str | Path
        Root directory with ``shard_XXXX/`` subdirectories.
    shard_indices : list[int]
        Which shards to include (from split.json).
    load_acts : bool
        Whether to load ACTS reco data per track.
    shard_buffer_size : int
        Number of shards to hold in memory simultaneously (default 8).
    seed : int
        Base seed for deterministic shard shuffling.
    """

    def __init__(
        self,
        preprocessed_dir: str | Path,
        shard_indices: list[int],
        load_acts: bool = False,
        shard_buffer_size: int = 8,
        seed: int = 42,
    ):
        super().__init__()
        self.preprocessed_dir = _resolve_data_path(preprocessed_dir)
        self.shard_indices = list(shard_indices)
        self.load_acts = load_acts
        self.shard_buffer_size = shard_buffer_size
        self.seed = seed
        self._epoch = 0

        # Pre-scan track counts per shard so we know total length for logging
        self._tracks_per_shard: dict[int, int] = {}
        for si in self.shard_indices:
            targets_path = (
                self.preprocessed_dir / f"shard_{si:04d}" / "selected_tracks" / "track_targets.npy"
            )
            if targets_path.exists():
                # Read just the shape from the npy header — no data loaded
                self._tracks_per_shard[si] = np.load(targets_path, mmap_mode="r").shape[0]

    def set_epoch(self, epoch: int) -> None:
        """Update epoch for deterministic shard shuffling."""
        self._epoch = epoch

    def __len__(self) -> int:
        """Track count for this dataset, DDP-corrected when called in a distributed context.

        Returns the per-rank track count when DDP is active so that Lightning's
        step estimates (and LR scheduler total_steps) are correct.
        """
        total = sum(self._tracks_per_shard.values())
        if dist.is_available() and dist.is_initialized():
            return total // dist.get_world_size()
        return total

    def _min_tracks_across_ranks(self, shuffled_shards: np.ndarray, world_size: int) -> int:
        """Compute the minimum per-rank track count for this epoch's shard shuffle.

        All ranks independently run the same deterministic computation, so no
        inter-rank communication is required.  The minimum is used to cap each
        rank's yield so that all ranks produce exactly the same number of tracks,
        preventing NCCL all-reduce deadlocks at epoch boundaries.
        """
        rank_totals = [
            sum(self._tracks_per_shard.get(int(s), 0) for s in shuffled_shards[r::world_size])
            for r in range(world_size)
        ]
        return min(rank_totals)

    def batches_per_epoch(self, batch_size: int, num_workers: int) -> int:
        """Exact number of batches the DataLoader will yield this epoch.

        The naive DataLoader length estimate (``total_tracks // batch_size``)
        overestimates the real count: with N workers each worker batches its
        own shard subset independently, so ``drop_last`` discards up to
        ``batch_size - 1`` tracks *per worker*, not per epoch.  Lightning uses
        the estimate to schedule end-of-epoch validation
        (``val_check_batch``); when the iterator exhausts before that batch
        index, validation is silently skipped — and under DDP, ranks that
        disagree on the batch count deadlock in the epoch-end NCCL sync.

        This mirrors ``__iter__``'s epoch shard shuffle, rank/worker
        partitioning and per-worker yield caps, then applies the per-worker
        ``drop_last`` flooring.  Returns the minimum across DDP ranks so all
        ranks can be capped to an identical step count.
        """
        if dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()
        else:
            world_size = 1
        num_workers = max(1, num_workers)

        rng = np.random.RandomState(self.seed + self._epoch)
        shards = np.array(self.shard_indices)
        rng.shuffle(shards)

        min_rank_total = (
            self._min_tracks_across_ranks(shards, world_size) if world_size > 1 else 0
        )

        rank_batches = []
        for r in range(world_size):
            shards_for_rank = shards[r::world_size]
            worker_totals = [
                sum(self._tracks_per_shard.get(int(s), 0)
                    for s in shards_for_rank[w::num_workers])
                for w in range(num_workers)
            ]
            if world_size > 1:
                # Same cap arithmetic as __iter__
                this_rank_total = sum(worker_totals)
                if this_rank_total > 0:
                    caps = [
                        min_rank_total * wt // this_rank_total for wt in worker_totals
                    ]
                    shortfall = min_rank_total - sum(caps)
                    for i in range(shortfall):
                        caps[i] += 1
                    worker_totals = caps
                else:
                    worker_totals = [0] * num_workers
            rank_batches.append(sum(wt // batch_size for wt in worker_totals))
        return min(rank_batches)

    def __iter__(self):
        # ---- Determine this worker's shard subset ----
        # DDP rank partitioning
        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        else:
            rank, world_size = 0, 1

        # DataLoader worker partitioning
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
        else:
            num_workers, worker_id = 1, 0

        # Deterministic shard shuffle for this epoch
        rng = np.random.RandomState(self.seed + self._epoch)
        shards = np.array(self.shard_indices)
        rng.shuffle(shards)

        # Partition: first by rank, then by worker (round-robin)
        shards_for_rank = shards[rank::world_size]
        shards_for_worker = shards_for_rank[worker_id::num_workers]

        # ---- Compute per-worker yield cap to balance batches across DDP ranks ----
        # Shards have varying track counts, so ranks end up with different totals.
        # The rank that finishes first causes an NCCL all-reduce deadlock because
        # the other ranks are still in training_step waiting for a gradient sync.
        # Fix: cap every rank to the same track count (the minimum across ranks),
        # distributed exactly across this rank's workers with remainder handling.
        if world_size > 1:
            min_rank_total = self._min_tracks_across_ranks(shards, world_size)
            this_rank_total = sum(self._tracks_per_shard.get(int(s), 0) for s in shards_for_rank)
            if this_rank_total > 0:
                # Compute caps for ALL workers on this rank deterministically.
                # Each worker independently runs this same computation, then picks
                # its own cap.  This avoids int() truncation rounding errors that
                # caused different ranks to yield different totals.
                all_worker_totals = [
                    sum(self._tracks_per_shard.get(int(s), 0)
                        for s in shards_for_rank[w::num_workers])
                    for w in range(num_workers)
                ]
                all_worker_caps = [
                    min_rank_total * wt // this_rank_total
                    for wt in all_worker_totals
                ]
                # Distribute remainder so sum(caps) == min_rank_total exactly
                shortfall = min_rank_total - sum(all_worker_caps)
                for i in range(shortfall):
                    all_worker_caps[i] += 1
                worker_max = all_worker_caps[worker_id]
            else:
                worker_max = 0
        else:
            worker_max = None  # no cap needed for single-GPU

        # ---- Stream through shard chunks ----
        buf_size = self.shard_buffer_size
        yielded = 0
        for chunk_start in range(0, len(shards_for_worker), buf_size):
            if worker_max is not None and yielded >= worker_max:
                break

            chunk_shard_ids = shards_for_worker[chunk_start : chunk_start + buf_size]

            # Open mmaps for this chunk
            shard_data = {}
            track_index = []  # (shard_id, local_idx)
            for si in chunk_shard_ids:
                si = int(si)
                shard_dir = self.preprocessed_dir / f"shard_{si:04d}"
                if not shard_dir.exists():
                    continue
                data = _open_shard(self.preprocessed_dir, si, self.load_acts)
                shard_data[si] = data
                n_tracks = len(data["targets"])
                for t in range(n_tracks):
                    track_index.append((si, t))

            # Shuffle tracks within this chunk
            chunk_rng = np.random.RandomState(self.seed + self._epoch * 10000 + chunk_start)
            chunk_rng.shuffle(track_index)

            # Yield tracks (respecting the per-worker cap)
            for si, local_idx in track_index:
                if worker_max is not None and yielded >= worker_max:
                    break
                yield _load_track(shard_data[si], local_idx)
                yielded += 1

            # Release mmaps for this chunk
            del shard_data, track_index
            gc.collect()



# ============================================================================
# Collate function (dynamic padding)
# ============================================================================


def collate_tracks(batch: list[dict[str, np.ndarray]]) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    """Collate variable-length tracks into padded tensors.

    Returns
    -------
    inputs : dict
        - ``hit_features``: ``(B, max_L, D)`` float32 (D=12: 11 raw + eta)
        - ``hit_s``: ``(B, max_L)`` float32 (kept for the input feature in
          column 6; not used as the encoder sort key any more)
        - ``hit_time``: ``(B, max_L)`` float32 — truth time, encoder sort key
        - ``hit_valid``: ``(B, max_L)`` bool
    targets : dict
        - ``d0``, ``z0``, ``phi``, ``theta``, ``qop``: each ``(B,)`` float32
        - ``track_valid``: ``(B,)`` bool (all True for selected tracks)
    """
    batch_size = len(batch)
    max_len = max(item["length"] for item in batch)
    feat_dim = batch[0]["hit_features"].shape[-1]

    hit_features = torch.zeros(batch_size, max_len, feat_dim, dtype=torch.float32)
    hit_s = torch.zeros(batch_size, max_len, dtype=torch.float32)
    hit_time = torch.zeros(batch_size, max_len, dtype=torch.float32)
    hit_valid = torch.zeros(batch_size, max_len, dtype=torch.bool)

    for i, item in enumerate(batch):
        L = item["length"]
        hit_features[i, :L] = torch.from_numpy(item["hit_features"])
        hit_s[i, :L] = torch.from_numpy(item["hit_s"])
        hit_time[i, :L] = torch.from_numpy(item["hit_time"])
        hit_valid[i, :L] = True

    # Vectorise target extraction
    all_targets = torch.as_tensor(
        np.stack([item["targets"] for item in batch]),
        dtype=torch.float32,
    )  # (B, 5)

    inputs = {
        "hit_features": hit_features,
        "hit_s": hit_s,
        "hit_time": hit_time,
        "hit_valid": hit_valid,
    }

    # Compute innermost hit features for delta parameterization.
    # Innermost = smallest truth time = first hit struck on the helix
    # (replaces the previous argmin-s heuristic). Padding positions
    # (time=0, invalid) are masked to +inf so they never win the argmin.
    innermost_idx = torch.where(hit_valid, hit_time, torch.inf).argmin(dim=1)
    batch_arange = torch.arange(batch_size)
    innermost_phi = hit_features[batch_arange, innermost_idx, 4]   # phi_hit
    innermost_theta = hit_features[batch_arange, innermost_idx, 5]  # theta_hit

    target_dict = {
        "d0": all_targets[:, 0],
        "z0": all_targets[:, 1],
        "phi": all_targets[:, 2],
        "theta": all_targets[:, 3],
        "qop": all_targets[:, 4],
        "track_valid": torch.ones(batch_size, dtype=torch.bool),
        "innermost_phi": innermost_phi,
        "innermost_theta": innermost_theta,
    }

    # Pass through ACTS reco data when available
    if all("acts_reco" in item for item in batch):
        acts_reco = torch.as_tensor(
            np.stack([item["acts_reco"] for item in batch]),
            dtype=torch.float32,
        )  # (B, 5)
        target_dict["acts_reco_d0"] = acts_reco[:, 0]
        target_dict["acts_reco_z0"] = acts_reco[:, 1]
        target_dict["acts_reco_phi"] = acts_reco[:, 2]
        target_dict["acts_reco_theta"] = acts_reco[:, 3]
        target_dict["acts_reco_qop"] = acts_reco[:, 4]
        target_dict["acts_dm_mask"] = torch.tensor(
            [item["acts_dm"] for item in batch], dtype=torch.bool,
        )

    # Pass through track metadata for tight selection filtering
    if all("track_pt" in item for item in batch):
        target_dict["track_pt"] = torch.tensor(
            [item["track_pt"] for item in batch], dtype=torch.float32,
        )
        target_dict["track_vertex_primary"] = torch.tensor(
            [item["track_vertex_primary"] for item in batch], dtype=torch.float32,
        )

    return inputs, target_dict


# ============================================================================
# Packed-batch collate (Phase 2 — opt-in via ``packed_batches=True``)
# ============================================================================


def collate_tracks_packed(
    batch: list[dict[str, np.ndarray]],
) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    """Collate variable-length tracks into a packed batch.

    All hits across the batch are concatenated into a single
    ``(1, total_L, D)`` tensor; per-track segment IDs and cumulative
    segment lengths drive Mamba-2's state reset / segment-wise flip
    (SSM-CLS) and flash-attn2's ``cu_seqlens`` argument (transformer).
    No padding tokens are ever emitted, so encoder compute scales with
    the *mean* track length instead of the *max*.

    Hits within each track are loaded already in the on-disk sort order
    (truth-time). The encoder does NOT re-sort in packed mode; the
    on-disk order is the order the model sees. The SSM-CLS encoder
    consumes ``cu_seqlens`` + ``seq_idx`` directly; the transformer
    encoder (``EncoderWithCLS``) consumes the same packed inputs via
    its ``_forward_packed`` path, which interleaves CLS tokens at each
    segment start and drives the inner encoder layers with an augmented
    ``cu_seqlens`` so ``flash_attn_varlen_func`` masks attention across
    segments naturally — no padded→unpad→pack→pad round-trip.

    Returns
    -------
    inputs : dict
        - ``hit_features``: ``(1, total_L, D)`` float32
        - ``hit_s``: ``(1, total_L)`` float32 (kept for parity with the
          padded path — input feature column 6, not used as sort key)
        - ``hit_time``: ``(1, total_L)`` float32 — truth time, sort key
        - ``seq_idx``: ``(1, total_L)`` int32 — per-token track ID
        - ``cu_seqlens``: ``(B + 1,)`` int32 — cumulative segment ends
        - ``track_lengths``: ``(B,)`` int32 — convenience copy
    targets : dict
        Same keys as :func:`collate_tracks`. ``innermost_phi`` /
        ``innermost_theta`` are extracted at each track's
        argmin-``time`` position.
    """
    batch_size = len(batch)
    track_lengths = np.fromiter(
        (item["length"] for item in batch), dtype=np.int32, count=batch_size,
    )
    total_L = int(track_lengths.sum())
    feat_dim = batch[0]["hit_features"].shape[-1]

    # Concatenate hit features, per-hit s and per-hit time in a single pass.
    hit_features_np = np.empty((total_L, feat_dim), dtype=np.float32)
    hit_s_np = np.empty(total_L, dtype=np.float32)
    hit_time_np = np.empty(total_L, dtype=np.float32)
    seq_idx_np = np.empty(total_L, dtype=np.int32)
    offset = 0
    for i, item in enumerate(batch):
        L = int(item["length"])
        hit_features_np[offset:offset + L] = item["hit_features"]
        hit_s_np[offset:offset + L] = item["hit_s"]
        hit_time_np[offset:offset + L] = item["hit_time"]
        seq_idx_np[offset:offset + L] = i
        offset += L

    hit_features = torch.from_numpy(hit_features_np).unsqueeze(0)  # (1, total_L, D)
    hit_s = torch.from_numpy(hit_s_np).unsqueeze(0)                  # (1, total_L)
    hit_time = torch.from_numpy(hit_time_np).unsqueeze(0)            # (1, total_L)
    seq_idx = torch.from_numpy(seq_idx_np).unsqueeze(0)              # (1, total_L)

    cu_seqlens = torch.zeros(batch_size + 1, dtype=torch.int32)
    cu_seqlens[1:] = torch.from_numpy(np.cumsum(track_lengths, dtype=np.int32))

    # Targets — same shapes as the padded path (per-track scalars).
    all_targets = torch.as_tensor(
        np.stack([item["targets"] for item in batch]),
        dtype=torch.float32,
    )  # (B, 5)

    inputs = {
        "hit_features": hit_features,
        "hit_s": hit_s,
        "hit_time": hit_time,
        "seq_idx": seq_idx,
        "cu_seqlens": cu_seqlens,
        "track_lengths": torch.from_numpy(track_lengths),
    }

    # Innermost-hit anchor: argmin-``time`` per segment.
    innermost_phi_np = np.empty(batch_size, dtype=np.float32)
    innermost_theta_np = np.empty(batch_size, dtype=np.float32)
    offset = 0
    for i, item in enumerate(batch):
        L = int(item["length"])
        track_time = hit_time_np[offset:offset + L]
        local_argmin = int(np.argmin(track_time))
        innermost_phi_np[i] = hit_features_np[offset + local_argmin, 4]    # phi_hit
        innermost_theta_np[i] = hit_features_np[offset + local_argmin, 5]  # theta_hit
        offset += L

    target_dict = {
        "d0": all_targets[:, 0],
        "z0": all_targets[:, 1],
        "phi": all_targets[:, 2],
        "theta": all_targets[:, 3],
        "qop": all_targets[:, 4],
        "track_valid": torch.ones(batch_size, dtype=torch.bool),
        "innermost_phi": torch.from_numpy(innermost_phi_np),
        "innermost_theta": torch.from_numpy(innermost_theta_np),
    }

    if all("acts_reco" in item for item in batch):
        acts_reco = torch.as_tensor(
            np.stack([item["acts_reco"] for item in batch]),
            dtype=torch.float32,
        )
        target_dict["acts_reco_d0"] = acts_reco[:, 0]
        target_dict["acts_reco_z0"] = acts_reco[:, 1]
        target_dict["acts_reco_phi"] = acts_reco[:, 2]
        target_dict["acts_reco_theta"] = acts_reco[:, 3]
        target_dict["acts_reco_qop"] = acts_reco[:, 4]
        target_dict["acts_dm_mask"] = torch.tensor(
            [item["acts_dm"] for item in batch], dtype=torch.bool,
        )

    if all("track_pt" in item for item in batch):
        target_dict["track_pt"] = torch.tensor(
            [item["track_pt"] for item in batch], dtype=torch.float32,
        )
        target_dict["track_vertex_primary"] = torch.tensor(
            [item["track_vertex_primary"] for item in batch], dtype=torch.float32,
        )

    return inputs, target_dict


# ============================================================================
# DataModule
# ============================================================================


class ColliderMLRegrDataModule(LightningDataModule):
    """Lightning DataModule for track parameter regression.

    Shard assignments are read from a ``split.json`` file in the preprocessed
    directory (created once by ``scripts/create_split.py``).  This guarantees
    that validation and test data always come from the same shards, regardless
    of how many training shards are actually used.

    All track selection is applied at preprocessing time.  The DataModule
    uses simple random sampling with dynamic padding.

    Parameters
    ----------
    preprocessed_dir : str
        Path to preprocessed memmap shards.
    batch_size : int
        Batch size (number of tracks per batch).
    num_workers : int
        DataLoader workers.
    pin_memory : bool
        Pin memory for GPU transfer.
    num_train_shards : int
        Limit the number of *training* shards loaded (for debugging).
        ``-1`` means use all training shards from the split file.
        Validation and test shards are always loaded in full.
    """

    def __init__(
        self,
        preprocessed_dir: str = "/scratch/colliderml/arxiv_retraining/p0_core_pretrain",
        batch_size: int = 256,
        num_workers: int = 8,
        pin_memory: bool = True,
        num_train_shards: int = -1,
        load_acts: bool = False,
        streaming: bool = False,
        shard_buffer_size: int = 8,
        prefetch_factor: int | None = None,
        tail_upsample_threshold_mm: float = 0.0,
        tail_upsample_weight: float = 1.0,
        packed_batches: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.preprocessed_dir = _resolve_data_path(preprocessed_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.num_train_shards_limit = num_train_shards
        self.load_acts = load_acts
        self.streaming = streaming
        self.shard_buffer_size = shard_buffer_size
        self.prefetch_factor = prefetch_factor
        # Phase 2 opt-in: emit packed batches (1, total_L, D + seq_idx +
        # cu_seqlens) instead of padded (B, max_L, D). Supported by the
        # SSM-CLS encoder (mamba_cls.BidirectionalMambaCLSEncoder) and
        # the transformer encoder (transformer_encoder.EncoderWithCLS)
        # — both consume cu_seqlens + seq_idx natively, skipping the
        # padded→unpad→varlen→pad round-trip on every forward.
        self.packed_batches = bool(packed_batches)
        self._collate_fn = collate_tracks_packed if self.packed_batches else collate_tracks
        # Tail upsampling (d0 based): when threshold_mm > 0 and weight != 1,
        # the train DataLoader uses a _StratifiedTailSampler that draws a
        # fixed fraction of each epoch from the "tail" group (|truth d0| >
        # threshold_mm).  The effective tail fraction is derived from the
        # natural class counts and the requested weight:
        #     target_frac = w * n_tail / (w * n_tail + n_core).
        # With natural tail fraction ~5% and weight 19 this yields ~0.5.
        # Only active for the non-streaming Dataset path.
        self.tail_upsample_threshold_mm = float(tail_upsample_threshold_mm)
        self.tail_upsample_weight = float(tail_upsample_weight)
        self._tail_sampler: _StratifiedTailSampler | None = None

        self._orig_limit_train_batches: int | float | None = None
        self._train_ds: ColliderMLTrackDataset | ColliderMLStreamingDataset | None = None
        self._val_ds: ColliderMLTrackDataset | None = None
        self._test_ds: ColliderMLTrackDataset | None = None

    def _load_split(self) -> dict[str, list[int]]:
        """Load shard split from ``split.json`` in the preprocessed directory."""
        split_path = self.preprocessed_dir / "split.json"
        if not split_path.exists():
            raise FileNotFoundError(
                f"Split file not found at {split_path}. "
                "Create it with: python scripts/create_split.py "
                f"--preprocessed-dir {self.preprocessed_dir}"
            )
        with open(split_path) as f:
            data = json.load(f)
        for key in ("train", "val", "test"):
            if key not in data:
                raise ValueError(f"split.json missing required key '{key}'")
        return {k: data[k] for k in ("train", "val", "test")}

    def setup(self, stage: str | None = None) -> None:
        """Load split file and create datasets for the requested stage."""
        split = self._load_split()

        train_shards = split["train"]
        val_shards = split["val"]
        test_shards = split["test"]

        # Optionally limit training shards (for debugging)
        if self.num_train_shards_limit > 0:
            train_shards = train_shards[: self.num_train_shards_limit]

        if stage in (None, "fit"):
            # Training never uses ACTS data (metrics are precomputed/constant),
            # so skip loading it to reduce I/O and memory overhead.
            if self.streaming:
                self._train_ds = ColliderMLStreamingDataset(
                    self.preprocessed_dir,
                    train_shards,
                    load_acts=False,
                    shard_buffer_size=self.shard_buffer_size,
                )
            else:
                self._train_ds = ColliderMLTrackDataset(
                    self.preprocessed_dir, train_shards, load_acts=False,
                )
                # Build a stratified tail sampler if tail upsampling is on.
                if (
                    self.tail_upsample_threshold_mm > 0.0
                    and self.tail_upsample_weight != 1.0
                ):
                    d0_pieces: list[np.ndarray] = []
                    for si in train_shards:
                        t = np.load(
                            self.preprocessed_dir / f"shard_{si:04d}"
                            / "selected_tracks" / "track_targets.npy",
                            mmap_mode="r",
                        )
                        d0_pieces.append(np.asarray(t[:, 0], dtype=np.float32))
                    d0_train = np.concatenate(d0_pieces, axis=0)
                    is_tail = np.abs(d0_train) > self.tail_upsample_threshold_mm
                    tail_idx = np.nonzero(is_tail)[0].astype(np.int64)
                    core_idx = np.nonzero(~is_tail)[0].astype(np.int64)
                    n_tail = int(len(tail_idx))
                    n_core = int(len(core_idx))
                    eff_tail_frac = (
                        self.tail_upsample_weight * n_tail
                        / (self.tail_upsample_weight * n_tail + n_core)
                    ) if (n_tail + n_core) > 0 else 0.0
                    self._tail_sampler = _StratifiedTailSampler(
                        tail_indices=tail_idx,
                        core_indices=core_idx,
                        num_samples=len(d0_train),
                        target_tail_fraction=eff_tail_frac,
                    )
                    print(
                        f"[DataModule] stratified tail sampler active: threshold "
                        f"{self.tail_upsample_threshold_mm*1e3:.1f} um, "
                        f"weight {self.tail_upsample_weight:g} — raw tail "
                        f"fraction {n_tail/len(d0_train)*100:.2f}% → "
                        f"effective {eff_tail_frac*100:.2f}% "
                        f"({n_tail:,} tail / {n_core:,} core)"
                    )
            self._val_ds = ColliderMLTrackDataset(
                self.preprocessed_dir, val_shards, load_acts=self.load_acts,
            )
        if stage in (None, "test"):
            self._test_ds = ColliderMLTrackDataset(
                self.preprocessed_dir, test_shards, load_acts=self.load_acts,
            )
        if stage == "predict":
            self._test_ds = ColliderMLTrackDataset(
                self.preprocessed_dir, test_shards, load_acts=self.load_acts,
            )

    def train_dataloader(self) -> DataLoader:
        assert self._train_ds is not None

        if self.streaming:
            # Streaming dataset handles shuffling and DDP partitioning internally.
            # Update epoch so shard order is re-shuffled each epoch.
            # NOTE: set_epoch must be called here (before fork) rather than in
            # on_train_epoch_start because forked worker processes hold their own
            # copy of the dataset — updates after fork do not propagate.
            # reload_dataloaders_every_n_epochs=1 in the trainer config ensures
            # this method is called at the start of every epoch.
            if isinstance(self._train_ds, ColliderMLStreamingDataset):
                self._train_ds.set_epoch(self.trainer.current_epoch)

                # ---- Exact batch-count cap (single-GPU AND DDP) ----
                # With num_workers > 1 each worker drops its own drop_last
                # remainder, so the epoch yields FEWER batches than the
                # DataLoader length estimate.  Lightning schedules the
                # end-of-epoch validation at val_check_batch derived from
                # that estimate — an early-exhausting iterator means
                # validation (and every val/* metric) is silently skipped.
                # Under DDP the per-rank counts additionally differ → NCCL
                # deadlock at the epoch boundary.  Capping
                # limit_train_batches to the exact minimum-across-ranks
                # count fixes both: validation triggers via the modulo
                # check on the true last batch, and all ranks step
                # identically.  This runs inside train_dataloader() because
                # Lightning parses limit_train_batches AFTER requesting the
                # dataloader (fit_loop.setup_data), every reload.
                exact_batches = self._train_ds.batches_per_epoch(
                    self.batch_size, self.num_workers
                )
                # Remember the ORIGINAL user-configured limit on first call —
                # we overwrite trainer.limit_train_batches every epoch, so
                # reading it back later would compare against our own cap.
                if self._orig_limit_train_batches is None:
                    self._orig_limit_train_batches = self.trainer.limit_train_batches
                user_limit = self._orig_limit_train_batches
                if isinstance(user_limit, int):
                    # Respect an explicit user cap (e.g. debug runs)
                    exact_batches = min(user_limit, exact_batches)
                elif isinstance(user_limit, float) and user_limit != 1.0:
                    exact_batches = max(1, int(exact_batches * user_limit))
                self.trainer.limit_train_batches = exact_batches

            dl_kwargs: dict = dict(
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                collate_fn=self._collate_fn,
                drop_last=True,
                # persistent_workers=False for streaming: reload_dataloaders_every_n_epochs=1
                # creates a new DataLoader each epoch anyway, so persistence adds no
                # benefit and risks stale worker state from the previous epoch.
                persistent_workers=False,
            )
            if self.prefetch_factor is not None and self.num_workers > 0:
                dl_kwargs["prefetch_factor"] = self.prefetch_factor
            return DataLoader(self._train_ds, **dl_kwargs)

        dl_kwargs = dict(
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self._collate_fn,
            drop_last=True,
            persistent_workers=self.num_workers > 0,
        )
        if self._tail_sampler is not None:
            dl_kwargs["sampler"] = self._tail_sampler
            dl_kwargs["shuffle"] = False  # cannot combine shuffle + sampler
        else:
            dl_kwargs["shuffle"] = True
        if self.prefetch_factor is not None and self.num_workers > 0:
            dl_kwargs["prefetch_factor"] = self.prefetch_factor
        return DataLoader(self._train_ds, **dl_kwargs)

    def val_dataloader(self) -> DataLoader:
        assert self._val_ds is not None
        dl_kwargs: dict = dict(
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self._collate_fn,
            # persistent_workers=False: avoids the Lightning warning about
            # pin_memory=True + persistent_workers=True + reload_dataloaders_every_n_epochs > 0
            # (pytorch/pytorch#91252 — potential deadlock on DataLoader recreation).
            persistent_workers=False,
        )
        if self.prefetch_factor is not None and self.num_workers > 0:
            dl_kwargs["prefetch_factor"] = self.prefetch_factor
        return DataLoader(self._val_ds, **dl_kwargs)

    def test_dataloader(self) -> DataLoader:
        assert self._test_ds is not None
        dl_kwargs: dict = dict(
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self._collate_fn,
        )
        if self.prefetch_factor is not None and self.num_workers > 0:
            dl_kwargs["prefetch_factor"] = self.prefetch_factor
        return DataLoader(self._test_ds, **dl_kwargs)

    def predict_dataloader(self) -> DataLoader:
        return self.test_dataloader()
