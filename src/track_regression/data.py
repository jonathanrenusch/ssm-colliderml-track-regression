"""Lightning DataModule over a flat track store (see :mod:`track_regression.flat_data`)."""

from __future__ import annotations

import os
from pathlib import Path

import torch.distributed as dist
from lightning import LightningDataModule
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from track_regression.flat_data import (
    BlockBatchSampler,
    FlatBlockTrackDataset,
    FlatTrackDataset,
    FlatTrackStore,
)


def _identity(x):
    """Pass-through collate: the flat datasets return an already-collated batch."""
    return x


class ColliderMLRegrDataModule(LightningDataModule):
    """DataModule for track parameter regression on a flat store.

    Parameters
    ----------
    preprocessed_dir : str
        Store root holding ``train/``, ``val/`` and ``test/`` (each with a
        ``manifest.json``).
    batch_size : int
        Tracks per batch.
    num_workers, pin_memory, prefetch_factor
        Passed to the DataLoaders.
    seed_residual_features : bool
        Append the three per-hit residuals to the analytic seed helix as hit
        features 12-14 (the collate computes the seed on the CPU).  ``True`` for
        training; evaluation passes ``False`` so the model computes seed and
        residuals on the GPU inside its forward, as in deployment.
    max_val_tracks : int | None
        Cap the validation split (spread over all parts) to keep per-epoch
        validation cheap.
    """

    def __init__(
        self,
        preprocessed_dir: str,
        batch_size: int = 2048,
        num_workers: int = 8,
        pin_memory: bool = True,
        prefetch_factor: int | None = None,
        seed_residual_features: bool = False,
        max_val_tracks: int | None = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.preprocessed_dir = Path(os.path.expanduser(str(preprocessed_dir)))
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.prefetch_factor = prefetch_factor
        self.seed_residual_features = bool(seed_residual_features)
        self.max_val_tracks = max_val_tracks
        self._train_ds = None
        self._val_ds = None
        self._test_ds = None
        self._train_sampler: BlockBatchSampler | None = None

    def setup(self, stage: str | None = None) -> None:
        srf = self.seed_residual_features
        if stage in (None, "fit"):
            self._train_ds = FlatBlockTrackDataset(
                FlatTrackStore(self.preprocessed_dir / "train"), seed_residual_features=srf)
            print(f"[DataModule] train: {self._train_ds.store.n:,} tracks")
            self._train_sampler = BlockBatchSampler(len(self._train_ds.store), self.batch_size, seed=42)
            self._val_ds = FlatTrackDataset(
                FlatTrackStore(self.preprocessed_dir / "val", max_tracks=self.max_val_tracks),
                seed_residual_features=srf)
            print(f"[DataModule] val: {self._val_ds.store.n:,} tracks")
        if stage in (None, "test", "predict"):
            self._test_ds = FlatTrackDataset(
                FlatTrackStore(self.preprocessed_dir / "test"), seed_residual_features=srf)
            print(f"[DataModule] test: {self._test_ds.store.n:,} tracks")

    def _assert_no_double_sharding(self) -> None:
        """``BlockBatchSampler`` shards by rank itself, so Lightning must not
        wrap it in a ``DistributedSamplerWrapper`` and shard it a second time
        (each rank would then see 1/world_size of its batches)."""
        if self.trainer is None or (getattr(self.trainer, "world_size", 1) or 1) <= 1:
            return
        if getattr(self.trainer._accelerator_connector, "use_distributed_sampler", False):
            raise RuntimeError(
                "Flat block sampling requires `trainer.use_distributed_sampler: false`: "
                "BlockBatchSampler already partitions the blocks across ranks."
            )

    def _loader_kwargs(self) -> dict:
        kw: dict = dict(num_workers=self.num_workers, pin_memory=self.pin_memory,
                        persistent_workers=False)
        if self.prefetch_factor is not None and self.num_workers > 0:
            kw["prefetch_factor"] = self.prefetch_factor
        return kw

    def train_dataloader(self) -> DataLoader:
        self._assert_no_double_sharding()
        self._train_sampler.set_epoch(self.trainer.current_epoch if self.trainer else 0)
        return DataLoader(self._train_ds, batch_size=None, sampler=self._train_sampler,
                          **self._loader_kwargs())

    def _eval_dataloader(self, ds: FlatTrackDataset, shard: bool) -> DataLoader:
        kw = dict(batch_size=self.batch_size, shuffle=False, drop_last=False,
                  collate_fn=_identity, **self._loader_kwargs())
        # use_distributed_sampler is off, so the validation loader shards itself
        # under DDP (otherwise every rank would evaluate the whole split).
        if shard and dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            kw["sampler"] = DistributedSampler(ds, shuffle=False, drop_last=False)
            kw.pop("shuffle")
        return DataLoader(ds, **kw)

    def val_dataloader(self) -> DataLoader:
        return self._eval_dataloader(self._val_ds, shard=True)

    def test_dataloader(self) -> DataLoader:
        return self._eval_dataloader(self._test_ds, shard=False)

    def predict_dataloader(self) -> DataLoader:
        return self.test_dataloader()
