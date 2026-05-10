"""Custom callbacks for the colliderml_regr experiment."""

from __future__ import annotations

from pathlib import Path
import subprocess

import h5py
import numpy as np
from lightning import Callback, LightningModule, Trainer
import torch
from torch import Tensor


class RegressionPredictionWriter(Callback):
    """Write track regression predictions and targets to an HDF5 file.

    Activates only during the ``test`` stage. Each batch appends into
    growable 1-D datasets so the file can be read as flat arrays by a
    downstream evaluation script.

    HDF5 layout::

        preds/{d0, z0, phi, theta, qop}      — (N,) float32
        targets/{d0, z0, phi, theta, qop}     — (N,) float32
        quantiles/{d0, z0, phi, theta, qop}   — (N, Q) float32  (if quantile loss)
            attrs: levels = [tau_1, ..., tau_Q]
    """

    def __init__(self) -> None:
        super().__init__()
        self.file: h5py.File | None = None
        self._output_path: Path | None = None
        self._quantile_levels: dict[str, np.ndarray] = {}

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        if stage != "test":
            return

        ckpt_path = trainer.ckpt_path
        if not ckpt_path:
            import sys
            import argparse
            # Fallback to sys.argv when trainer.ckpt_path is None (e.g. during CLI test initialization)
            for i, arg in enumerate(sys.argv):
                if arg == "--ckpt_path" and i + 1 < len(sys.argv):
                    ckpt_path = sys.argv[i + 1]
                    break
                elif arg.startswith("--ckpt_path="):
                    ckpt_path = arg.split("=", 1)[1]
                    break

        if ckpt_path:
            # logs/.../experiment_id/ckpts/epoch=X.ckpt -> logs/.../experiment_id
            log_dir = Path(ckpt_path).parent.parent
            ckpt_name = Path(ckpt_path).stem
            self._output_path = log_dir / f"{ckpt_name}__test_predictions.h5"
        else:
            log_dir = Path(trainer.log_dir)
            self._output_path = log_dir / "test_predictions.h5"
        
        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        self.file = h5py.File(self._output_path, "w")

        # Create resizable datasets (unknown total size)
        self.file.create_group("preds")
        self.file.create_group("targets")

        # Discover which parameters have quantile losses and store levels
        from hepattn.experiments.colliderml_regr.losses import (
            EtaQuantileLoss,
            QuantileLoss,
        )

        loss_module = pl_module.model.loss_module
        has_quantiles = False
        for name in loss_module.parameter_order:
            loss_fn = loss_module.losses[name]
            if isinstance(loss_fn, (QuantileLoss, EtaQuantileLoss)):
                self._quantile_levels[name] = loss_fn.quantiles.cpu().numpy()
                has_quantiles = True

        if has_quantiles:
            self.file.create_group("quantiles")

    def _append_1d(self, grp: h5py.Group, name: str, arr: np.ndarray) -> None:
        """Append a 1-D array to a resizable dataset, creating it on first call."""
        if name not in grp:
            grp.create_dataset(
                name,
                data=arr,
                maxshape=(None,),
                chunks=True,
                compression="gzip",
                compression_opts=1,
            )
        else:
            ds = grp[name]
            old_len = ds.shape[0]
            ds.resize(old_len + len(arr), axis=0)
            ds[old_len:] = arr

    def _append_2d(self, grp: h5py.Group, name: str, arr: np.ndarray) -> None:
        """Append a 2-D array (N, Q) to a resizable dataset, creating it on first call."""
        if name not in grp:
            ds = grp.create_dataset(
                name,
                data=arr,
                maxshape=(None, arr.shape[1]),
                chunks=True,
                compression="gzip",
                compression_opts=1,
            )
            # Store quantile levels as attribute
            if name in self._quantile_levels:
                ds.attrs["levels"] = self._quantile_levels[name]
        else:
            ds = grp[name]
            old_len = ds.shape[0]
            ds.resize(old_len + arr.shape[0], axis=0)
            ds[old_len:] = arr

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self.file is None:
            return

        preds: dict[str, Tensor] = outputs["preds"]
        targets: dict[str, Tensor] = outputs["targets"]
        quantile_preds: dict[str, Tensor] | None = outputs.get("quantile_preds")

        for name in ["d0", "z0", "phi", "theta", "qop"]:
            # Always write the truth target when available (useful even for
            # params the model doesn't predict — downstream analyses may need
            # theta for eta binning etc.).  Only write preds for params this
            # model actually emits.
            if name in targets:
                t = targets[name].detach().float().cpu().numpy().ravel()
                self._append_1d(self.file["targets"], name, t)
            if name not in preds:
                continue
            p = preds[name].detach().float().cpu().numpy().ravel()
            self._append_1d(self.file["preds"], name, p)

            # Write quantile predictions if available and multi-dimensional
            if (
                quantile_preds is not None
                and name in quantile_preds
                and name in self._quantile_levels
            ):
                q = quantile_preds[name].detach().float().cpu().numpy()
                if q.ndim == 2:
                    self._append_2d(self.file["quantiles"], name, q)

    def teardown(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        if stage != "test":
            return
        if self.file is not None:
            self.file.close()
            self.file = None
        if self._output_path is not None:
            print("-" * 80)
            print(f"Predictions written to {self._output_path}")
            if self._quantile_levels:
                params = ", ".join(self._quantile_levels.keys())
                print(f"Quantile predictions included for: {params}")
            print("-" * 80)




class MinimalGpuMonitor(Callback):
    """Log only coarse GPU utilization and memory utilization metrics.

    Metrics are designed to mirror a compact subset of ``nvidia-smi``:
    - ``gpu/utilization_pct``
    - ``gpu/memory_utilization_pct``
    """

    def __init__(self, log_every_n_steps: int = 50) -> None:
        super().__init__()
        self.log_every_n_steps = log_every_n_steps
        self._sync_dist = False

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        if trainer.fast_dev_run or stage != "fit":
            return
        self._sync_dist = len(trainer.device_ids) > 1

    @staticmethod
    def _query_nvidia_smi(device_idx: int) -> tuple[float, float] | None:
        """Return ``(gpu_util_pct, mem_util_pct)`` from nvidia-smi if available."""
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                    "-i",
                    str(device_idx),
                ],
                capture_output=True,
                text=True,
                check=True,
                timeout=2,
            )
            line = result.stdout.strip().splitlines()[0]
            gpu_util_str, mem_used_str, mem_total_str = [x.strip() for x in line.split(",")]
            gpu_util = float(gpu_util_str)
            mem_used = float(mem_used_str)
            mem_total = float(mem_total_str)
            mem_util = 100.0 * mem_used / max(mem_total, 1.0)
            return gpu_util, mem_util
        except (IndexError, ValueError, subprocess.SubprocessError):
            return None

    @staticmethod
    def _torch_memory_util(device_idx: int) -> float:
        free_b, total_b = torch.cuda.mem_get_info(device_idx)
        used_b = total_b - free_b
        return 100.0 * float(used_b) / max(float(total_b), 1.0)

    def on_train_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs, batch, batch_idx) -> None:
        if not torch.cuda.is_available() or self.log_every_n_steps <= 0:
            return
        if trainer.global_step % self.log_every_n_steps != 0:
            return

        device_idx = torch.cuda.current_device()
        stats = self._query_nvidia_smi(device_idx)

        if stats is not None:
            gpu_util, mem_util = stats
            pl_module.log(
                "gpu/utilization_pct",
                gpu_util,
                on_step=True,
                on_epoch=False,
                logger=True,
                sync_dist=self._sync_dist,
            )
        else:
            mem_util = self._torch_memory_util(device_idx)

        pl_module.log(
            "gpu/memory_utilization_pct",
            mem_util,
            on_step=True,
            on_epoch=False,
            logger=True,
            sync_dist=self._sync_dist,
        )


class GradientSpikeSkip(Callback):
    """Zero gradients when the pre-clip grad norm exceeds ``k`` x running median.

    Guards against rare bad-batch spikes that would otherwise blow up optimizer
    momentum — a particular hazard for Lion, whose sign-based update ignores
    gradient magnitude (so ``gradient_clip_val`` alone does *not* protect it).

    The check runs in ``on_after_backward`` (before Lightning's grad clipping),
    maintains a rolling window of recent L2 norms, and when a spike is
    detected, zeros every parameter's ``.grad`` so the subsequent
    ``optimizer.step()`` moves only under stale momentum (AdamW: exp-decayed;
    Lion: sign of decayed momentum). Warm-up period prevents false positives
    from the first few erratic steps.
    """

    def __init__(
        self,
        k: float = 5.0,
        warmup_steps: int = 200,
        buffer_size: int = 500,
    ) -> None:
        super().__init__()
        self.k = float(k)
        self.warmup_steps = int(warmup_steps)
        self.buffer_size = int(buffer_size)
        self._norms: list[float] = []
        self._skip_count = 0
        self._sync_dist = False

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        if trainer.fast_dev_run or stage != "fit":
            return
        self._sync_dist = len(trainer.device_ids) > 1

    def on_after_backward(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if trainer.fast_dev_run:
            return

        total_sq = torch.zeros((), device=pl_module.device)
        for p in pl_module.parameters():
            if p.grad is not None:
                total_sq = total_sq + p.grad.detach().pow(2).sum()
        grad_norm = float(total_sq.sqrt().item())

        skipped = 0.0
        if len(self._norms) >= self.warmup_steps:
            sorted_norms = sorted(self._norms)
            median = sorted_norms[len(sorted_norms) // 2]
            if median > 0.0 and grad_norm > self.k * median:
                for p in pl_module.parameters():
                    if p.grad is not None:
                        p.grad.zero_()
                self._skip_count += 1
                skipped = 1.0

        if skipped == 0.0:
            self._norms.append(grad_norm)
            if len(self._norms) > self.buffer_size:
                self._norms.pop(0)

        pl_module.log(
            "grad/skipped",
            skipped,
            on_step=True,
            on_epoch=False,
            logger=True,
            sync_dist=self._sync_dist,
        )
        pl_module.log(
            "grad/skip_count_total",
            float(self._skip_count),
            on_step=True,
            on_epoch=False,
            logger=True,
            sync_dist=self._sync_dist,
        )


class OffsetWarmupCallback(Callback):
    """Ramp ``lambda_offset`` on every :class:`BinnedDFLQuantileOffsetLoss`
    from ``start_factor`` up to the target value over the first
    ``warmup_fraction`` of training steps.

    Defence-in-depth for the scaled YOLO pretrain: during the first few
    hundred steps the classification softmax is near-uniform, so the
    QFL-coupled pinball loss drives the offset head against a very noisy
    target (each bin's "truth offset" is weighted by 1/K).  Starting
    ``lambda_offset`` near zero and ramping up lets the classifier sharpen
    before the offset head carries any weight — a standard GFL-style
    guard that is free when the loss is stable anyway (the warmup is a
    no-op after the first ``warmup_fraction`` of steps).

    Not needed on the smoke config — empirically verified to train cleanly
    with ``lambda_offset = 1.0`` from step 0.  Recommended on the scaled
    pretrain / finetune where a NaN recovery cost is high.

    Parameters
    ----------
    warmup_fraction : float
        Fraction of total training steps over which to ramp.  Default 0.05.
    start_factor : float
        Initial multiplier applied to each loss's configured
        ``lambda_offset``.  Default 0.0 (offset head inactive at step 0).

    Notes
    -----
    Mutates ``sub_loss.lambda_offset`` in-place each step.  The original
    target value is captured at ``on_train_start`` and restored when the
    ramp completes.  Safe under DDP — the same scalar is applied on every
    rank because ``trainer.global_step`` is synchronised.
    """

    def __init__(self, warmup_fraction: float = 0.05, start_factor: float = 0.0):
        super().__init__()
        if not (0.0 <= warmup_fraction <= 1.0):
            raise ValueError(f"warmup_fraction must be in [0, 1], got {warmup_fraction}")
        if not (0.0 <= start_factor <= 1.0):
            raise ValueError(f"start_factor must be in [0, 1], got {start_factor}")
        self.warmup_fraction = float(warmup_fraction)
        self.start_factor = float(start_factor)
        self._targets: dict[str, float] = {}
        self._total_warmup_steps: int | None = None

    def _iter_binned_losses(self, pl_module: LightningModule):
        """Yield (name, sub_loss) for every BinnedDFLQuantileOffsetLoss in the model."""
        from hepattn.experiments.colliderml_regr.losses import BinnedDFLQuantileOffsetLoss

        loss_module = getattr(pl_module.model, "loss_module", None)
        if loss_module is None:
            return
        for name, sub in loss_module.losses.items():
            if isinstance(sub, BinnedDFLQuantileOffsetLoss):
                yield name, sub

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        total_steps = int(trainer.estimated_stepping_batches or 0)
        self._total_warmup_steps = max(1, int(self.warmup_fraction * total_steps))
        self._targets = {
            name: float(sub.lambda_offset)
            for name, sub in self._iter_binned_losses(pl_module)
        }
        if not self._targets:
            return
        if trainer.is_global_zero:
            print(
                f"[OffsetWarmupCallback] ramping lambda_offset "
                f"from start_factor={self.start_factor} to target over "
                f"{self._total_warmup_steps}/{total_steps} steps "
                f"({self.warmup_fraction:.0%}) for "
                f"{sorted(self._targets)}"
            )

    def on_train_batch_start(
        self, trainer: Trainer, pl_module: LightningModule, batch, batch_idx
    ) -> None:
        if not self._targets or self._total_warmup_steps is None:
            return
        step = int(trainer.global_step)
        if step >= self._total_warmup_steps:
            # Ensure we've landed exactly on the target value — protects
            # against float drift from the step-by-step interpolation.
            for name, sub in self._iter_binned_losses(pl_module):
                if name in self._targets:
                    sub.lambda_offset = self._targets[name]
            return
        # Linear ramp from start_factor → 1.0 over warmup_steps.
        frac = step / self._total_warmup_steps
        factor = self.start_factor + (1.0 - self.start_factor) * frac
        for name, sub in self._iter_binned_losses(pl_module):
            if name in self._targets:
                sub.lambda_offset = self._targets[name] * factor
        # Log the current scale once per ramp step for transparency.
        if trainer.is_global_zero and step % 50 == 0:
            pl_module.log("train/lambda_offset_scale", factor,
                          on_step=True, on_epoch=False, sync_dist=False)
