"""Custom callbacks for the colliderml_regr experiment."""

from __future__ import annotations

from pathlib import Path
import subprocess

import h5py
import numpy as np
from lightning import Callback, LightningModule, Trainer
from lightning.pytorch.utilities import rank_zero_info
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
        from track_regression.losses import (
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
        # NVML cannot query memory on MIG slices the way the caching allocator
        # expects, raising "NVML_SUCCESS == r INTERNAL ASSERT FAILED" on the
        # first call. Fall through to NaN so the metric is just absent in
        # Comet rather than killing the run.
        try:
            free_b, total_b = torch.cuda.mem_get_info(device_idx)
        except RuntimeError:
            return float("nan")
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



class KernelSwapCallback(Callback):
    """Swap the encoder onto a short-sequence kernel variant before training.

    Campaign hook (docs/perf/OPTIMIZATION_LOG.md): applies
    ``track_regression.mamba_short.apply_variant(pl_module.model, variant)``
    in ``setup`` — i.e. after the checkpoint warm-start machinery built the
    module but before ``configure_optimizers`` collects parameters, so
    optimizer param groups reference the swapped modules. The math is an
    algebraically identical re-expression of the Mamba2 update (oracle chain
    in tests/test_mamba2short.py); training dynamics are unchanged up to
    floating-point noise.

    Parameters
    ----------
    variant : str
        One of the campaign variants, e.g. ``v3`` (eager quadratic dual) or
        ``v3c`` (+torch.compile of the static core).
    """

    def __init__(self, variant: str = "v3c") -> None:
        self.variant = variant

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        from track_regression.mamba_short import apply_variant

        apply_variant(pl_module.model, self.variant)
        # apply_variant builds fresh Mamba2Short modules under the default dtype
        # (float32).  Under trainer.precision 64-true Lightning has already
        # converted the module to float64, so re-align the swapped layers with
        # the LightningModule's dtype (no-op for the default fp32 setting).
        pl_module.model.to(dtype=pl_module.dtype)
        rank_zero_info(
            f"[KernelSwapCallback] applied variant {self.variant!r} to "
            f"{type(pl_module.model.encoder).__name__}"
        )
