#!/usr/bin/env python3
"""Train track parameter regression model.

Usage::

    python train.py fit --config config/study1_baseline_l1.yaml

    # Override data dir for testing:
    python train.py fit --config config/study1_baseline_l1.yaml \\
        --data.preprocessed_dir /scratch/colliderml/arxiv_retraining/p0_preprocessed_test \\
        --data.num_train_shards 2 --trainer.max_epochs 2
"""

import os
import warnings

# Redirect Triton cache to /tmp to avoid AFS quota issues.
# Must be set before any torch/triton import.
os.environ.setdefault("TRITON_CACHE_DIR", "/tmp/triton_cache")

# Suppress noisy PyTorch DDP barrier warning (harmless, no explicit device_id)
warnings.filterwarnings("ignore", message=".*barrier.*device under current context.*")

import torch

# fp32 matmul precision. DEFAULT: "high" = TF32 tensor-core linears (10 mantissa
# bits inside the nn.Linear GEMMs; the scan/conv/norm are always full IEEE fp32
# in the kernel). This is the setting the published checkpoints were trained and
# validated under — it recovered the physics goals (physics gate ≤0.2% drift) and
# is the FAST option for training. Set TRK_MATMUL_PRECISION=highest for full IEEE
# fp32 in the linears too (measured inference cost: ~free on 4L, ~15% on 10L;
# larger for training) — stricter, but not required for the physics targets.
import os as _os

_MATMUL_PRECISION = _os.environ.get("TRK_MATMUL_PRECISION", "high")
torch.set_float32_matmul_precision(_MATMUL_PRECISION)
if _MATMUL_PRECISION == "highest":
    # Also forbid TF32 in cuBLAS matmul and cuDNN conv (separate switches), so
    # full fp32 holds across the whole graph incl. any cuDNN/F.conv1d fallback.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
print(f"[train] float32_matmul_precision = {_MATMUL_PRECISION!r}"
      + (" (TF32 GEMMs!)" if _MATMUL_PRECISION == "high" else " (full IEEE fp32)"),
      flush=True)

from pathlib import Path

from lightning.pytorch.cli import LightningCLI

from track_regression.data import ColliderMLRegrDataModule
from track_regression.model import TrackRegressionWrapper


class CLI(LightningCLI):
    """CLI with implicit base config loading."""

    def add_arguments_to_parser(self, parser):
        import sys
        
        # Determine the base config dynamically based on the provided config file
        default_config = Path(__file__).parent / "config" / "base.yaml"
        for i, arg in enumerate(sys.argv):
            if arg in ("-c", "--config") and i + 1 < len(sys.argv):
                config_path = Path(sys.argv[i + 1])
                if (config_path.parent / "base.yaml").exists():
                    default_config = config_path.parent / "base.yaml"
                break

        parser.default_config_files = [str(default_config)]


def main():
    CLI(
        model_class=TrackRegressionWrapper,
        datamodule_class=ColliderMLRegrDataModule,
        seed_everything_default=42,
        save_config_kwargs={"overwrite": True},
    )


if __name__ == "__main__":
    main()
