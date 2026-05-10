#!/usr/bin/env python3
"""Train track parameter regression model.

Usage::

    python train.py fit --config config/study1_baseline_l1.yaml

    # Override data dir for testing:
    python train.py fit --config config/study1_baseline_l1.yaml \\
        --data.preprocessed_dir ${DATA_ROOT}/p0_preprocessed_test \\
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

torch.set_float32_matmul_precision("high")

from pathlib import Path

from lightning.pytorch.cli import LightningCLI

from hepattn.experiments.colliderml_regr.data import ColliderMLRegrDataModule
from hepattn.experiments.colliderml_regr.model import TrackRegressionWrapper


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
