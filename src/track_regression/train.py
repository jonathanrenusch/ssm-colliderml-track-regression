"""Training entry point (Lightning CLI).

    python -m track_regression.train fit --config configs/minGRU_stage1.yaml

Training runs in strict IEEE fp32 (no TF32 in the GEMMs); set
``TRK_MATMUL_PRECISION=high`` to allow TF32.
"""

import os
import warnings

import torch
from lightning.pytorch.cli import LightningCLI

from track_regression.data import ColliderMLRegrDataModule
from track_regression.model import TrackRegressionWrapper


def main():
    precision = os.environ.get("TRK_MATMUL_PRECISION", "highest")
    torch.set_float32_matmul_precision(precision)
    if precision == "highest":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        warnings.filterwarnings("ignore", message=".*TensorFloat32 tensor cores.*")
    LightningCLI(
        model_class=TrackRegressionWrapper,
        datamodule_class=ColliderMLRegrDataModule,
        seed_everything_default=42,
        save_config_kwargs={"overwrite": True},
    )


if __name__ == "__main__":
    main()
