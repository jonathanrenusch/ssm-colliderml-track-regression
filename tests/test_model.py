"""Paper configs build, the deployed GPU-seed path matches the training path, and training runs."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import yaml

from track_regression.flat_data import FlatTrackDataset, FlatTrackStore
from track_regression.inference import _instantiate

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = sorted((ROOT / "configs").glob("*.yaml"))
CUDA = torch.cuda.is_available()
P = ("d0", "z0", "phi", "theta", "qop")


def _model(cfg_path):
    torch.manual_seed(0)
    return _instantiate(yaml.safe_load(open(cfg_path))["model"]["model"])


@pytest.mark.parametrize("cfg", CONFIGS, ids=lambda p: p.stem)
def test_config_builds(cfg):
    m = _model(cfg)
    n = sum(p.numel() for p in m.parameters())
    assert 0.60e6 < n < 0.70e6, n                   # all encoders are parameter-matched
    assert m.loss_module.total_outputs == 35         # 5 parameters x 7 quantiles


@pytest.mark.skipif(not CUDA, reason="CUDA")
def test_gpu_seed_path_matches_training_inputs(synthetic_store):
    """Inference batches carry 12 features and the forward computes the seed on the
    GPU; training batches carry the 15 features and the seed from the data loader."""
    os.environ["TRK_REFERENCE_KERNELS"] = "1"
    try:
        m = _model(ROOT / "configs" / "minGRU_stage1.yaml").cuda().eval()
        store = FlatTrackStore(synthetic_store / "test")
        idx = list(range(100))
        with torch.inference_mode():
            outs = []
            for srf in (True, False):
                inp, tgt = FlatTrackDataset(store, seed_residual_features=srf).__getitems__(idx)
                o = m({k: v.cuda() for k, v in inp.items()})
                anchors = ({f"seed_{p}": o["seed"][:, i] for i, p in enumerate(P)} if "seed" in o
                           else {k: v.cuda() for k, v in tgt.items()})
                outs.append(m.loss_module.predict_physical(o["pred"], anchors))
        assert "seed" not in outs[0]
        for p in P:
            assert torch.allclose(outs[0][p], outs[1][p], rtol=1e-4, atol=1e-5), p
    finally:
        os.environ.pop("TRK_REFERENCE_KERNELS", None)


@pytest.mark.skipif(not CUDA, reason="CUDA")
def test_training_runs(synthetic_store, tmp_path):
    cmd = [sys.executable, "-m", "track_regression.train", "fit",
           "--config", str(ROOT / "configs" / "minGRU_stage1.yaml"),
           "--data.preprocessed_dir", str(synthetic_store), "--data.batch_size", "32",
           "--data.num_workers", "0", "--trainer.max_epochs", "1", "--trainer.limit_train_batches", "4",
           "--trainer.limit_val_batches", "2", "--trainer.logger.init_args.save_dir", str(tmp_path)]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=tmp_path)
    assert r.returncode == 0, r.stderr[-3000:]
    assert list(tmp_path.glob("minGRU_stage1/version_0/checkpoints/last.ckpt"))
