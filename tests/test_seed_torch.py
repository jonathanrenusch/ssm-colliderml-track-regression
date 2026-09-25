"""Parity of the torch (GPU) seed front end with the numpy reference on real hits.

Set ``TRACKFIT_TEST_STORE`` to a part directory of a preprocessed store
(e.g. ``data/stores/single_muon_uniform/test/part_0000``) to run it."""
import os
from pathlib import Path
import numpy as np
import pytest

torch = pytest.importorskip("torch")
from track_regression.seed import seed_from_csr, seed_residuals, compress_residuals
from track_regression.seed_torch import gpu_seed_features

STORE = Path(os.environ.get("TRACKFIT_TEST_STORE", "/nonexistent"))


@pytest.mark.skipif(not STORE.exists(), reason="set TRACKFIT_TEST_STORE")
@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def test_gpu_seed_features_match_numpy(device):
    hits = np.load(STORE / "hits.npy", mmap_mode="r"); off = np.load(STORE / "offsets.npy"); ln = np.load(STORE / "lengths.npy")
    n = 20000; a, b = int(off[0]), int(off[n - 1] + ln[n - 1])
    H = np.asarray(hits[a:b]); lens = ln[:n]
    seed_np = seed_from_csr(H, lens)
    row = np.repeat(np.arange(n), lens)
    res_np = compress_residuals(seed_residuals(H[:, :3], seed_np, row))
    cu = torch.from_numpy(np.r_[0, np.cumsum(lens)].astype(np.int32)).to(device)
    seed_t, res_t = gpu_seed_features(torch.from_numpy(H).to(device), cu, dtype=torch.float64)
    seed_t = seed_t.double().cpu().numpy(); res_t = res_t.double().cpu().numpy()
    dphi = np.angle(np.exp(1j * (seed_t[:, 2] - seed_np[:, 2])))
    assert np.abs(seed_t[:, 0] - seed_np[:, 0]).max() < 1e-3          # d0 [mm]  (float32 in/out)
    assert np.abs(seed_t[:, 1] - seed_np[:, 1]).max() < 1e-2          # z0 [mm]
    assert np.abs(dphi).max() < 1e-5
    assert np.abs(seed_t[:, 3] - seed_np[:, 3]).max() < 1e-5
    assert np.abs(seed_t[:, 4] - seed_np[:, 4]).max() < 1e-4 * (1 + np.abs(seed_np[:, 4]).max())
    assert np.abs(res_t - res_np).max() < 2e-3, np.abs(res_t - res_np).max()
