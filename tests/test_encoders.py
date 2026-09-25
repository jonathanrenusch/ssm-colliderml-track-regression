"""The four sequence encoders of the paper: fused inference path vs training path,
track independence inside a packed batch, order sensitivity, gradients."""

from __future__ import annotations

import os

import pytest
import torch

from track_regression.mamba import BidirectionalMambaCLSEncoder, Mamba2Short, mamba2_block_ref
from track_regression.mingru import DiagRNNCLSEncoder, MinGRUCLSEncoder, mingru_scan, mingru_scan_ref
from track_regression.transformer import TransformerCLSEncoder

CUDA = torch.cuda.is_available()
DIM = 128
ENCODERS = {
    "minGRU": lambda: MinGRUCLSEncoder(dim=DIM, hidden_size=192, num_layers=2, pool_out_dim=256),
    "mamba2": lambda: BidirectionalMambaCLSEncoder(num_layers=2, dim=DIM, d_state=64, d_conv=1, expand=2, headdim=32),
    "transformer": lambda: TransformerCLSEncoder(dim=DIM, num_layers=3, num_heads=4, hidden_dim_scale=3,
                                                 layer_scale=1e-5, num_cls_tokens=2),
    "diagssm": lambda: DiagRNNCLSEncoder(dim=DIM, hidden_size=270, num_layers=2, pool_out_dim=256),
}
LENS = [7, 12, 20, 6, 13, 20, 9, 18, 6, 15]


def _build(name):
    torch.manual_seed(0)
    enc = ENCODERS[name]()
    if name == "transformer":                   # LayerScale 1e-5 would hide the layers
        with torch.no_grad():
            for layer in enc.encoder.layers:
                layer.attn.ls.gamma.fill_(0.5)
                layer.dense.ls.gamma.fill_(0.5)
    return enc.cuda()


def _batch(lens=LENS, seed=1):
    g = torch.Generator().manual_seed(seed)
    cu = torch.zeros(len(lens) + 1, dtype=torch.int32)
    cu[1:] = torch.cumsum(torch.tensor(lens), 0)
    x = torch.randn(1, int(cu[-1]), DIM, generator=g)
    seq_idx = torch.repeat_interleave(torch.arange(len(lens), dtype=torch.int32), torch.tensor(lens))[None]
    return x.cuda(), cu.cuda(), seq_idx.cuda()


def _pooled(enc, x, cu, seq_idx, reference: bool, dtype=None):
    os.environ["TRK_REFERENCE_KERNELS"] = "1" if reference else "0"
    try:
        with torch.inference_mode(), torch.autocast("cuda", dtype=dtype or torch.float16, enabled=dtype is not None):
            return enc.eval()(x, cu_seqlens=cu, seq_idx=seq_idx)[1].float()
    finally:
        os.environ.pop("TRK_REFERENCE_KERNELS", None)


@pytest.mark.skipif(not CUDA, reason="CUDA kernels")
@pytest.mark.parametrize("name", ["minGRU", "mamba2", "transformer"])
def test_fused_path_matches_training_path(name):
    torch.backends.cuda.matmul.allow_tf32 = False
    enc = _build(name)
    x, cu, seq_idx = _batch()
    ref = _pooled(enc, x, cu, seq_idx, reference=True)
    fused = _pooled(enc, x, cu, seq_idx, reference=False)
    assert ref.shape == fused.shape == (len(LENS), enc.pool_dim)
    assert (fused - ref).abs().max().item() / ref.abs().max().item() < 1e-4
    # fp16 inference stays within fp16 distance of the fp32 result
    f16 = _pooled(enc, x, cu, seq_idx, reference=False, dtype=torch.float16)
    assert (f16 - ref).abs().max().item() / ref.abs().max().item() < 2e-2


@pytest.mark.skipif(not CUDA, reason="CUDA")
@pytest.mark.parametrize("name", list(ENCODERS))
@pytest.mark.parametrize("reference", [True, False])
def test_tracks_are_independent_and_order_matters(name, reference):
    enc = _build(name)
    x, cu, seq_idx = _batch()
    a = _pooled(enc, x, cu, seq_idx, reference)
    x2 = x.clone()
    s, e = int(cu[0]), int(cu[1])
    x2[0, s:e] = x[0, s:e].flip(0)             # reverse the hit order of track 0 only
    b = _pooled(enc, x2, cu, seq_idx, reference)
    assert (a[1:] - b[1:]).abs().max().item() < 1e-5          # other tracks untouched
    assert (a[0] - b[0]).abs().max().item() > 1e-5            # the encoder sees the hit order


@pytest.mark.skipif(not CUDA, reason="CUDA")
@pytest.mark.parametrize("name", list(ENCODERS))
def test_training_step_reaches_every_parameter(name):
    enc = _build(name).train()
    x, cu, seq_idx = _batch()
    enc(x.requires_grad_(), cu_seqlens=cu, seq_idx=seq_idx)[1].square().sum().backward()
    for n, p in enc.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), n


def test_mamba2_block_matches_fp64_reference():
    torch.manual_seed(3)
    blk = Mamba2Short(d_model=32, d_state=16, d_conv=4, expand=2, headdim=16).double()
    u = torch.randn(3, 22, 32, dtype=torch.float64)
    assert torch.allclose(blk(u), mamba2_block_ref(blk, u), atol=1e-10, rtol=1e-8)


def test_mingru_scan_and_adjoint():
    torch.manual_seed(4)
    a = torch.rand(2, 20, 5, dtype=torch.float64, requires_grad=True)
    b = torch.randn(2, 20, 5, dtype=torch.float64, requires_grad=True)
    assert torch.allclose(mingru_scan(a, b), mingru_scan_ref(a, b), atol=1e-12)
    assert torch.autograd.gradcheck(mingru_scan, (a, b))
