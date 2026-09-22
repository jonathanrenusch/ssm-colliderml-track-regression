"""Parity gate for the packed transformer inference path (TRK_TXF_PACKED=1).

* the per-track attention kernel against a pure-torch oracle (fp32 IEEE dots
  and fp16 tensor-core dots), with and without the fused q/k/v RMSNorm;
* the whole ``IndexPosEncTransformerCLS`` packed path against the padded path
  it replaces, on a random packed batch, in fp32 and under fp16 autocast.
"""

from __future__ import annotations

import os

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA kernel")

D, NH = 128, 4


def _packed(lens, dtype, device="cuda"):
    cu = torch.zeros(len(lens) + 1, dtype=torch.int32, device=device)
    cu[1:] = torch.cumsum(torch.tensor(lens, device=device), 0)
    return cu, int(cu[-1])


@pytest.mark.parametrize("dtype,tol", [(torch.float32, 2e-5), (torch.float16, 2e-2)])
@pytest.mark.parametrize("norm", [True, False])
def test_kernel_matches_oracle(dtype, tol, norm):
    from track_regression.ops.attn_short_triton import (
        attn_packed_tracks, attn_packed_tracks_reference)
    torch.manual_seed(0)
    lens = [8, 22, 13, 6, 20, 15, 9, 22, 11]
    cu, T = _packed(lens, dtype)
    qkv = torch.randn(T, 3 * D, device="cuda", dtype=dtype)
    w = [1.0 + 0.1 * torch.randn(D, device="cuda") for _ in range(3)]
    eps = torch.finfo(torch.float32).eps
    out = attn_packed_tracks(qkv.contiguous(), cu, *w, NH, eps, norm, 22)
    ref = attn_packed_tracks_reference(qkv.double(), cu, *w, NH, eps, norm)
    err = (out.double() - ref).abs().max().item() / ref.abs().max().item()
    assert err < tol, err


def _build_encoder():
    from track_regression.ablation_encoders import IndexPosEncTransformerCLS
    return IndexPosEncTransformerCLS(
        dim=D, num_cls_tokens=2, num_layers=3, attn_type="torch", norm="RMSNorm",
        value_residual=False, qkv_norm=True, layer_scale=1.0e-5,
        attn_kwargs={"num_heads": NH}, dense_kwargs={"hidden_dim_scale": 3},
        posenc_fourier_scales=[-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
        posenc_fourier_base=2, posenc_time_scale=1.0,
    ).cuda().eval()


def _batch(lens):
    cu, T = _packed(lens, torch.float32)
    x = torch.randn(1, T, D, device="cuda")
    seq_idx = torch.repeat_interleave(
        torch.arange(len(lens), device="cuda", dtype=torch.int32),
        torch.tensor(lens, device="cuda")).unsqueeze(0)
    return x, seq_idx, cu


def _run(enc, x, seq_idx, cu, packed: bool, autocast_dtype=None):
    os.environ["TRK_TXF_PACKED"] = "1" if packed else "0"
    try:
        with torch.inference_mode(), torch.autocast(
                "cuda", dtype=autocast_dtype or torch.float16,
                enabled=autocast_dtype is not None):
            _, pooled = enc(x, x_sort_value=None, seq_idx=seq_idx, cu_seqlens=cu)
        return pooled.float()
    finally:
        os.environ.pop("TRK_TXF_PACKED", None)


def test_packed_path_matches_padded_fp32():
    torch.manual_seed(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    enc = _build_encoder()
    # LayerScale at 1e-5 hides the layers; scale it up so the test sees them.
    with torch.no_grad():
        for layer in enc.encoder.layers:
            layer.attn.ls.gamma.fill_(0.5)
            layer.dense.ls.gamma.fill_(0.5)
    lens = [7, 12, 20, 6, 13, 20, 9, 18, 6, 15]
    x, seq_idx, cu = _batch(lens)
    a = _run(enc, x, seq_idx, cu, packed=False)
    b = _run(enc, x, seq_idx, cu, packed=True)
    assert a.shape == b.shape == (len(lens), 2 * D)
    err = (a - b).abs().max().item() / a.abs().max().item()
    assert err < 1e-4, err


def test_packed_path_matches_padded_fp16_autocast():
    torch.manual_seed(2)
    enc = _build_encoder()
    with torch.no_grad():
        for layer in enc.encoder.layers:
            layer.attn.ls.gamma.fill_(0.5)
            layer.dense.ls.gamma.fill_(0.5)
    lens = [7, 12, 20, 6, 13, 20, 9, 18, 6, 15]
    x, seq_idx, cu = _batch(lens)
    a = _run(enc, x, seq_idx, cu, packed=False, autocast_dtype=torch.float16)
    b = _run(enc, x, seq_idx, cu, packed=True, autocast_dtype=torch.float16)
    ref = _run(enc, x, seq_idx, cu, packed=False)          # fp32 truth
    # both fp16 paths must sit within fp16 distance of the fp32 result
    for p in (a, b):
        assert (p - ref).abs().max().item() / ref.abs().max().item() < 2e-2


def test_packed_path_is_segment_independent():
    """One track's pooled vector must not depend on the other tracks in the
    batch (the block-diagonal attention)."""
    torch.manual_seed(3)
    enc = _build_encoder()
    lens = [7, 12, 9, 20]
    x, seq_idx, cu = _batch(lens)
    a = _run(enc, x, seq_idx, cu, packed=True)
    x2 = x.clone()
    x2[0, int(cu[0]):int(cu[1])] = torch.randn_like(x2[0, int(cu[0]):int(cu[1])])
    b = _run(enc, x2, seq_idx, cu, packed=True)
    assert (a[1:] - b[1:]).abs().max().item() < 1e-6
    assert (a[0] - b[0]).abs().max().item() > 1e-6


@pytest.mark.parametrize("has_res,want_h", [(True, True), (False, True), (True, False)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_add_rmsnorm_matches_torch(has_res, want_h, dtype):
    from track_regression.ops.attn_short_triton import add_rmsnorm_packed
    torch.manual_seed(4)
    T = 1001
    x = torch.randn(T, D, device="cuda")
    o = torch.randn(T, D, device="cuda", dtype=dtype)
    g = 0.1 * torch.randn(D, device="cuda")
    w = 1.0 + 0.1 * torch.randn(D, device="cuda")
    eps = torch.finfo(torch.float32).eps
    y, h = add_rmsnorm_packed(x, o, g, w, eps, has_res, want_h, dtype)
    y_ref = x + g * o.float() if has_res else x
    if has_res:
        assert (y.float() - y_ref).abs().max().item() < 1e-6
    else:
        assert y.shape == (0, D)
    if want_h:
        h_ref = torch.nn.functional.rms_norm(y_ref, (D,), w, eps)
        tol = 1e-5 if dtype == torch.float32 else 2e-3
        assert (h.float() - h_ref).abs().max().item() / h_ref.abs().max().item() < tol
    else:
        assert h.shape == (0, D)


def test_packed_fused_norm_matches_unfused():
    torch.manual_seed(5)
    torch.backends.cuda.matmul.allow_tf32 = False
    enc = _build_encoder()
    with torch.no_grad():
        for layer in enc.encoder.layers:
            layer.attn.ls.gamma.fill_(0.5)
            layer.dense.ls.gamma.fill_(0.5)
    lens = [7, 12, 20, 6, 13, 20, 9, 18, 6, 15]
    x, seq_idx, cu = _batch(lens)
    ref = _run(enc, x, seq_idx, cu, packed=False)
    os.environ["TRK_TXF_PACKED_FUSED_NORM"] = "1"
    try:
        enc._packed_body = None            # force a fresh compile with the flag
        b = _run(enc, x, seq_idx, cu, packed=True)
    finally:
        os.environ.pop("TRK_TXF_PACKED_FUSED_NORM", None)
        enc._packed_body = None
    assert (ref - b).abs().max().item() / ref.abs().max().item() < 1e-4


@pytest.mark.parametrize("epi", [0, 1, 2])
def test_gemm_epilogue_matches_torch(epi):
    from track_regression.ops.attn_short_triton import gemm_epilogue_fp16
    torch.manual_seed(6)
    M, K, N = 1234, 128, (384 if epi == 1 else 128)
    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    w = (torch.randn(N, K, device="cuda") * 0.05).half()
    b = (0.1 * torch.randn(N, device="cuda")).half()
    x = torch.randn(M, N, device="cuda")
    g = 0.5 * torch.randn(N, device="cuda")
    nw = 1.0 + 0.1 * torch.randn(N, device="cuda")
    eps = torch.finfo(torch.float32).eps
    out, y, h = gemm_epilogue_fp16(a, w, b, x, g, nw, eps, epi, True)
    acc = a.float() @ w.float().t() + b.float()
    if epi == 0:
        assert (out.float() - acc).abs().max().item() / acc.abs().max().item() < 2e-3
    elif epi == 1:
        ref = torch.nn.functional.silu(acc)
        assert (out.float() - ref).abs().max().item() / ref.abs().max().item() < 2e-3
    else:
        y_ref = x + g * acc
        h_ref = torch.nn.functional.rms_norm(y_ref, (N,), nw, eps)
        assert (y - y_ref).abs().max().item() / y_ref.abs().max().item() < 2e-3
        assert (h.float() - h_ref).abs().max().item() / h_ref.abs().max().item() < 4e-3


def test_packed_fused_gemm_matches_padded_fp16():
    torch.manual_seed(7)
    enc = _build_encoder()
    with torch.no_grad():
        for layer in enc.encoder.layers:
            layer.attn.ls.gamma.fill_(0.5)
            layer.dense.ls.gamma.fill_(0.5)
    lens = [7, 12, 20, 6, 13, 20, 9, 18, 6, 15]
    x, seq_idx, cu = _batch(lens)
    ref = _run(enc, x, seq_idx, cu, packed=False)           # fp32 truth
    os.environ["TRK_TXF_PACKED_FUSED_GEMM"] = "1"
    try:
        enc._packed_body = None
        b = _run(enc, x, seq_idx, cu, packed=True, autocast_dtype=torch.float16)
    finally:
        os.environ.pop("TRK_TXF_PACKED_FUSED_GEMM", None)
        enc._packed_body = None
    assert (b - ref).abs().max().item() / ref.abs().max().item() < 2e-2
