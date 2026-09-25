"""Triton inference kernels against pure-PyTorch references."""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA kernels")


def _cu(lens, device="cuda", dtype=torch.int32):
    cu = torch.zeros(len(lens) + 1, dtype=dtype, device=device)
    cu[1:] = torch.cumsum(torch.as_tensor(lens, device=device), 0)
    return cu


def _rel(a, b):
    return ((a.double() - b.double()).abs().max() / b.double().abs().max()).item()


# ---------------------------------------------------------------- minGRU scan

@pytest.mark.parametrize("dtype,tol", [(torch.float32, 1e-5), (torch.float16, 2e-3)])
def test_mingru_packed_kernel_matches_scan(dtype, tol):
    from track_regression.mingru import mingru_scan_ref
    from track_regression.ops.mingru_short_triton import mingru_bidi_packed

    torch.manual_seed(0)
    H, lens = 192, [6, 20, 13, 9, 17, 20, 7]
    cu = _cu(lens)
    zn = torch.randn(int(cu[-1]), 4 * H, device="cuda")
    out = mingru_bidi_packed(zn.to(dtype).contiguous(), cu, H, 20)
    assert out.dtype == dtype
    zn = zn.to(dtype).double()
    for i, (a, b) in enumerate(zip(cu[:-1].tolist(), cu[1:].tolist())):
        z_f, n_f, z_b, n_b = zn[a:b].split(H, dim=-1)
        zf, zb = torch.sigmoid(z_f), torch.sigmoid(z_b)
        hf = mingru_scan_ref((1 - zf)[None], (zf * n_f)[None])[0]
        hb = mingru_scan_ref((1 - zb)[None], (zb * n_b)[None], reverse=True)[0]
        assert _rel(out[a:b, :H], hf) < tol and _rel(out[a:b, H:], hb) < tol, i


def test_mingru_kernel_compiles_for_ada_sm89():
    """Every autotune config cross-compiles for sm_89 without Hopper-only PTX."""
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource, compile as tcompile

    from track_regression.ops import mingru_short_triton as M

    kern = M._mingru_bidi_packed_kernel
    sig = {"zn_ptr": "*fp32", "cu_ptr": "*i32", "out_ptr": "*fp32",
           "H": "constexpr", "MAXL": "constexpr", "BD": "constexpr"}
    for cfg in kern.configs:
        cc = tcompile(ASTSource(fn=kern.fn, signature=sig, constexprs={"H": 192, "MAXL": 20, "BD": cfg.kwargs["BD"]}),
                      target=GPUTarget("cuda", 89, 32),
                      options={"num_warps": cfg.num_warps, "num_stages": cfg.num_stages})
        ptx = cc.asm["ptx"]
        assert ".target sm_89" in ptx and cc.metadata.shared <= 101_376
        assert not any(k in ptx for k in ("wgmma", "cp.async.bulk", "mbarrier", "setmaxnreg"))


# ---------------------------------------------------------------- Mamba-2 SSD

H, P, N, DCONV = 8, 32, 64, 4
DPROJ = 2 * H * P + 2 * N + H


def _ssd_batch(n_tracks=3000, seed=0):
    g = torch.Generator().manual_seed(seed)
    lens = torch.randint(6, 21, (n_tracks,), generator=g) + 2       # + 2 CLS tokens, both BL buckets
    cu = _cu(lens, dtype=torch.int64)
    zx = (torch.randn(int(cu[-1]), DPROJ, generator=g) * 0.5).cuda()
    w = dict(conv_w=(torch.randn(H * P + 2 * N, 1, DCONV, generator=g) * 0.3).cuda(),
             conv_b=(torch.randn(H * P + 2 * N, generator=g) * 0.1).cuda(),
             dt_bias=(torch.randn(H, generator=g) * 0.5 - 2.0).cuda(),
             A_log=(torch.randn(H, generator=g) * 0.3).cuda(), D=torch.randn(H, generator=g).cuda())
    return zx, w, cu


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("dtype,tol", [(torch.float32, 2e-4), (torch.float16, 5e-3)])
def test_ssd_packed_kernel_matches_reference(reverse, dtype, tol):
    from track_regression.ops.ssd_short_triton import _packed_scan_torch_ref, ssd_short_fwd_packed

    zx, w, cu = _ssd_batch()
    args = (w["conv_w"], w["conv_b"], w["dt_bias"], w["A_log"], w["D"], cu, H, P, N, reverse)
    y = ssd_short_fwd_packed(zx.to(dtype), *args)
    ref = _packed_scan_torch_ref(zx, *args)
    assert y.dtype == dtype and torch.isfinite(y).all()
    assert _rel(y, ref) < tol


@pytest.mark.parametrize("dtype,tol", [(torch.float32, 1e-5), (torch.float16, 2e-3)])
def test_gated_rmsnorm_matches_torch(dtype, tol):
    from track_regression.ops.ssd_short_triton import gated_rmsnorm

    torch.manual_seed(1)
    y = torch.randn(1000, 256, device="cuda").to(dtype)
    z = torch.randn(1000, 648, device="cuda").to(dtype)       # z is the leading slice of the projection row
    wt = 1.0 + 0.1 * torch.randn(256, device="cuda")
    g = y.float() * torch.nn.functional.silu(z[:, :256].float())
    ref = g * torch.rsqrt(g.square().mean(-1, keepdim=True) + 1e-5) * wt
    assert _rel(gated_rmsnorm(y.contiguous(), z, wt, 1e-5), ref) < tol


# ---------------------------------------------------------------- Transformer

D_TXF, NH = 128, 4


@pytest.mark.parametrize("dtype,tol", [(torch.float32, 2e-5), (torch.float16, 2e-2)])
@pytest.mark.parametrize("norm", [True, False])
def test_attention_kernel_matches_oracle(dtype, tol, norm):
    from track_regression.ops.attn_short_triton import attn_packed_tracks, attn_packed_tracks_reference

    torch.manual_seed(2)
    cu = _cu([8, 22, 13, 6, 20, 15, 9, 22, 11])
    qkv = torch.randn(int(cu[-1]), 3 * D_TXF, device="cuda", dtype=dtype)
    w = [1.0 + 0.1 * torch.randn(D_TXF, device="cuda") for _ in range(3)]
    eps = torch.finfo(torch.float32).eps
    out = attn_packed_tracks(qkv.contiguous(), cu, *w, NH, eps, norm, 22)
    assert _rel(out, attn_packed_tracks_reference(qkv.double(), cu, *w, NH, eps, norm)) < tol


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_add_rmsnorm_matches_torch(dtype):
    from track_regression.ops.attn_short_triton import add_rmsnorm_packed

    torch.manual_seed(3)
    x = torch.randn(1001, D_TXF, device="cuda")
    o = torch.randn(1001, D_TXF, device="cuda", dtype=dtype)
    g, w = 0.1 * torch.randn(D_TXF, device="cuda"), 1.0 + 0.1 * torch.randn(D_TXF, device="cuda")
    eps = torch.finfo(torch.float32).eps
    y, h = add_rmsnorm_packed(x, o, g, w, eps, True, True, dtype)
    y_ref = x + g * o.float()
    assert (y.float() - y_ref).abs().max().item() < 1e-6
    assert _rel(h, torch.nn.functional.rms_norm(y_ref, (D_TXF,), w, eps)) < (1e-5 if dtype == torch.float32 else 2e-3)


@pytest.mark.parametrize("epi", [0, 1, 2])
def test_gemm_epilogue_matches_torch(epi):
    from track_regression.ops.attn_short_triton import gemm_epilogue_fp16

    torch.manual_seed(4)
    M, K, Nn = 1234, 128, (384 if epi == 1 else 128)
    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    w = (torch.randn(Nn, K, device="cuda") * 0.05).half()
    b = (0.1 * torch.randn(Nn, device="cuda")).half()
    x, g = torch.randn(M, Nn, device="cuda"), 0.5 * torch.randn(Nn, device="cuda")
    nw = 1.0 + 0.1 * torch.randn(Nn, device="cuda")
    eps = torch.finfo(torch.float32).eps
    out, y, h = gemm_epilogue_fp16(a, w, b, x, g, nw, eps, epi, True)
    acc = a.float() @ w.float().t() + b.float()
    if epi == 0:
        assert _rel(out, acc) < 2e-3
    elif epi == 1:
        assert _rel(out, torch.nn.functional.silu(acc)) < 2e-3
    else:
        y_ref = x + g * acc
        assert _rel(y, y_ref) < 2e-3
        assert _rel(h, torch.nn.functional.rms_norm(y_ref, (Nn,), nw, eps)) < 4e-3
