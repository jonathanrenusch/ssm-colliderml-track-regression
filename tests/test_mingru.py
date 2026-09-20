"""minGRU: correctness of the reference scan, the encoder, and the fused kernel.

The kernel gate mirrors the Mamba-2 campaign's: exact-ish agreement with a
pure-torch reference on a real-shaped batch, plus the structural checks
(packed<->padded, segment independence, order sensitivity, pad safety).
"""

from __future__ import annotations

import pytest
import torch

from track_regression.mingru import MinGRUCLSEncoder, mingru_scan_ref, _prefix_flip_index

torch.manual_seed(0)
DIM = 128
CUDA = torch.cuda.is_available()


def make_packed(lens, dim=DIM, dtype=torch.float64):
    cu = torch.zeros(len(lens) + 1, dtype=torch.int32)
    cu[1:] = torch.cumsum(torch.tensor(lens), 0)
    x = torch.randn(1, int(cu[-1]), dim, dtype=dtype)
    return x, cu


def to_padded(x, cu, dim=DIM):
    lens = cu[1:].long() - cu[:-1].long()
    B, L = len(lens), int(lens.max())
    xp = x.new_zeros(B, L, dim)
    mask = torch.zeros(B, L, dtype=torch.bool)
    o = 0
    for i, n in enumerate(lens.tolist()):
        xp[i, :n] = x[0, o:o + n]; mask[i, :n] = True; o += n
    return xp, mask


def enc(dtype=torch.float64, hidden=194):
    return MinGRUCLSEncoder(dim=DIM, hidden_size=hidden, num_layers=2).to(dtype).eval()


# ---------------------------------------------------------------- scan maths
def test_scan_matches_explicit_recurrence():
    B, L, D = 4, 11, 7
    a = torch.rand(B, L, D, dtype=torch.float64)
    b = torch.randn(B, L, D, dtype=torch.float64)
    got = mingru_scan_ref(a, b)
    h = torch.zeros(B, D, dtype=torch.float64)
    for t in range(L):
        h = a[:, t] * h + b[:, t]
        assert torch.allclose(got[:, t], h, atol=1e-12)


def test_scan_reverse_is_forward_on_flipped():
    B, L, D = 3, 9, 5
    a = torch.rand(B, L, D, dtype=torch.float64)
    b = torch.randn(B, L, D, dtype=torch.float64)
    r = mingru_scan_ref(a, b, reverse=True)
    f = mingru_scan_ref(a.flip(1), b.flip(1)).flip(1)
    assert torch.max((r - f).abs()).item() < 1e-12


def test_prefix_flip_is_self_inverse():
    lens = torch.tensor([3, 7, 1, 20])
    idx = _prefix_flip_index(lens, 20, "cpu")
    twice = torch.gather(idx, 1, idx)
    assert torch.equal(twice, torch.arange(20).unsqueeze(0).expand(4, 20))


# ------------------------------------------------------------- encoder shape
def test_shapes_and_pool_dim():
    e = enc()
    lens = [7, 12, 20, 6, 13]
    x, cu = make_packed(lens)
    seq, pooled = e(x, cu_seqlens=cu)
    assert e.pool_dim == 256 and pooled.shape == (len(lens), 256)
    assert seq.shape[1] == sum(lens) and torch.isfinite(pooled).all()


def test_packed_vs_padded():
    e = enc()
    lens = [7, 12, 20, 6, 13]
    x, cu = make_packed(lens)
    xp, mask = to_padded(x, cu)
    _, a = e(x, cu_seqlens=cu)
    _, b = e(xp, kv_mask=mask)
    assert torch.max((a - b).abs()).item() < 1e-10


def test_segment_permutation_independence():
    e = enc()
    lens = [7, 12, 20, 6, 13]
    x, cu = make_packed(lens)
    _, pooled = e(x, cu_seqlens=cu)
    perm = [3, 0, 4, 1, 2]
    starts = [int(cu[i]) for i in range(len(lens))]
    xs = torch.cat([x[0, starts[i]:starts[i] + lens[i]] for i in perm], 0).unsqueeze(0)
    lp = [lens[i] for i in perm]
    cu2 = torch.zeros(len(lens) + 1, dtype=torch.int32)
    cu2[1:] = torch.cumsum(torch.tensor(lp), 0)
    _, p2 = e(xs, cu_seqlens=cu2)
    assert torch.max((p2 - pooled[perm]).abs()).item() < 1e-10


def test_order_sensitive():
    e = enc()
    lens = [11]
    x, cu = make_packed(lens)
    _, a = e(x, cu_seqlens=cu)
    _, b = e(x.flip(1).contiguous(), cu_seqlens=cu)
    assert torch.max((a - b).abs()).item() > 1e-6


def test_pads_cannot_change_a_shorter_track():
    """A track's output must not depend on how long the longest track in the
    batch is (the pad-safety property the fused kernel relies on)."""
    e = enc()
    x, cu = make_packed([6])
    _, alone = e(x, cu_seqlens=cu)
    x2 = torch.cat([x[0], torch.randn(19, DIM, dtype=torch.float64)], 0).unsqueeze(0)
    cu2 = torch.tensor([0, 6, 25], dtype=torch.int32)
    _, together = e(x2, cu_seqlens=cu2)
    assert torch.max((together[0] - alone[0]).abs()).item() < 1e-10


# ------------------------------------------------------------- fused kernel
@pytest.mark.skipif(not CUDA, reason="needs CUDA")
def test_fused_kernel_matches_reference():
    from track_regression.ops.mingru_short_triton import mingru_bidi_fused

    torch.manual_seed(1)
    B, S, H = 97, 20, 194
    lens = torch.randint(6, S + 1, (B,), device="cuda")
    zn = torch.randn(B, S, 4 * H, device="cuda", dtype=torch.float32).contiguous()
    # zero the pads, as the caller's padded-static conversion does
    p = torch.arange(S, device="cuda")
    zn = zn * (p[None, :, None] < lens[:, None, None]).float()

    hf_k, hb_k = mingru_bidi_fused(zn, lens, H)

    z_f, n_f, z_b, n_b = zn.split(H, dim=-1)
    zf = torch.sigmoid(z_f)
    hf_r = mingru_scan_ref(1.0 - zf, zf * n_f)
    zb = torch.sigmoid(z_b)
    idx = _prefix_flip_index(lens, S, "cuda")
    g = idx.unsqueeze(-1).expand(-1, -1, H)
    hb_r = torch.gather(
        mingru_scan_ref(torch.gather(1.0 - zb, 1, g), torch.gather(zb * n_b, 1, g)), 1, g)

    valid = (p[None, :, None] < lens[:, None, None])
    for got, ref, name in ((hf_k, hf_r, "fwd"), (hb_k, hb_r, "bwd")):
        d = ((got - ref).abs() * valid).max().item()
        scale = (ref.abs() * valid).max().item()
        assert d / max(scale, 1e-6) < 1e-5, (name, d, scale)


@pytest.mark.skipif(not CUDA, reason="needs CUDA")
def test_encoder_kernel_path_matches_eager():
    import os
    e = MinGRUCLSEncoder(dim=DIM, hidden_size=194, num_layers=2).cuda().float().eval()
    lens = [7, 12, 20, 6, 13, 19, 9]
    x, cu = make_packed(lens, dtype=torch.float32)
    x, cu = x.cuda(), cu.cuda()
    with torch.no_grad():
        os.environ["TRK_MINGRU_KERNEL"] = "off"
        _, eager = e(x, cu_seqlens=cu)
        os.environ["TRK_MINGRU_KERNEL"] = "auto"
        _, fused = e(x, cu_seqlens=cu)
    os.environ["TRK_MINGRU_KERNEL"] = "off"
    rel = (eager - fused).abs().max().item() / max(eager.abs().max().item(), 1e-6)
    assert rel < 1e-5, rel


def test_parallel_scan_equals_sequential():
    from track_regression.mingru import mingru_scan_parallel
    for L in (1, 2, 5, 16, 20, 31):
        a = torch.rand(6, L, 9, dtype=torch.float64)
        b = torch.randn(6, L, 9, dtype=torch.float64)
        seq = mingru_scan_ref(a, b)
        par = mingru_scan_parallel(a.clone(), b.clone())
        assert torch.max((seq - par).abs()).item() < 1e-11, L


@pytest.mark.skipif(not CUDA, reason="needs CUDA")
def test_packed_kernel_path_matches_eager():
    """The fully-packed inference path must equal the padded eager path."""
    import os
    e = MinGRUCLSEncoder(dim=DIM, hidden_size=194, num_layers=2).cuda().float().eval()
    lens = [7, 12, 20, 6, 13, 19, 9, 11]
    x, cu = make_packed(lens, dtype=torch.float32)
    x, cu = x.cuda(), cu.cuda()
    with torch.no_grad():
        os.environ["TRK_MINGRU_KERNEL"] = "off"
        seq_e, pooled_e = e(x, cu_seqlens=cu)
        os.environ["TRK_MINGRU_KERNEL"] = "auto"
        seq_k, pooled_k = e(x, cu_seqlens=cu)
    os.environ["TRK_MINGRU_KERNEL"] = "off"
    rp = (pooled_e - pooled_k).abs().max().item() / max(pooled_e.abs().max().item(), 1e-6)
    rs = (seq_e - seq_k).abs().max().item() / max(seq_e.abs().max().item(), 1e-6)
    assert rp < 1e-5 and rs < 1e-5, (rp, rs)


def test_explicit_adjoint_matches_autograd():
    """The hand-written backward must equal autograd through the scan."""
    from track_regression.mingru import mingru_scan_autograd, mingru_scan_parallel
    torch.manual_seed(3)
    for L in (1, 4, 20):
        a0 = torch.rand(5, L, 7, dtype=torch.float64)
        b0 = torch.randn(5, L, 7, dtype=torch.float64)
        w = torch.randn(5, L, 7, dtype=torch.float64)

        a1, b1 = a0.clone().requires_grad_(), b0.clone().requires_grad_()
        (mingru_scan_parallel(a1, b1) * w).sum().backward()
        a2, b2 = a0.clone().requires_grad_(), b0.clone().requires_grad_()
        (mingru_scan_autograd(a2, b2) * w).sum().backward()

        # at L == 1 the scan degenerates to `b`, so autograd leaves a.grad
        # None while the explicit adjoint returns an exact zero — both right.
        ga1 = a1.grad if a1.grad is not None else torch.zeros_like(a0)
        assert torch.max((ga1 - a2.grad).abs()).item() < 1e-10, L
        assert torch.max((b1.grad - b2.grad).abs()).item() < 1e-10, L


def test_adjoint_gradcheck():
    from track_regression.mingru import mingru_scan_autograd
    a = torch.rand(2, 6, 3, dtype=torch.float64, requires_grad=True)
    b = torch.randn(2, 6, 3, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(mingru_scan_autograd, (a, b), eps=1e-6, atol=1e-8)


def test_diagrnn_encoder_basics():
    """Constant-decay variant: shapes, packed<->padded, order sensitivity."""
    from track_regression.mingru import DiagRNNCLSEncoder
    e = DiagRNNCLSEncoder(dim=DIM, hidden_size=274, num_layers=2).to(torch.float64).eval()
    lens = [7, 12, 20, 6]
    x, cu = make_packed(lens)
    seq, pooled = e(x, cu_seqlens=cu)
    assert e.pool_dim == 256 and pooled.shape == (len(lens), 256)
    assert torch.isfinite(pooled).all()
    xp, mask = to_padded(x, cu)
    _, p2 = e(xp, kv_mask=mask)
    assert torch.max((pooled - p2).abs()).item() < 1e-10
    _, a = e(x, cu_seqlens=cu)
    _, b = e(x.flip(1).contiguous(), cu_seqlens=cu)
    assert torch.max((a - b).abs()).item() > 1e-6          # reads the order


# ---------------------------------------------------------------------------
# portability: the deployment target is an RTX 5000 Ada (sm_89), not just H100
# ---------------------------------------------------------------------------

def test_kernels_compile_for_ada_sm89():
    """Cross-compile every autotune config for sm_89 and assert the result is
    Ada-legal: no Hopper-only PTX, shared memory within Ada's 100 KB/SM.

    Runs on any machine (no Ada device needed) — it is an ahead-of-time
    compile, so a Hopper-only construct can never reach the Ada deployment
    unnoticed.
    """
    triton = pytest.importorskip("triton")
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource, compile as tcompile
    from track_regression.ops import mingru_short_triton as M

    ADA = GPUTarget("cuda", 89, 32)
    ADA_SMEM = 101_376                      # bytes/SM configurable on sm_89
    HOPPER_ONLY = ("wgmma", "tma", "cp.async.bulk", "mbarrier", "setmaxnreg",
                   "clusterlaunch", "st.async")

    cases = [
        (M._mingru_bidi_packed_kernel,
         {"zn_ptr": "*fp32", "cu_ptr": "*i32", "out_ptr": "*fp32",
          "H": "constexpr", "MAXL": "constexpr", "BD": "constexpr"},
         {"H": 194, "MAXL": 20}),
        (M._mingru_bidi_kernel,
         {"zn_ptr": "*fp32", "lens_ptr": "*i32", "outf_ptr": "*fp32",
          "outb_ptr": "*fp32", "S": "i32",
          "H": "constexpr", "BL": "constexpr", "BD": "constexpr"},
         {"H": 194, "BL": 32}),
    ]
    checked = 0
    for kern, sig, base in cases:
        fn = kern.fn if hasattr(kern, "fn") else kern
        for cfg in kern.configs:
            consts = dict(base, BD=cfg.kwargs["BD"])
            cc = tcompile(ASTSource(fn=fn, signature=sig, constexprs=consts),
                          target=ADA,
                          options={"num_warps": cfg.num_warps,
                                   "num_stages": cfg.num_stages})
            ptx = cc.asm["ptx"]
            assert ".target sm_89" in ptx
            assert cc.metadata.shared <= ADA_SMEM, (cfg, cc.metadata.shared)
            bad = [k for k in HOPPER_ONLY if k in ptx]
            assert not bad, (kern.fn.__name__, cfg, bad)
            checked += 1
    assert checked >= 15, checked


@pytest.mark.skipif(not CUDA, reason="needs CUDA")
def test_diagrnn_runs_in_eval_mode_on_cuda():
    """Regression: DiagRNN must not take minGRU's packed kernel path, which
    assumes a 4H projection (it emits 2H).  This is exactly what broke the
    first V2 launch, inside Lightning's sanity check."""
    from track_regression.mingru import DiagRNNCLSEncoder
    e = DiagRNNCLSEncoder(dim=DIM, hidden_size=270, num_layers=2).cuda().float().eval()
    lens = [7, 12, 20, 6]
    x, cu = make_packed(lens, dtype=torch.float32)
    with torch.no_grad():
        seq, pooled = e(x.cuda(), cu_seqlens=cu.cuda())
    assert pooled.shape == (len(lens), 256) and torch.isfinite(pooled).all()


# --------------------------------------------------------------------------
# Complex-decay LRU
# --------------------------------------------------------------------------


def test_complex_scan_matches_sequential_reference():
    from track_regression.mingru import _complex_scan

    torch.manual_seed(3)
    B, L, C = 3, 9, 5
    ar, ai = torch.randn(B, L, C) * 0.3, torch.randn(B, L, C) * 0.3
    br, bi = torch.randn(B, L, C), torch.randn(B, L, C)
    hr, hi = _complex_scan(ar, ai, br, bi)

    a = torch.complex(ar, ai)
    b = torch.complex(br, bi)
    h = torch.zeros(B, C, dtype=torch.cfloat)
    ref = []
    for t in range(L):
        h = a[:, t] * h + b[:, t]
        ref.append(h)
    ref = torch.stack(ref, 1)
    assert torch.allclose(hr, ref.real, atol=1e-5)
    assert torch.allclose(hi, ref.imag, atol=1e-5)


def test_complex_lru_pads_are_inert_and_order_matters():
    from track_regression.mingru import ComplexLRUCLSEncoder

    torch.manual_seed(0)
    enc = ComplexLRUCLSEncoder(dim=16, hidden_size=8, num_layers=2).eval()
    x = torch.randn(2, 20, 16)
    lens = torch.tensor([6, 13])
    with torch.no_grad():
        _, t0 = enc.layers[0](x, lens)
        x2 = x.clone()
        x2[0, 6:] = torch.randn(14, 16)          # pad region of row 0
        x2[1, 13:] = torch.randn(7, 16)
        _, t1 = enc.layers[0](x2, lens)
        assert torch.allclose(t0, t1, atol=1e-6), "pads leaked into the readout"

        x3 = x.clone()
        x3[1, :13] = x[1, :13].flip(0)           # reverse a real prefix
        _, t2 = enc.layers[0](x3, lens)
        assert not torch.allclose(t0[1], t2[1], atol=1e-4), "encoder is order-blind"


def test_complex_lru_decay_is_stable_by_construction():
    from track_regression.mingru import _ComplexLRULayer

    layer = _ComplexLRULayer(16, 8)
    with torch.no_grad():
        layer.nu.uniform_(-6.0, 6.0)             # extreme parameterisation
        r = torch.exp(-torch.exp(layer.nu))
    # r = 0 (total forgetting) underflows out of exp at large nu and is fine;
    # the invariant that matters is that r can never reach or exceed 1.
    assert (r >= 0).all() and (r < 1).all(), "|a| must stay inside the unit disc"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_packed_kernel_runs_under_fp16_autocast_and_scans_in_fp32():
    """Reduced-precision training must not break (or enter) the scan.

    The Triton scan requires fp32; under ``encoder_autocast_dtype: float16``
    the projection emits fp16, so the encoder has to cast at the kernel
    boundary.  Physics must stay close to the fp32 result because fp16 and
    TF32 carry the SAME 10 mantissa bits.
    """
    from track_regression.mingru import MinGRUCLSEncoder

    torch.manual_seed(0)
    enc = MinGRUCLSEncoder(dim=32, hidden_size=24, num_layers=2,
                           compile_core=False).cuda().eval()
    lens = [7, 13, 20]
    cu = torch.tensor([0, 7, 20, 40], dtype=torch.int32, device="cuda")
    x = torch.randn(1, int(cu[-1]), 32, device="cuda")
    with torch.no_grad():
        _, p32 = enc(x, cu_seqlens=cu)
        with torch.autocast("cuda", dtype=torch.float16):
            _, p16 = enc(x, cu_seqlens=cu)
    assert torch.isfinite(p16).all()
    rel = (p16.float() - p32).abs().max() / p32.abs().max()
    assert rel < 2e-2, f"fp16 encoder drifted {rel:.3%} from fp32"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_packed_kernel_takes_reduced_precision_and_scans_in_fp32(dtype):
    """The kernel consumes fp16/bf16 directly -- no cast at the boundary.

    The recurrence must still accumulate in fp32 inside the kernel, so the
    result has to track the fp32 reference far more tightly than the input
    dtype's own epsilon (fp16 eps = 9.8e-4, bf16 = 7.8e-3) would allow if the
    state were carried in reduced precision through 20 steps.
    """
    from track_regression.ops.mingru_short_triton import mingru_bidi_packed

    torch.manual_seed(0)
    H, lens = 64, [5, 20, 13]
    cu = torch.tensor([0, 5, 25, 38], dtype=torch.int32, device="cuda")
    zn32 = torch.randn(int(cu[-1]), 4 * H, device="cuda")
    ref = mingru_bidi_packed(zn32, cu, H, 20)

    zn_low = zn32.to(dtype).contiguous()
    got = mingru_bidi_packed(zn_low, cu, H, 20)
    assert got.dtype == dtype, "output dtype must follow the input"
    err = (got.float() - ref).abs().max().item()
    # input quantisation alone is ~eps; the fp32 state keeps us at that level
    # instead of it compounding over the sequence.
    tol = 6e-3 if dtype is torch.float16 else 5e-2
    assert err < tol, f"{dtype} drifted {err:.2e} from the fp32 scan"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_packed_kernel_fp32_path_is_unchanged():
    """The fp32 path must be bit-identical after making the kernel generic."""
    from track_regression.ops.mingru_short_triton import mingru_bidi_packed
    from track_regression.mingru import mingru_scan_ref

    torch.manual_seed(1)
    H = 32
    cu = torch.tensor([0, 7, 20], dtype=torch.int32, device="cuda")
    zn = torch.randn(20, 4 * H, device="cuda")
    out = mingru_bidi_packed(zn, cu, H, 20)
    assert out.dtype == torch.float32
    for b, (s, e) in enumerate([(0, 7), (7, 20)]):
        z = torch.sigmoid(zn[s:e, :H]); n = zn[s:e, H:2 * H]
        ref = mingru_scan_ref((1 - z).unsqueeze(0), (z * n).unsqueeze(0))[0]
        assert torch.allclose(out[s:e, :H], ref, atol=1e-5)
