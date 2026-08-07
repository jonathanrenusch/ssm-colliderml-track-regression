"""Correctness-oracle chain for the short-sequence SSD kernel campaign.

Chain:

- O1  stock-kernel vs fp64 reference  -> measures the stock TF32 noise floor
- O2  Mamba2Short fp64 vs independent fp64 reference -> pure-algebra check
- O3  Mamba2Short fp32 vs stock block -> tolerance calibrated by O1
- O4  V2' padded-static on the STOCK kernel vs packed stock (parity bridge)
- O5  state-dict key compatibility (strict load)
- O6  full encoder V3 (Mamba2Short, static path) vs packed stock
- O8  fp64 gradcheck of Mamba2Short
- O9  fused Triton kernel vs pure-torch quadratic dual (both IEEE fp32)
- O10 CLS grad-flow + track-order invariance on the static path
"""

from __future__ import annotations

import copy

import pytest
import torch

from track_regression.mamba_short import (
    Mamba2Short,
    apply_variant,
    mamba2_block_ref,
)

from tests.test_packed_equivalence import (  # noqa: E402  (shared oracles)
    _REQUIRES_MAMBA_GPU,
    _build_encoder,
    _build_padded_and_packed_inputs,
)

@pytest.fixture(autouse=True)
def _pin_matmul_precision():
    """Isolate the global fp32-matmul flag.

    The O7 golden check intentionally restores the artifact's production
    precision ("high" = TF32 linears) via build_model_and_batch — without
    this fixture that leaks into every later test and inflates
    IEEE-vs-IEEE comparisons by the TF32 noise floor (seen as in-suite-only
    failures of O9 and of the stock layout-invariance test).
    """
    saved = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    yield
    torch.set_float32_matmul_precision(saved if saved else "highest")


# Block shapes that exist in production: the 4L bench target and the
# paper-shape 10L encoder.  Mamba2Short must be shape-generic.
SHAPES = [
    {"d_model": 128, "d_state": 16, "headdim": 32},  # 4L target (H=8)
    {"d_model": 192, "d_state": 32, "headdim": 32},  # 10L paper shape (H=12)
]

VAR_LENGTHS = [6, 13, 20, 9, 1, 20]  # incl. min-ish and max tracks


def _mk_short(shape: dict, seed: int = 0, dtype=torch.float32, device="cpu") -> Mamba2Short:
    torch.manual_seed(seed)
    m = Mamba2Short(
        d_model=shape["d_model"],
        d_state=shape["d_state"],
        d_conv=4,
        expand=2,
        headdim=shape["headdim"],
        ngroups=1,
        chunk_size=16,
    )
    return m.to(device=device, dtype=dtype)


def _mk_stock(shape: dict, seed: int = 0, device="cuda"):
    from mamba_ssm import Mamba2

    torch.manual_seed(seed)
    m = Mamba2(
        d_model=shape["d_model"],
        d_state=shape["d_state"],
        d_conv=4,
        expand=2,
        headdim=shape["headdim"],
        ngroups=1,
        chunk_size=16,
    )
    return m.to(device)


def _stock_noise_floor(shape: dict, device: str = "cuda") -> float:
    """O1: |stock fp32 - fp64 reference| on identical weights/input."""
    stock = _mk_stock(shape, seed=1, device=device)
    short64 = _mk_short(shape, seed=2).to(device)
    short64.load_state_dict(stock.state_dict(), strict=True)
    short64 = short64.double()

    torch.manual_seed(3)
    u = torch.randn(4, 22, shape["d_model"], device=device, dtype=torch.float32)
    with torch.no_grad():
        y_stock = stock(u)
        y_ref = mamba2_block_ref(short64, u.double())
    return (y_stock.double() - y_ref).abs().max().item()


# ---------------------------------------------------------------------------
# O1 / O2 / O3 — block-level numerics
# ---------------------------------------------------------------------------


class TestBlockNumerics:
    @_REQUIRES_MAMBA_GPU
    @pytest.mark.parametrize("shape", SHAPES, ids=["dim128", "dim192"])
    def test_o1_stock_vs_fp64_reference_floor(self, shape):
        floor = _stock_noise_floor(shape)
        print(f"\nO1 stock-vs-fp64 noise floor ({shape}): {floor:.3e}")
        # Loose sanity bound only — this test *measures*, the gate is O3.
        assert floor < 1e-2

    @pytest.mark.parametrize("shape", SHAPES, ids=["dim128", "dim192"])
    def test_o2_short_fp64_vs_reference_fp64(self, shape):
        m = _mk_short(shape, seed=4, dtype=torch.float64)
        torch.manual_seed(5)
        u = torch.randn(3, 22, shape["d_model"], dtype=torch.float64)
        with torch.no_grad():
            y_fused = m(u)
            y_ref = mamba2_block_ref(m, u)
        diff = (y_fused - y_ref).abs().max().item()
        print(f"\nO2 fp64 fused-vs-ref ({shape}): {diff:.3e}")
        assert diff < 1e-10

    @_REQUIRES_MAMBA_GPU
    @pytest.mark.parametrize("shape", SHAPES, ids=["dim128", "dim192"])
    @pytest.mark.parametrize("L", [1, 7, 22], ids=lambda x: f"L{x}")
    def test_o3_short_fp32_vs_stock_block(self, shape, L):
        stock = _mk_stock(shape, seed=6, device="cuda")
        short = _mk_short(shape, seed=7).cuda()
        short.load_state_dict(stock.state_dict(), strict=True)

        floor = _stock_noise_floor(shape)
        tol = max(4.0 * floor, 1e-4)

        torch.manual_seed(8)
        u = torch.randn(5, L, shape["d_model"], device="cuda")
        with torch.no_grad():
            y_stock = stock(u)
            y_short = short(u)
        diff = (y_stock - y_short).abs().max().item()
        print(f"\nO3 fp32 short-vs-stock ({shape}, L={L}): {diff:.3e} (tol {tol:.3e})")
        assert diff < tol


# ---------------------------------------------------------------------------
# O4 — V2' parity bridge: static layout on the STOCK kernel == packed stock
# ---------------------------------------------------------------------------


def _encoder_pair(num_layers=2, dim=64):
    """Two weight-identical encoders on GPU (one to mutate, one reference)."""
    torch.manual_seed(42)
    enc_ref = _build_encoder(num_layers=num_layers, dim=dim).cuda().eval()
    enc_mut = _build_encoder(num_layers=num_layers, dim=dim).cuda().eval()
    enc_mut.load_state_dict(enc_ref.state_dict(), strict=True)
    return enc_ref, enc_mut


def _packed_inputs(dim=64, lengths=VAR_LENGTHS):
    _, packed, cu, seq_idx = _build_padded_and_packed_inputs(lengths, dim)
    return packed.cuda(), cu.cuda(), seq_idx.cuda()


class TestStaticPathParity:
    @_REQUIRES_MAMBA_GPU
    @pytest.mark.parametrize("variant,tol", [("v2p", 1e-4), ("v3", 1e-3)],
                             ids=["O4_v2p_stock", "O6_v3_short"])
    def test_static_vs_packed(self, variant, tol):
        enc_ref, enc_mut = _encoder_pair()
        packed, cu, seq_idx = _packed_inputs()

        with torch.no_grad():
            hits_ref, cls_ref = enc_ref(packed, seq_idx=seq_idx, cu_seqlens=cu)

        apply_variant(enc_mut, variant)
        with torch.no_grad():
            hits_new, cls_new = enc_mut(packed, seq_idx=seq_idx, cu_seqlens=cu)

        d_cls = (cls_ref - cls_new).abs().max().item()
        d_hits = (hits_ref - hits_new).abs().max().item()
        print(f"\n{variant} static-vs-packed: cls {d_cls:.3e}  hits {d_hits:.3e}")
        assert d_cls < tol and d_hits < tol

    @_REQUIRES_MAMBA_GPU
    def test_o10_track_order_invariance_static(self):
        """Reordering tracks in the packed batch permutes rows, nothing else."""
        enc_ref, enc_mut = _encoder_pair()
        apply_variant(enc_mut, "v3")

        lengths = [6, 13, 20, 9]
        packed, cu, seq_idx = _packed_inputs(lengths=lengths)
        with torch.no_grad():
            _, cls_a = enc_mut(packed, seq_idx=seq_idx, cu_seqlens=cu)

        perm = [2, 0, 3, 1]
        lengths_p = [lengths[i] for i in perm]
        # Rebuild the packed batch in permuted track order from the original.
        segs = [packed[0, cu[i] : cu[i + 1]] for i in range(len(lengths))]
        packed_p = torch.cat([segs[i] for i in perm], dim=0).unsqueeze(0)
        cu_p = torch.zeros(len(perm) + 1, dtype=torch.int32, device="cuda")
        cu_p[1:] = torch.cumsum(torch.tensor(lengths_p, device="cuda"), 0)
        seq_idx_p = torch.repeat_interleave(
            torch.arange(len(perm), device="cuda", dtype=torch.int32),
            torch.tensor(lengths_p, device="cuda"),
        ).unsqueeze(0)

        with torch.no_grad():
            _, cls_b = enc_mut(packed_p, seq_idx=seq_idx_p, cu_seqlens=cu_p)

        diff = (cls_a[perm] - cls_b).abs().max().item()
        assert diff < 1e-5

    @_REQUIRES_MAMBA_GPU
    def test_o10_cls_gradients_flow_static(self):
        _, enc = _encoder_pair()
        apply_variant(enc, "v3")
        packed, cu, seq_idx = _packed_inputs()
        _, cls = enc(packed, seq_idx=seq_idx, cu_seqlens=cu)
        cls.sum().backward()
        assert enc.cls_fwd.grad is not None and enc.cls_fwd.grad.abs().sum() > 0
        assert enc.cls_bwd.grad is not None and enc.cls_bwd.grad.abs().sum() > 0


# ---------------------------------------------------------------------------
# O5 — state-dict compatibility
# ---------------------------------------------------------------------------


class TestStateDict:
    @_REQUIRES_MAMBA_GPU
    def test_o5_strict_load_after_swap(self):
        enc_ref, enc_mut = _encoder_pair()
        sd = copy.deepcopy(enc_ref.state_dict())
        apply_variant(enc_mut, "v3")
        # Key sets identical -> a trained checkpoint loads unchanged.
        assert set(sd.keys()) == set(enc_mut.state_dict().keys())
        enc_mut.load_state_dict(sd, strict=True)
        for k, v in enc_mut.state_dict().items():
            assert torch.equal(v.cpu(), sd[k].cpu()), k


# ---------------------------------------------------------------------------
# O9 — V4 fused Triton kernel vs V3 (same algebra, both IEEE fp32)
# ---------------------------------------------------------------------------


class TestV4Kernel:
    @_REQUIRES_MAMBA_GPU
    @pytest.mark.parametrize("shape", SHAPES, ids=["dim128", "dim192"])
    @pytest.mark.parametrize("L", [1, 6, 13, 22], ids=lambda x: f"L{x}")
    def test_o9_block_v4_vs_v3(self, shape, L):
        torch.manual_seed(21)
        m = _mk_short(shape, seed=21).cuda()
        u = torch.randn(64, L, shape["d_model"], device="cuda")
        with torch.inference_mode():
            m.impl = "torch"
            y3 = m(u)
            m.impl = "triton"
            y4 = m(u)
        diff = (y3 - y4).abs().max().item()
        print(f"\nO9 v4-vs-v3 ({shape}, L={L}): {diff:.3e}")
        assert diff < 1e-5

    @_REQUIRES_MAMBA_GPU
    @pytest.mark.parametrize("variant", ["v4", "v5", "v5p", "v5pc", "auto"])
    def test_o9_encoder_kernel_vs_packed_stock(self, variant):
        enc_ref, enc_mut = _encoder_pair()
        packed, cu, seq_idx = _packed_inputs()
        with torch.no_grad():
            hits_ref, cls_ref = enc_ref(packed, seq_idx=seq_idx, cu_seqlens=cu)
        apply_variant(enc_mut, variant)
        with torch.no_grad():
            hits_new, cls_new = enc_mut(packed, seq_idx=seq_idx, cu_seqlens=cu)
        d_cls = (cls_ref - cls_new).abs().max().item()
        d_hits = (hits_ref - hits_new).abs().max().item()
        print(f"\n{variant} static-vs-packed: cls {d_cls:.3e}  hits {d_hits:.3e}")
        assert d_cls < 1e-3 and d_hits < 1e-3


# ---------------------------------------------------------------------------
# O11 — trainable v5pc: gradients of the fused packed path vs pure autograd
# ---------------------------------------------------------------------------


class TestV5pcTrainable:
    @_REQUIRES_MAMBA_GPU
    def test_o11_grads_v5p_vs_v3(self):
        """Backward through the Triton packed path == autograd through v3."""
        enc_a, enc_b = _encoder_pair()
        apply_variant(enc_a, "v3")   # pure autograd reference
        apply_variant(enc_b, "v5p")  # Triton fwd + recompute bwd
        packed, cu, seq_idx = _packed_inputs()

        # NOTE: the loss must NOT be (near-)invariant to the RMS-normalised
        # CLS outputs (cls**2.mean() is — true grads are ~1e-18 and any
        # comparison then measures cancellation noise; found the hard way).
        torch.manual_seed(7)
        probe = torch.randn(1, 128, device="cuda")

        grads = {}
        for tag, enc in (("v3", enc_a), ("v5p", enc_b)):
            enc.zero_grad(set_to_none=True)
            pk = packed.clone().requires_grad_(True)
            _, cls = enc(pk, seq_idx=seq_idx, cu_seqlens=cu)
            (cls * probe).sum().backward()
            grads[tag] = {
                "input": pk.grad.detach().clone(),
                "cls_fwd": enc.cls_fwd.grad.detach().clone(),
                "in_proj": enc.layers[0].forward_mamba.in_proj.weight.grad.detach().clone(),
                "conv": enc.layers[0].backward_mamba.conv1d.weight.grad.detach().clone(),
                "A_log": enc.final_layer.forward_mamba.A_log.grad.detach().clone(),
                "norm": enc.layers[0].forward_mamba.norm.weight.grad.detach().clone(),
            }
        for k in grads["v3"]:
            a, b = grads["v3"][k], grads["v5p"][k]
            scale = a.abs().max().clamp_min(1e-12)
            rel = (a - b).abs().max() / scale
            print(f"O11 grad {k}: rel {rel.item():.3e}")
            assert rel < 2e-3, k


# ---------------------------------------------------------------------------
# O8 — gradcheck (fp64, tiny shapes)
# ---------------------------------------------------------------------------


class TestGradcheck:
    def test_o8_gradcheck_fp64_input(self):
        torch.manual_seed(11)
        m = Mamba2Short(d_model=8, d_state=4, d_conv=4, expand=2, headdim=4).double()
        u = torch.randn(2, 5, 8, dtype=torch.float64, requires_grad=True)
        assert torch.autograd.gradcheck(lambda t: m(t), (u,), eps=1e-6, atol=1e-8)

    def test_o8_gradcheck_fp64_params(self):
        torch.manual_seed(12)
        m = Mamba2Short(d_model=8, d_state=4, d_conv=4, expand=2, headdim=4).double()
        u = torch.randn(2, 5, 8, dtype=torch.float64)

        def f(a_log, dt_bias):
            backup_a, backup_dt = m.A_log, m.dt_bias
            del m.A_log, m.dt_bias
            m.A_log, m.dt_bias = a_log, dt_bias
            out = m(u).sum()
            del m.A_log, m.dt_bias
            m.A_log, m.dt_bias = backup_a, backup_dt
            return out

        a = m.A_log.detach().clone().requires_grad_(True)
        d = m.dt_bias.detach().clone().requires_grad_(True)
        assert torch.autograd.gradcheck(f, (a, d), eps=1e-6, atol=1e-8)
