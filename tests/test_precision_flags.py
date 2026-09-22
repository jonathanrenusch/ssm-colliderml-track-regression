"""Opt-in inference precision flags (docs/PRECISION_STUDY_2026-09-21.md).

Defaults (env unset) must leave the model and the loss untouched; each flag
must produce finite outputs within the precision it promises.
"""
import os

import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

from track_regression.losses import QuantileLoss, _ladder_prefix_sum  # noqa: E402
from track_regression.model import _env_dtype  # noqa: E402


def _with_env(name, value):
    class _Ctx:
        def __enter__(self):
            self.old = os.environ.get(name)
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        def __exit__(self, *a):
            os.environ.pop(name, None)
            if self.old is not None:
                os.environ[name] = self.old
    return _Ctx()


def test_env_dtype_parsing():
    with _with_env("TRK_X", None):
        assert _env_dtype("TRK_X") is None
    with _with_env("TRK_X", "float32"):
        assert _env_dtype("TRK_X") is None
    with _with_env("TRK_X", "float16"):
        assert _env_dtype("TRK_X") is torch.float16
    with _with_env("TRK_X", "bfloat16"):
        assert _env_dtype("TRK_X") is torch.bfloat16
    with _with_env("TRK_X", "int8"), pytest.raises(ValueError):
        _env_dtype("TRK_X")


def test_ladder_matmul_is_the_exact_sum():
    """TRK_QUANTILE_LADDER=matmul: the fp64 triangular product IS the exact prefix
    sum (to fp64), its fp32 result is within 1 ulp of the correctly rounded sum,
    it is never farther from the exact sum than torch.cumsum's sequential fp32
    rounding is, and the ladder ordering is preserved."""
    torch.manual_seed(0)
    loss = QuantileLoss(quantiles=[0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]).cuda()
    raw = torch.randn(200_000, 7, device="cuda") * 3
    base = raw[:, :1]
    deltas = torch.nn.functional.softplus(raw[:, 1:]) + loss.monotone_eps
    exact = torch.cumsum(deltas.double(), -1)                    # exact to fp64
    with _with_env("TRK_QUANTILE_LADDER", None):
        ref = loss._ordered_from_raw(raw)
    with _with_env("TRK_QUANTILE_LADDER", "matmul"):
        got = loss._ordered_from_raw(raw)
        p64 = _ladder_prefix_sum(deltas.double(), loss)
        p32 = _ladder_prefix_sum(deltas, loss)
    assert ((p64 - exact).abs() <= 4 * torch.finfo(torch.float64).eps * exact).all()
    # one rounding of the exact sum: |p32 - exact| <= 0.5 ulp(exact) (+ fp64 GEMM noise)
    ulp32 = torch.finfo(torch.float32).eps * exact
    assert ((p32.double() - exact).abs() <= 0.5001 * ulp32).all(), \
        ((p32.double() - exact).abs() / ulp32).max().item()
    # the full ladder: matmul path never farther from the exact value than the cumsum path
    exact_l = base.double() + exact
    err_got = (got[:, 1:].double() - exact_l).abs()
    err_ref = (ref[:, 1:].double() - exact_l).abs()
    assert got.dtype == ref.dtype == torch.float32
    assert err_got.max() <= err_ref.max()
    assert (err_got <= 2 * torch.finfo(torch.float32).eps * (base.abs().double() + exact)).all()
    assert (got[:, 1:] > got[:, :-1]).all()
    assert torch.equal(got[:, 0], ref[:, 0])


def test_ladder_default_is_cumsum_and_cpu_untouched():
    loss = QuantileLoss()
    raw = torch.randn(64, 5)
    with _with_env("TRK_QUANTILE_LADDER", "matmul"):
        out = loss._ordered_from_raw(raw)          # CPU: always the cumsum path
    base = raw[:, :1]; d = torch.nn.functional.softplus(raw[:, 1:]) + loss.monotone_eps
    assert torch.equal(out, torch.cat([base, base + torch.cumsum(d, -1)], -1))
