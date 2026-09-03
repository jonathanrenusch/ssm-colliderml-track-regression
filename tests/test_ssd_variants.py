"""Opt-in inference-kernel variants (CLAUDE.md §4.16) must match the exact torch reference.

Default path (no env vars) is bit-for-bit the previous IEEE single-launch kernel."""
import os
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for the Triton kernel")
from track_regression.ops.ssd_short_triton import ssd_short_fwd_packed, _packed_scan_torch_ref  # noqa: E402

H, P, N, DCONV = 8, 32, 64, 4
DPROJ = 2 * H * P + 2 * N + H


def _batch(n_tracks=3000, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    lens = torch.randint(6, 23, (n_tracks,), generator=g)                    # augmented lengths 8..24 incl. 2 CLS -> keep <= 32
    lens = lens + 2
    cu = torch.zeros(n_tracks + 1, dtype=torch.int64); cu[1:] = torch.cumsum(lens, 0)
    T = int(cu[-1])
    zx = (torch.randn(T, DPROJ, generator=g) * 0.5).cuda()
    conv_w = (torch.randn(H * P + 2 * N, 1, DCONV, generator=g) * 0.3).cuda()
    conv_b = (torch.randn(H * P + 2 * N, generator=g) * 0.1).cuda()
    dt_bias = (torch.randn(H, generator=g) * 0.5 - 2.0).cuda()
    A_log = (torch.randn(H, generator=g) * 0.3).cuda()
    D = torch.randn(H, generator=g).cuda()
    return zx, conv_w, conv_b, dt_bias, A_log, D, cu.cuda(), lens


def _run(env, reverse):
    old = {k: os.environ.get(k) for k in ("TRK_SSD_DOT_PRECISION", "TRK_SSD_BUCKET16")}
    for k in old: os.environ.pop(k, None)
    os.environ.update(env)
    try:
        zx, cw, cb, dtb, al, D, cu, lens = _batch()
        y = ssd_short_fwd_packed(zx, cw, cb, dtb, al, D, cu, H, P, N, reverse)
        ref = _packed_scan_torch_ref(zx, cw, cb, dtb, al, D, cu, H, P, N, reverse)
        return y, ref, lens
    finally:
        for k, v in old.items():
            os.environ.pop(k, None)
            if v is not None: os.environ[k] = v


@pytest.mark.parametrize("reverse", [False, True])
def test_default_ieee_matches_reference(reverse):
    y, ref, _ = _run({}, reverse)
    assert torch.isfinite(y).all()
    err = (y - ref).abs().max().item(); scale = ref.abs().max().item()
    assert err < 2e-4 * max(scale, 1.0), (err, scale)


@pytest.mark.parametrize("reverse", [False, True])
def test_bucket16_matches_reference(reverse):
    y, ref, lens = _run({"TRK_SSD_BUCKET16": "1"}, reverse)
    assert (lens <= 16).any() and (lens > 16).any()          # both buckets exercised
    err = (y - ref).abs().max().item(); scale = ref.abs().max().item()
    assert err < 2e-4 * max(scale, 1.0), (err, scale)


@pytest.mark.parametrize("reverse", [False, True])
def test_tf32_dots_within_tolerance(reverse):
    y, ref, _ = _run({"TRK_SSD_DOT_PRECISION": "tf32"}, reverse)
    rel = ((y - ref).abs().max() / ref.abs().max()).item()
    assert rel < 5e-3, rel                                    # TF32 mantissa (10 bits) -> ~1e-3 relative
    y32, _, _ = _run({}, reverse)
    assert ((y - y32).abs().max() / ref.abs().max()).item() > 1e-6   # the switch actually did something


def _batch2(n_tracks=3000, seed=1):
    """Two-direction batch for the merged kernel: fused rows + stacked weights."""
    zx_f, cw_f, cb_f, dtb_f, al_f, D_f, cu, lens = _batch(n_tracks, seed)
    g = torch.Generator(device="cpu").manual_seed(seed + 100)
    zx_b = (torch.randn(zx_f.shape[0], DPROJ, generator=g) * 0.5).cuda()
    cw_b = (torch.randn(H * P + 2 * N, 1, DCONV, generator=g) * 0.3).cuda()
    cb_b = (torch.randn(H * P + 2 * N, generator=g) * 0.1).cuda()
    dtb_b = (torch.randn(H, generator=g) * 0.5 - 2.0).cuda()
    al_b = (torch.randn(H, generator=g) * 0.3).cuda()
    D_b = torch.randn(H, generator=g).cuda()
    zx_fb = torch.cat([zx_f, zx_b], dim=1).contiguous()          # (T, 2*DPROJ)
    stk = lambda a, b: torch.stack([a, b]).contiguous()          # noqa: E731
    cw2 = stk(cw_f.reshape(cw_f.shape[0], -1), cw_b.reshape(cw_b.shape[0], -1))
    return (zx_fb, cw2, stk(cb_f, cb_b), stk(dtb_f, dtb_b), stk(al_f, al_b), stk(D_f, D_b),
            cu, (zx_f, cw_f, cb_f, dtb_f, al_f, D_f), (zx_b, cw_b, cb_b, dtb_b, al_b, D_b), lens)


@pytest.mark.parametrize("bucket16", [False, True])
def test_merged_bidi_matches_reference(bucket16):
    """TRK_SSD_MERGED_BIDI: one launch, direction 0 = forward scan of the fwd
    rows, direction 1 = REVERSED scan of the bwd rows — each must equal the
    torch reference of the corresponding single-direction call."""
    from track_regression.ops.ssd_short_triton import ssd_short_fwd_packed_merged

    old = os.environ.get("TRK_SSD_BUCKET16")
    os.environ.pop("TRK_SSD_BUCKET16", None)
    if bucket16:
        os.environ["TRK_SSD_BUCKET16"] = "1"
    try:
        zx_fb, cw2, cb2, dtb2, al2, D2, cu, fwd, bwd, lens = _batch2()
        if bucket16:
            assert (lens <= 16).any() and (lens > 16).any()
        y2 = ssd_short_fwd_packed_merged(zx_fb, cw2, cb2, dtb2, al2, D2, cu, H, P, N)
        ref_f = _packed_scan_torch_ref(*fwd, cu, H, P, N, False)
        ref_b = _packed_scan_torch_ref(*bwd, cu, H, P, N, True)
        for y, ref, tag in ((y2[0], ref_f, "fwd"), (y2[1], ref_b, "bwd")):
            err = (y - ref).abs().max().item(); scale = ref.abs().max().item()
            assert err < 2e-4 * max(scale, 1.0), (tag, err, scale)
    finally:
        os.environ.pop("TRK_SSD_BUCKET16", None)
        if old is not None:
            os.environ["TRK_SSD_BUCKET16"] = old


def test_merged_bidi_full_layer_matches_default():
    """End-to-end: fused_bidi_scan_packed with TRK_SSD_MERGED_BIDI=1 equals the
    default two-launch path on a randomly initialised bidirectional layer."""
    import torch.nn as nn
    from track_regression.mamba_short import Mamba2Short, fused_bidi_scan_packed

    torch.manual_seed(0)
    layer = nn.Module()
    layer.forward_mamba = Mamba2Short(d_model=128, d_state=N, d_conv=DCONV, headdim=P).cuda().float()
    layer.backward_mamba = Mamba2Short(d_model=128, d_state=N, d_conv=DCONV, headdim=P).cuda().float()
    g = torch.Generator(device="cpu").manual_seed(2)
    lens = (torch.randint(6, 23, (512,), generator=g) + 2)
    cu = torch.zeros(513, dtype=torch.int64); cu[1:] = torch.cumsum(lens, 0)
    x = torch.randn(1, int(cu[-1]), 128, generator=g).cuda()
    cu = cu.cuda()
    with torch.inference_mode():
        f0, b0 = fused_bidi_scan_packed(layer, x, cu)
        os.environ["TRK_SSD_MERGED_BIDI"] = "1"
        try:
            f1, b1 = fused_bidi_scan_packed(layer, x, cu)
        finally:
            os.environ.pop("TRK_SSD_MERGED_BIDI", None)
    for a, b, tag in ((f0, f1, "fwd"), (b0, b1, "bwd")):
        err = (a - b).abs().max().item(); scale = a.abs().max().item()
        assert err < 2e-4 * max(scale, 1.0), (tag, err, scale)
