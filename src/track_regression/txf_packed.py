"""Packed inference path for the ablation transformer (opt-in, inference only).

What the padded path does today (``_PaddedRoutedTransformerCLS`` ->
``EncoderWithCLS.forward`` -> ``_lib.encoder.Encoder``): scatter the packed
batch to ``(B, L_max, D)``, add the index positional encoding, argsort by a
key that is already sorted, prepend two class tokens, and run every RMSNorm,
projection and feed-forward on ``B x (L_max + 2)`` rows -- of which only
13.3 + 2 in 22 are real -- with dense SDPA over a materialised boolean mask.
Attention is 2.8 % of the layer's FLOPs; the padding lands on the 97 % that is
dense GEMM.

This module runs the SAME function on the packed stream:

* one augmented stream of ``T + 2B`` rows (hits with their class tokens
  interleaved per track, built once with two index writes, not an argsort);
* every RMSNorm / GEMM / SiLU / LayerScale / residual on real rows only,
  with the non-GEMM glue fused by ``torch.compile``;
* attention by :func:`~track_regression.ops.attn_short_triton.attn_packed_tracks`,
  one program per track, with the model's q/k/v RMSNorm folded in;
* the class-token readout gathered from the known augmented positions.

Numerics: the residual stream stays fp32; the GEMMs run in the encoder
autocast dtype (fp32 / TF32 / fp16, exactly like the padded path under
``torch.amp.autocast``); the norms run in fp32 with ``torch.finfo(fp32).eps``,
which is what the strict-fp32 training path used.  Parity with the padded path
is locked in by ``tests/test_txf_packed.py``.

Enable with ``TRK_TXF_PACKED=1`` (inference only, CUDA only, packed batches
only); the default leaves every existing path untouched.
"""

from __future__ import annotations

import math
import os

import torch
import torch.nn.functional as F
from torch import Tensor

from track_regression.ops.attn_short_triton import (
    attn_packed_tracks, add_rmsnorm_packed, gemm_epilogue_fp16)

_FP32_EPS = torch.finfo(torch.float32).eps


def _rms(x: Tensor, w: Tensor) -> Tensor:
    """RMSNorm in fp32 with the fp32 default eps (matches ``nn.RMSNorm`` on
    an fp32 input, which is what the training path evaluated)."""
    x = x.float()
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + _FP32_EPS) * w


def _encoder_body(x: Tensor, cu_aug: Tensor, params: list[Tensor],
                  num_layers: int, num_heads: int, gemm_dtype: torch.dtype,
                  qkv_norm: bool, max_len: int, resid_dtype: torch.dtype,
                  fused_norm: bool, fused_gemm: bool) -> Tensor:
    """The three-layer body on the augmented packed stream.  ``params`` is the
    flat list produced by :func:`_collect_params` (compile-friendly).

    ``resid_dtype``: dtype of the residual stream between layers (fp32
    reproduces the padded autocast path; fp16 is opt-in,
    ``TRK_TXF_PACKED_RESID16=1``).  ``fused_norm`` routes every residual-add
    + RMSNorm through the single-pass Triton op (``TRK_TXF_PACKED_FUSED_NORM=1``).
    ``fused_gemm`` (fp16 only, ``TRK_TXF_PACKED_FUSED_GEMM=1``) runs the
    out-projection and both feed-forward GEMMs through the Triton GEMM with the
    bias / SiLU / LayerScale-residual / RMSNorm epilogues fused in, so a layer is
    five kernels: QKV GEMM, attention, out-proj(+res+norm), FFN1(+SiLU),
    FFN2(+res+norm)."""
    x = x.to(resid_dtype)
    eps = _FP32_EPS
    fused_gemm = fused_gemm and gemm_dtype == torch.float16
    fused_norm = fused_norm or fused_gemm
    h = None
    if fused_norm:
        _, h = add_rmsnorm_packed(x, x, params[0], params[0], eps, False, True, gemm_dtype)
    for l in range(num_layers):
        (w_an, w_in, b_in, wq, wk, wv, w_o, b_o, g_a,
         w_dn, w_1, b_1, w_2, b_2, g_d) = params[15 * l:15 * (l + 1)]
        last = l == num_layers - 1
        w_next = w_dn if last else params[15 * (l + 1)]
        if not fused_norm:
            h = _rms(x, w_an).to(gemm_dtype)
        qkv = F.linear(h, w_in, b_in)
        a = attn_packed_tracks(qkv, cu_aug, wq, wk, wv, num_heads, eps,
                               qkv_norm, max_len + 2)
        if fused_gemm:
            _, x, h2 = gemm_epilogue_fp16(a, w_o, b_o, x, g_a, w_dn, eps, 2, True)
            f, _, _ = gemm_epilogue_fp16(h2, w_1, b_1, x, g_a, w_dn, eps, 1, False)
            _, x, h = gemm_epilogue_fp16(f, w_2, b_2, x, g_d, w_next, eps, 2, not last)
            continue
        o = F.linear(a, w_o, b_o)
        if fused_norm:
            x, h2 = add_rmsnorm_packed(x, o, g_a, w_dn, eps, True, True, gemm_dtype)
        else:
            x = (x.float() + g_a * o.float()).to(resid_dtype)
            h2 = _rms(x, w_dn).to(gemm_dtype)
        f = F.silu(F.linear(h2, w_1, b_1))
        o2 = F.linear(f, w_2, b_2)
        if fused_norm:
            x, h = add_rmsnorm_packed(x, o2, g_d, w_next, eps, True, not last, gemm_dtype)
        else:
            x = (x.float() + g_d * o2.float()).to(resid_dtype)
    return x


def _collect_params(enc, gemm_dtype: torch.dtype) -> list[Tensor]:
    """Flatten the encoder's weights, GEMM weights pre-cast to ``gemm_dtype``
    (done once per dtype and cached on the module -- inference only)."""
    out: list[Tensor] = []
    g = lambda t: t.detach().to(gemm_dtype).contiguous()  # noqa: E731
    f32 = lambda t: t.detach().float().contiguous()       # noqa: E731
    for layer in enc.encoder.layers:
        attn = layer.attn.fn
        dense = layer.dense.fn.net
        out += [
            f32(layer.attn.norm.weight),
            g(attn.in_proj_weight), g(attn.in_proj_bias),
            f32(attn.q_norm.weight) if attn.qkv_norm else f32(layer.attn.norm.weight),
            f32(attn.k_norm.weight) if attn.qkv_norm else f32(layer.attn.norm.weight),
            f32(attn.v_norm.weight) if attn.qkv_norm else f32(layer.attn.norm.weight),
            g(attn.out_proj.weight), g(attn.out_proj.bias),
            f32(layer.attn.ls.gamma),
            f32(layer.dense.norm.weight),
            g(dense[0].weight), g(dense[0].bias),
            g(dense[2].weight), g(dense[2].bias),
            f32(layer.dense.ls.gamma),
        ]
    return out


def _check_supported(enc) -> None:
    inner = enc.encoder
    if inner.attn_type != "torch":
        raise NotImplementedError(f"packed path expects attn_type='torch', got {inner.attn_type!r}")
    if inner.value_residual or inner.window_size or inner.score_mod is not None:
        raise NotImplementedError("packed path: value_residual / window / score_mod unsupported")
    for layer in inner.layers:
        if not isinstance(layer.attn.norm, torch.nn.RMSNorm) or not isinstance(layer.dense.norm, torch.nn.RMSNorm):
            raise NotImplementedError("packed path expects RMSNorm pre-norm layers")
        if layer.attn.post_norm or layer.dense.post_norm:
            raise NotImplementedError("packed path: hybrid post-norm unsupported")
        if not isinstance(layer.attn.ls, torch.nn.Module) or not hasattr(layer.attn.ls, "gamma"):
            raise NotImplementedError("packed path expects LayerScale residuals")
        attn = layer.attn.fn
        if attn.value_residual or attn.in_proj_bias is None:
            raise NotImplementedError("packed path: value_residual / bias-free in_proj unsupported")
        if attn.qkv_norm and not isinstance(attn.q_norm, torch.nn.RMSNorm):
            raise NotImplementedError("packed path expects RMSNorm qkv_norm")
        net = layer.dense.fn.net
        if len(net) != 3 or not isinstance(net[1], torch.nn.SiLU):
            raise NotImplementedError("packed path expects Linear-SiLU-Linear feed-forward")
    if not isinstance(enc.final_norm, torch.nn.RMSNorm):
        raise NotImplementedError("packed path expects an RMSNorm final norm")


def packed_transformer_forward(enc, x: Tensor, seq_idx: Tensor | None,
                               cu_seqlens: Tensor, max_len: int = 20) -> tuple[Tensor, Tensor]:
    """Inference forward of ``_PaddedRoutedTransformerCLS`` on the packed
    stream.  ``x``: (1, T, D) embedded hits; returns ``(x_aug (1, T+2B, D),
    pooled (B, 2D))`` -- the sequence output is the augmented stream (the
    regressor does not consume it)."""
    if not getattr(enc, "_packed_checked", False):
        _check_supported(enc)
        enc._packed_checked = True
    D = enc.dim
    K = enc.num_cls_tokens
    cu = cu_seqlens.to(torch.long)
    B = cu.numel() - 1
    T = x.shape[1]
    device = x.device
    ar = torch.arange(T, device=device)
    if seq_idx is None:
        seg = torch.bucketize(ar, cu[1:], right=True)
    else:
        seg = seq_idx[0].to(torch.long)
    pos = (ar - cu[seg])                                     # within-track index

    # dtype of the GEMMs = the encoder autocast dtype the model runs under
    gemm_dtype = torch.get_autocast_dtype("cuda") if torch.is_autocast_enabled("cuda") else torch.float32
    cache = getattr(enc, "_packed_param_cache", None)
    if cache is None or cache[0] != gemm_dtype or cache[1] is not enc.encoder.layers[0].attn.fn.in_proj_weight:
        params = _collect_params(enc, gemm_dtype)
        enc._packed_param_cache = (gemm_dtype, enc.encoder.layers[0].attn.fn.in_proj_weight, params)
    params = enc._packed_param_cache[2]
    body = getattr(enc, "_packed_body", None)
    if body is None:
        if os.environ.get("TRK_TXF_PACKED_COMPILE", "1") == "1":
            mode = os.environ.get("TRK_TXF_PACKED_COMPILE_MODE", "default")
            body = torch.compile(_encoder_body, dynamic=True,
                                 mode=None if mode == "default" else mode)
        else:
            body = _encoder_body
        enc._packed_body = body
    resid_dtype = (torch.float16 if os.environ.get("TRK_TXF_PACKED_RESID16", "0") == "1"
                   and gemm_dtype == torch.float16 else torch.float32)

    with torch.autocast("cuda", enabled=False):
        xh = x[0].float()
        # index positional encoding (the IndexPosEnc arm): Fourier ladder of the
        # within-track position, projected to D and added -- identical to the
        # padded path's ``_apply_posenc`` with key = arange(S).  The key takes
        # only the values 0..max_len-1, so the encoding is a (max_len, D) table
        # computed once (the same arithmetic, once per position instead of once
        # per hit) and gathered -- one kernel instead of 2*scales sin/cos, a
        # 22-way cat and a GEMM per forward.
        if enc.posenc_proj is not None:
            tab = getattr(enc, "_packed_posenc_table", None)
            if tab is None or tab[0] is not enc.posenc_proj.weight or tab[1].device != device:
                t = (torch.arange(max_len, device=device, dtype=torch.float32)
                     / enc.posenc_time_scale).unsqueeze(-1)
                base = float(enc.posenc_fourier_base)
                feats = [torch.sin(t / base ** n) for n in enc.posenc_fourier_scales] + \
                        [torch.cos(t / base ** n) for n in enc.posenc_fourier_scales]
                table = F.linear(torch.cat(feats, -1), enc.posenc_proj.weight.float(),
                                 enc.posenc_proj.bias.float()).contiguous()
                enc._packed_posenc_table = (enc.posenc_proj.weight, table)
            xh = xh + enc._packed_posenc_table[1][pos]
        # augmented stream: [cls_0, cls_1, hits...] per track
        cu_aug = (cu + K * torch.arange(B + 1, device=device)).to(torch.int32)
        hit_pos = ar + (seg + 1) * K
        cls_pos = (cu[:-1] + K * torch.arange(B, device=device)).unsqueeze(1) + \
                  torch.arange(K, device=device).unsqueeze(0)          # (B, K)
        x_aug = torch.empty(T + K * B, D, device=device, dtype=torch.float32)
        x_aug[hit_pos] = xh
        x_aug[cls_pos.reshape(-1)] = enc.cls_token[0].float().repeat(B, 1)

        y = body(x_aug, cu_aug, params, enc.encoder.num_layers,
                 enc.encoder.layers[0].attn.fn.num_heads, gemm_dtype,
                 bool(enc.encoder.layers[0].attn.fn.qkv_norm), int(max_len),
                 resid_dtype, os.environ.get("TRK_TXF_PACKED_FUSED_NORM", "0") == "1",
                 os.environ.get("TRK_TXF_PACKED_FUSED_GEMM", "0") == "1")
        cls = _rms(y[cls_pos.reshape(-1)], enc.final_norm.weight.float())
        pooled = cls.view(B, K * D)
    return y.unsqueeze(0), pooled
