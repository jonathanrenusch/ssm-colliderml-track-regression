"""Fused packed inference path of :class:`~track_regression.transformer.TransformerCLSEncoder`.

The padded path scatters the packed batch to ``(B, L_max + 2, D)`` and runs
every norm, projection and feed-forward on the padded rows (about 15 of 22
are real).  This module runs the same function on the packed stream:

* one augmented stream of ``T + 2B`` rows (the two class tokens interleaved
  per track with two index writes);
* every RMSNorm / GEMM / SiLU / LayerScale / residual on real rows only;
* attention by :func:`~track_regression.ops.attn_short_triton.attn_packed_tracks`,
  one program per track, with the q/k/v RMSNorm folded in;
* residual-add + RMSNorm in one pass, and under fp16 the out-projection and
  both feed-forward GEMMs with their bias / SiLU / LayerScale-residual /
  RMSNorm epilogues fused (a layer is then five kernels).

Numerics: the residual stream stays fp32; the GEMMs run in the encoder
autocast dtype; the norms use ``torch.finfo(fp32).eps`` like ``nn.RMSNorm`` on
fp32 inputs.  Parity with the padded path: ``tests/test_txf_packed.py``.
"""

from __future__ import annotations

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
    reproduces the padded autocast path).  ``fused_norm`` routes every
    residual-add + RMSNorm through the single-pass Triton op.
    ``fused_gemm`` (fp16 only) runs the
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


def packed_transformer_forward(enc, x: Tensor, seq_idx: Tensor | None,
                               cu_seqlens: Tensor, max_len: int = 20) -> tuple[Tensor, Tensor]:
    """Inference forward of ``TransformerCLSEncoder`` on the packed
    stream.  ``x``: (1, T, D) embedded hits; returns ``(x_aug (1, T+2B, D),
    pooled (B, 2D))`` -- the sequence output is the augmented stream (the
    regressor does not consume it)."""
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
        body = torch.compile(_encoder_body, dynamic=True)
        enc._packed_body = body

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
                 torch.float32, True, True)
        cls = _rms(y[cls_pos.reshape(-1)], enc.final_norm.weight.float())
        pooled = cls.view(B, K * D)
    return y.unsqueeze(0), pooled
