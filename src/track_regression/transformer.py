"""Transformer encoder baseline (pre-norm, two class tokens, hit-index positional encoding).

Structure, per layer: ``x + LayerScale(Attention(RMSNorm(x)))`` followed by
``x + LayerScale(FFN(RMSNorm(x)))``, with RMSNorm on the projected queries,
keys and values.  The within-track hit index ``0, 1, 2, ...`` (the position in
the stored hit order) is Fourier-encoded, projected to ``dim`` and added to
the embedded hits; two learned class tokens are prepended to every track and
their final (RMS-normalised) states form the pooled output.

Training and the reference inference path scatter the packed batch into a
padded ``(B, L_max + 2, D)`` layout and run dense SDPA with a key mask, so
attention stays inside each track.  The fused inference path
(:mod:`track_regression.txf_packed`) evaluates the same function on the
packed stream with Triton kernels.

The layer/submodule layout follows the ``hepattn`` package
(https://github.com/samvanstroud/hepattn), from which it was adapted.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from track_regression.dense import Dense
from track_regression.layout import fused_kernels_enabled, packed_to_padded


class LayerScale(nn.Module):
    """Learned per-channel residual scale (https://arxiv.org/abs/2103.17239)."""

    def __init__(self, dim: int, init_value: float = 1e-5) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim) * init_value)

    def forward(self, x: Tensor) -> Tensor:
        return x * self.gamma


class Residual(nn.Module):
    """``x + ls(fn(norm(x)))``."""

    def __init__(self, fn: nn.Module, dim: int, layer_scale: float) -> None:
        super().__init__()
        self.fn = fn
        self.ls = LayerScale(dim, layer_scale)
        self.norm = nn.RMSNorm(dim)

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        return x + self.ls(self.fn(self.norm(x), **kwargs))


class Attention(nn.Module):
    """Multi-head self-attention with RMSNorm on q, k and v."""

    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        self.dim, self.num_heads = dim, num_heads
        self.qkv_norm = True
        self.in_proj_weight = nn.Parameter(torch.empty(3 * dim, dim))
        self.in_proj_bias = nn.Parameter(torch.empty(3 * dim))
        self.out_proj = nn.Linear(dim, dim)
        self.q_norm, self.k_norm, self.v_norm = nn.RMSNorm(dim), nn.RMSNorm(dim), nn.RMSNorm(dim)
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.constant_(self.in_proj_bias, 0.0)

    def forward(self, x: Tensor, kv_mask: Tensor) -> Tensor:
        q, k, v = F.linear(x, self.in_proj_weight, self.in_proj_bias).chunk(3, dim=-1)
        q, k, v = self.q_norm(q), self.k_norm(k), self.v_norm(v)
        heads = lambda t: t.unflatten(-1, (self.num_heads, -1)).transpose(-3, -2)  # noqa: E731
        # every query attends only to the valid keys of its own track
        mask = kv_mask.unsqueeze(-2).expand(-1, x.shape[1], -1).unsqueeze(-3)
        out = F.scaled_dot_product_attention(heads(q), heads(k), heads(v), attn_mask=mask)
        return self.out_proj(out.transpose(-3, -2).flatten(-2))


class EncoderLayer(nn.Module):
    def __init__(self, dim: int, num_heads: int, layer_scale: float, hidden_dim_scale: int) -> None:
        super().__init__()
        self.attn = Residual(Attention(dim, num_heads), dim, layer_scale)
        self.dense = Residual(Dense(dim, hidden_dim_scale=hidden_dim_scale), dim, layer_scale)

    def forward(self, x: Tensor, kv_mask: Tensor) -> Tensor:
        return self.dense(self.attn(x, kv_mask=kv_mask))


class _Layers(nn.Module):
    def __init__(self, num_layers: int, **layer_kwargs) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList([EncoderLayer(**layer_kwargs) for _ in range(num_layers)])


class TransformerCLSEncoder(nn.Module):
    """Transformer encoder returning ``(hit_output, pooled (B, num_cls_tokens * dim))``."""

    def __init__(
        self,
        dim: int,
        num_layers: int = 3,
        num_heads: int = 4,
        hidden_dim_scale: int = 3,
        layer_scale: float = 1e-5,
        num_cls_tokens: int = 2,
        cls_init_scale: float = 0.02,
        posenc_fourier_scales: list[int] = (-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5),
        posenc_fourier_base: int = 2,
        posenc_init_scale: float = 0.02,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.num_cls_tokens = int(num_cls_tokens)
        self.encoder = _Layers(num_layers, dim=dim, num_heads=num_heads,
                               layer_scale=layer_scale, hidden_dim_scale=hidden_dim_scale)
        self.cls_token = nn.Parameter(torch.randn(1, self.num_cls_tokens, dim) * cls_init_scale)
        self.final_norm = nn.RMSNorm(dim)
        self.posenc_fourier_scales = list(posenc_fourier_scales)
        self.posenc_fourier_base = int(posenc_fourier_base)
        self.posenc_time_scale = 1.0
        self.posenc_proj = nn.Linear(2 * len(self.posenc_fourier_scales), dim, bias=True)
        with torch.no_grad():
            self.posenc_proj.weight.mul_(posenc_init_scale)
            self.posenc_proj.bias.zero_()

    @property
    def pool_dim(self) -> int:
        return self.num_cls_tokens * self.dim

    def _posenc(self, n: int, device, dtype) -> Tensor:
        """Projected Fourier encoding of the hit index, ``(n, dim)``."""
        t = torch.arange(n, device=device, dtype=dtype).unsqueeze(-1) / self.posenc_time_scale
        b = self.posenc_fourier_base
        pe = torch.cat([torch.sin(t / b ** s) for s in self.posenc_fourier_scales]
                       + [torch.cos(t / b ** s) for s in self.posenc_fourier_scales], dim=-1)
        return self.posenc_proj(pe)

    def forward(self, x: Tensor, cu_seqlens: Tensor, seq_idx: Tensor | None = None):
        if x.is_cuda and not torch.is_grad_enabled() and fused_kernels_enabled():
            from track_regression.txf_packed import packed_transformer_forward
            return packed_transformer_forward(self, x, seq_idx, cu_seqlens)

        x_pad, row, pos, lens = packed_to_padded(x, cu_seqlens)
        B, S = x_pad.shape[0], x_pad.shape[1]
        mask = torch.zeros(B, S, dtype=torch.bool, device=x.device)
        mask[row, pos] = True
        x_pad = x_pad + self._posenc(S, x.device, x_pad.dtype)
        K = self.num_cls_tokens
        h = torch.cat([self.cls_token.expand(B, -1, -1).to(x_pad.dtype), x_pad], dim=1)
        mask = torch.cat([mask.new_ones(B, K), mask], dim=1)
        for layer in self.encoder.layers:
            h = layer(h, kv_mask=mask)
        h = self.final_norm(h)
        pooled = h[:, :K, :].reshape(B, K * self.dim)
        hit_out = h[:, K:, :][row, pos].unsqueeze(0)
        if self.training:
            # keep every parameter in the autograd graph (DDP), numerically a no-op
            pooled = pooled + 0.0 * hit_out.float().sum()
        return hit_out, pooled
