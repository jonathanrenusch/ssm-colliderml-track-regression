"""Largely inspired by / copied from https://github.com/samvanstroud/hepattn."""

from torch import nn


def get_hybrid_norm_config(norm: str | None, depth: int, hybrid_norm: bool, qkv_norm: bool) -> tuple[str | None, bool, bool]:
    """Get the normalization configuration for HybridNorm.

    Args:
        norm: The normalization type.
        depth: The layer depth.
        hybrid_norm: Whether to use HybridNorm.
        qkv_norm: Whether to use QKV normalization.

    Returns:
        attn_norm: The normalization to use before attention.
        dense_post_norm: Whether to use post-normalization for the dense layer.
        qkv_norm: Whether to use QKV normalization.
    """
    qkv_norm = qkv_norm or hybrid_norm
    is_hybrid_subsequent = hybrid_norm and depth > 0
    attn_norm = None if is_hybrid_subsequent else norm
    dense_post_norm = is_hybrid_subsequent

    return attn_norm, dense_post_norm, qkv_norm


# Mapping of normalization type strings to their corresponding nn.Module classes.
NORM_TYPES: dict[str, type[nn.Module]] = {
    "LayerNorm": nn.LayerNorm,
    "RMSNorm": nn.RMSNorm,
}
