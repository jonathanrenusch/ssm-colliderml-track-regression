from torch import Tensor
from torch.nn.attention.flex_attention import _score_mod_signature


def identity(score, b, h, q_idx, kv_idx):  # noqa: ARG001
    return score


def relative_position(score, b, h, q_idx, kv_idx):  # noqa: ARG001
    return score - (q_idx - kv_idx).abs()


def relative_position_wrapped(q_len: Tensor) -> _score_mod_signature:
    def score_mod(score, b, h, q_idx, kv_idx) -> Tensor:  # noqa: ARG001
        raw_diff = q_idx - kv_idx
        abs_diff = abs(raw_diff)
        is_second_half = abs_diff > q_len[0] // 2
        cyclic_dist = abs_diff * ~is_second_half + (q_len[0] - abs_diff) * is_second_half
        return score - cyclic_dist

    return score_mod


def absolute_positional(coords: Tensor) -> _score_mod_signature:
    def score_mod(score, b, h, q_idx, kv_idx) -> Tensor:  # noqa: ARG001
        return score - (coords[q_idx] - coords[kv_idx]).abs()

    return score_mod


def absolute_positional_wrapped(coords: Tensor, max_value: float) -> _score_mod_signature:
    def score_mod(score, b, h, q_idx, kv_idx) -> Tensor:  # noqa: ARG001
        raw_diff = coords[q_idx] - coords[kv_idx]
        abs_diff = abs(raw_diff)
        is_second_half = abs_diff > max_value / 2
        cyclic_dist = abs_diff * ~is_second_half + (max_value - abs_diff) * is_second_half
        return score - cyclic_dist

    return score_mod
