"""Largely inspired by / copied from https://github.com/samvanstroud/hepattn."""

from track_regression._lib.flex.relative_position import relative_position, relative_position_wrapped
from track_regression._lib.flex.sliding_window import sliding_window_mask, sliding_window_mask_wrapped

__all__ = ["relative_position", "relative_position_wrapped", "sliding_window_mask", "sliding_window_mask_wrapped"]
