"""Seed-guided bidirectional sequence models for charged-particle track fitting."""

import os

# torch.compile's parallel compile-worker pool can block interpreter exit for
# minutes; compile in-process instead (numerically irrelevant).
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")
