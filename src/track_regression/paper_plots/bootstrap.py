"""Bootstrap σ for scalar metrics on residual arrays."""
from __future__ import annotations

from typing import Callable

import numpy as np


def bootstrap_metric(
    values: np.ndarray,
    fn: Callable[[np.ndarray], float],
    n: int = 200,
    seed: int = 0,
) -> tuple[float, float]:
    """Return (mean, SE_boot) of `fn` evaluated on `n` bootstrap resamples.

    ``mean`` is the mean of the bootstrap distribution and ``SE_boot`` is its
    standard deviation (= bootstrap standard error). ``mean ± 2·SE_boot`` is a
    normal-approximation 95 % CI; for the percentile-bootstrap CI take the
    2.5/97.5 percentiles of the bootstrap distribution instead.
    """
    if len(values) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    n_total = len(values)
    samples = np.empty(n)
    for i in range(n):
        idx = rng.integers(0, n_total, n_total)
        samples[i] = fn(values[idx])
    return float(np.mean(samples)), float(np.std(samples, ddof=1))


def bootstrap_paired(
    a: np.ndarray,
    b: np.ndarray,
    fn: Callable[[np.ndarray], float],
    n: int = 200,
    seed: int = 0,
) -> dict:
    """Bootstrap a metric on two paired arrays + their ratio.

    Returns ``{"a": (mean, SE), "b": (mean, SE), "ratio": (mean, SE)}`` where
    ``mean`` is the mean of the bootstrap distribution (≈ ``fn(original)`` at
    large N) and ``SE`` is the bootstrap standard error (std of the bootstrap
    distribution). Multiplying SE by 2 gives a normal-approximation 95 % CI;
    this is NOT the 2.5/97.5 percentile bootstrap CI.
    Uses the SAME resample indices for a and b so the ratio is properly paired.
    """
    if len(a) == 0 or len(a) != len(b):
        nan = (float("nan"), float("nan"))
        return {"a": nan, "b": nan, "ratio": nan}
    rng = np.random.default_rng(seed)
    n_total = len(a)
    sa = np.empty(n)
    sb = np.empty(n)
    for i in range(n):
        idx = rng.integers(0, n_total, n_total)
        sa[i] = fn(a[idx])
        sb[i] = fn(b[idx])
    sr = sa / sb
    return {
        "a": (float(np.mean(sa)), float(np.std(sa, ddof=1))),
        "b": (float(np.mean(sb)), float(np.std(sb, ddof=1))),
        "ratio": (float(np.mean(sr)), float(np.std(sr, ddof=1))),
    }
