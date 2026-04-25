"""Correlation tests + multiple-testing correction.

Per CONVENTIONS "Statistical Framework":
- Per-PC × binary harm  → point-biserial.
- Aggregate-rate × continuous PC projection (paper's r=0.39-0.52) → Pearson.
- Ordinal-3-level × continuous → Kendall τ (or rank-biserial, alias).
- Multiple testing across all PCs → BH-FDR at q=0.05.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from src.analysis.types import CorrelationResult

NDArrayF = npt.NDArray[np.float64]


def pearson_with_ci(x: Sequence[float], y: Sequence[float]) -> CorrelationResult:
    """Pearson correlation + p-value. CI is reported separately via bootstrap."""
    from scipy.stats import pearsonr

    xa = np.asarray(x, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    res = pearsonr(xa, ya)
    return CorrelationResult(stat="pearson", r=float(res.statistic), p=float(res.pvalue), n=xa.size)


def point_biserial(continuous: Sequence[float], binary: Sequence[int]) -> CorrelationResult:
    """Point-biserial correlation: continuous PC projection vs binary harm label."""
    from scipy.stats import pointbiserialr

    cs = np.asarray(continuous, dtype=np.float64)
    bs = np.asarray(binary, dtype=np.int64)
    if set(np.unique(bs).tolist()) - {0, 1}:
        raise ValueError(f"binary array must contain only 0/1, got {set(np.unique(bs).tolist())}")
    res = pointbiserialr(bs, cs)
    return CorrelationResult(
        stat="point_biserial", r=float(res.statistic), p=float(res.pvalue), n=cs.size
    )


def kendall_tau(x: Sequence[float], y: Sequence[float | int]) -> CorrelationResult:
    """Kendall τ — used for the 3-level ordinal robustness check."""
    from scipy.stats import kendalltau

    xa = np.asarray(x, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    res = kendalltau(xa, ya)
    return CorrelationResult(stat="kendall", r=float(res.statistic), p=float(res.pvalue), n=xa.size)


def bh_fdr(pvalues: Sequence[float], q: float = 0.05) -> NDArrayF:
    """Benjamini-Hochberg FDR-adjusted q-values for `pvalues` (any order).

    Returns an array same length / order as `pvalues`. A test is "significant"
    at FDR=q iff `q_adjusted <= q`.
    """
    p = np.asarray(pvalues, dtype=np.float64)
    n = p.size
    order = np.argsort(p)
    ranked = p[order]
    adjusted = ranked * n / (np.arange(n) + 1)
    # Enforce monotonicity from the largest p down.
    for i in range(n - 2, -1, -1):
        adjusted[i] = min(adjusted[i], adjusted[i + 1])
    out = np.empty_like(p)
    out[order] = np.minimum(adjusted, 1.0)
    return out
