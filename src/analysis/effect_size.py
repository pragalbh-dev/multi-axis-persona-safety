"""Cohen's d with bootstrap CI.

Per CONVENTIONS line 93: medium d ≥ 0.5; large d ≥ 0.8 (Cohen 1988). Used in
Plan B Step 9 to quantify projection differences between harmful and
non-harmful response groups within an AA-capped condition.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from src.analysis.types import BootstrapResult

NDArrayF = npt.NDArray[np.float64]


def _pooled_std(a: NDArrayF, b: NDArrayF) -> float:
    n_a = a.size
    n_b = b.size
    if n_a < 2 or n_b < 2:
        return float("nan")
    var_a = float(np.var(a, ddof=1))
    var_b = float(np.var(b, ddof=1))
    pooled = ((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2)
    return float(np.sqrt(max(pooled, 0.0)))


def cohens_d_point(a: NDArrayF | Sequence[float], b: NDArrayF | Sequence[float]) -> float:
    """Standard Cohen's d using pooled SD (Cohen 1988)."""
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    if aa.size == 0 or bb.size == 0:
        return float("nan")
    sd = _pooled_std(aa, bb)
    if not np.isfinite(sd) or sd == 0.0:
        return float("nan")
    return float((np.mean(aa) - np.mean(bb)) / sd)


def cohens_d(
    a: NDArrayF | Sequence[float],
    b: NDArrayF | Sequence[float],
    *,
    n_resamples: int = 10_000,
    seed: int = 42,
) -> BootstrapResult:
    """Cohen's d (pooled-SD effect size) with bootstrap percentile CI.

    BCa would require a bivariate jackknife; the percentile interval here is
    standard for two-sample-d intervals and matches scipy.stats.bootstrap
    behavior for 'percentile'.
    """
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    point = cohens_d_point(aa, bb)
    if not np.isfinite(point):
        return BootstrapResult(point=float("nan"), ci_low=float("nan"), ci_high=float("nan"), n_resamples=0)

    rng = np.random.default_rng(seed)
    samples = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        ia = rng.integers(0, aa.size, size=aa.size)
        ib = rng.integers(0, bb.size, size=bb.size)
        samples[i] = cohens_d_point(aa[ia], bb[ib])
    samples = samples[np.isfinite(samples)]
    if samples.size == 0:
        return BootstrapResult(point=point, ci_low=float("nan"), ci_high=float("nan"), n_resamples=0)
    lo = float(np.quantile(samples, 0.025))
    hi = float(np.quantile(samples, 0.975))
    return BootstrapResult(point=point, ci_low=lo, ci_high=hi, n_resamples=n_resamples)


def cohens_d_thresholds() -> dict[str, float]:
    """Return the Cohen 1988 thresholds for interpreting d."""
    return {"small": 0.2, "medium": 0.5, "large": 0.8}
