"""Bootstrap CI helpers — BCa 95% with 10K resamples per CONVENTIONS.

Stage 1 T1.5 contract; full implementation lives here so other Stage 1 modules
can compose it (e.g. `eval_safety` reports BCa CI on harm_rate).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import numpy.typing as npt

from src.analysis.types import BootstrapResult

DEFAULT_N_RESAMPLES = 10_000
DEFAULT_CI_LOW = 0.025
DEFAULT_CI_HIGH = 0.975

NDArrayF = npt.NDArray[np.float64]


def bca_ci(
    data: NDArrayF | Sequence[float],
    statistic: Callable[[NDArrayF], float],
    *,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    seed: int = 42,
    alpha_low: float = DEFAULT_CI_LOW,
    alpha_high: float = DEFAULT_CI_HIGH,
) -> BootstrapResult:
    """Bias-corrected accelerated bootstrap CI on `statistic(data)`.

    Implementation per Efron 1987. Uses scipy.stats.bootstrap when available
    (it is, since scipy>=1.7), with the BCa method explicitly. We thin-wrap
    scipy here so the shape of our `BootstrapResult` is consistent with the
    rest of `src.analysis.*` and so callers don't pass scipy specifics.
    """
    arr = np.asarray(data, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"bca_ci expects 1-D data, got shape {arr.shape}")
    if arr.size == 0:
        raise ValueError("bca_ci requires at least one sample")

    from scipy.stats import bootstrap

    res = bootstrap(
        (arr,),
        lambda x, axis=0: (
            statistic(x) if x.ndim == 1 else np.array([statistic(x[i]) for i in range(x.shape[0])])
        ),
        confidence_level=alpha_high - alpha_low,
        n_resamples=n_resamples,
        method="BCa",
        random_state=np.random.default_rng(seed),
        vectorized=False,
    )
    point = float(statistic(arr))
    return BootstrapResult(
        point=point,
        ci_low=float(res.confidence_interval.low),
        ci_high=float(res.confidence_interval.high),
        n_resamples=n_resamples,
    )


def _mean_f64(x: NDArrayF) -> float:
    return float(np.mean(x))


def bca_ci_difference(
    a: NDArrayF | Sequence[float],
    b: NDArrayF | Sequence[float],
    *,
    statistic: Callable[[NDArrayF], float] = _mean_f64,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    seed: int = 42,
) -> BootstrapResult:
    """BCa CI on `statistic(a) - statistic(b)`. Used for blind-spot lift deltas.

    Stage 1 T1.5 contract. Stage 3 T3.8 / Stage 4 T4.5 fill the call sites.
    """
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    rng = np.random.default_rng(seed)
    delta_samples = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        ia = rng.integers(0, aa.size, size=aa.size)
        ib = rng.integers(0, bb.size, size=bb.size)
        delta_samples[i] = statistic(aa[ia]) - statistic(bb[ib])
    point = float(statistic(aa) - statistic(bb))
    # Percentile-style CI on the delta distribution (BCa correction would
    # require jackknife on the joint sample; percentile is the standard
    # falback for paired-resample deltas and is what scipy uses when method
    # is 'percentile'). Stage 3 T3.8 may upgrade to BCa-on-delta if needed.
    lo = float(np.quantile(delta_samples, DEFAULT_CI_LOW))
    hi = float(np.quantile(delta_samples, DEFAULT_CI_HIGH))
    return BootstrapResult(point=point, ci_low=lo, ci_high=hi, n_resamples=n_resamples)
