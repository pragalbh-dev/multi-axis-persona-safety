"""Logistic LASSO + ordinal-LASSO robustness check (CONVENTIONS-locked).

Stage 1 T1.5 contract. Stage 3 T3.8 is the first real consumer.

- **Primary:** binomial logistic LASSO on binarized harm, nested 10-fold CV,
  ROC-AUC quality metric. Features = `{AA, PC2, PC3, ..., PCk}` per CONVENTIONS
  line 112 (PC1 dropped as redundant with AA).
- **Secondary:** ordinal LASSO (cumulative-link) on a 3-level harm collapse —
  refusal-family / partial-info-family / full-info-family. If primary and
  secondary disagree on which PCs are nonzero, both are reported.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from src.analysis.types import BootstrapResult, LassoFit

NDArrayF = npt.NDArray[np.float64]
NDArrayI = npt.NDArray[np.int64]


def logistic_lasso_cv(
    X: NDArrayF,
    y: NDArrayI | Sequence[int],
    feature_names: Sequence[str],
    *,
    n_folds: int = 10,
    seed: int = 42,
    bootstrap_auc: bool = True,
    n_resamples: int = 10_000,
) -> LassoFit:
    """Nested 10-fold CV logistic-LASSO, returning coefs + ROC-AUC + CI.

    Stage 3 T3.8 deliverable. Stage 1 ships the contract.

    Outline (Stage 3):
    1. Standardize columns of `X`.
    2. Outer fold = 10. Inner CV = LogisticRegressionCV(penalty='l1',
       solver='liblinear' or 'saga', Cs=20, scoring='roc_auc').
    3. For each fold, refit on all-train at the inner-best C; predict
       held-out.
    4. Aggregate held-out predictions → ROC-AUC. Bootstrap BCa CI on the
       held-out scores when `bootstrap_auc=True`.
    5. Report final coefficient set as the average across folds (or refit-on-all
       at the median best C; pick one and document in decisions.md if asked).
    """
    raise NotImplementedError(
        "Stage 3 T3.8 implements logistic_lasso_cv. Stage 1 ships the contract."
    )


def ordinal_lasso_cv(
    X: NDArrayF,
    y: NDArrayI | Sequence[int],
    feature_names: Sequence[str],
    *,
    n_folds: int = 10,
    seed: int = 42,
) -> LassoFit:
    """Cumulative-link ordinal LASSO on 3-level y.

    `y` levels (per CONVENTIONS line 99): 0 = refusal-family, 1 =
    partial-info-family, 2 = full-info-family. Drop nonsensical /
    out_of_context / other before calling this.

    Stage 3 T3.8 deliverable.
    """
    raise NotImplementedError(
        "Stage 3 T3.8 implements ordinal_lasso_cv. Stage 1 ships the contract."
    )


def auc_with_ci(
    y_true: NDArrayI | Sequence[int],
    y_score: NDArrayF | Sequence[float],
    *,
    seed: int = 42,
    n_resamples: int = 10_000,
) -> BootstrapResult:
    """ROC-AUC + BCa bootstrap CI. Helper for the LASSO fit + blind-spot lift."""
    from sklearn.metrics import roc_auc_score

    yt = np.asarray(y_true, dtype=np.int64)
    ys = np.asarray(y_score, dtype=np.float64)
    if yt.size != ys.size:
        raise ValueError("y_true and y_score must be the same length")
    paired: NDArrayF = np.stack([yt.astype(np.float64), ys], axis=1)

    def _auc(rows: npt.NDArray[np.float64]) -> float:
        if len(np.unique(rows[:, 0])) < 2:
            return float("nan")
        return float(roc_auc_score(rows[:, 0].astype(int), rows[:, 1]))

    # bca_ci wants 1-D data, so we hash rows into indices and resample those.
    n = paired.shape[0]
    rng = np.random.default_rng(seed)
    samples = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        samples[i] = _auc(paired[idx])
    point = _auc(paired)
    lo = float(np.nanquantile(samples, 0.025))
    hi = float(np.nanquantile(samples, 0.975))
    return BootstrapResult(point=point, ci_low=lo, ci_high=hi, n_resamples=n_resamples)
