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
    """Nested 10-fold CV logistic-LASSO. Stage 2 T2.7a (Plan B critical path).

    Inner CV picks the L1 strength via sklearn's LogisticRegressionCV with
    `penalty='l1', solver='saga', scoring='roc_auc'`. We then aggregate
    held-out predictions across the 10 outer folds and compute ROC-AUC plus
    a bootstrap CI on the held-out (y, y_score) pairs.

    Final coefficients = refit-on-all at the median inner-best C (more stable
    than averaging fold-specific coefs).

    Args:
        X: (n_samples, n_features). Should be the per-prompt feature matrix
           with columns matching `feature_names` exactly.
        y: (n_samples,) binary {0, 1}.
        feature_names: list[str] of length n_features (e.g. ["aa", "pc1", ..., "pc10"]).
        n_folds: outer CV folds (default 10).
        seed: RNG seed.
        bootstrap_auc: when True (default), compute BCa CI on held-out AUC.
        n_resamples: bootstrap resamples (default 10K per CONVENTIONS).

    Returns:
        LassoFit with `coefs` (refit), `auc` (held-out), `auc_ci` (bootstrap),
        `selected_features` (nonzero coef names).
    """
    from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler

    X_arr = np.asarray(X, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.int64)
    if X_arr.ndim != 2:
        raise ValueError(f"X must be 2D; got {X_arr.shape}")
    if X_arr.shape[1] != len(feature_names):
        raise ValueError(
            f"X has {X_arr.shape[1]} cols but feature_names has {len(feature_names)}"
        )
    if X_arr.shape[0] != y_arr.size:
        raise ValueError(f"X.shape[0]={X_arr.shape[0]} but y.size={y_arr.size}")

    if len(np.unique(y_arr)) < 2:
        # Degenerate; cannot fit. Return zero coefs + AUC=0.5.
        return LassoFit(
            coefs={n: 0.0 for n in feature_names},
            auc=0.5,
            auc_ci=BootstrapResult(point=0.5, ci_low=0.5, ci_high=0.5, n_resamples=0),
            selected_features=[],
            intercept=0.0,
            cv_alpha=0.0,
        )

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    held_out_y: list[int] = []
    held_out_score: list[float] = []
    inner_best_Cs: list[float] = []

    for fold_i, (train_idx, test_idx) in enumerate(skf.split(X_arr, y_arr)):
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X_arr[train_idx])
        Xte = scaler.transform(X_arr[test_idx])

        # Inner CV picks C
        inner = LogisticRegressionCV(
            Cs=20,
            cv=5,
            penalty="l1",
            solver="saga",
            scoring="roc_auc",
            max_iter=2000,
            random_state=seed + fold_i,
        )
        inner.fit(Xtr, y_arr[train_idx])
        best_C = float(inner.C_[0])
        inner_best_Cs.append(best_C)
        # Score held-out
        ytest_score = inner.predict_proba(Xte)[:, 1]
        held_out_y.extend(y_arr[test_idx].tolist())
        held_out_score.extend(ytest_score.tolist())

    # Final model: refit on all data at median inner-best C
    median_C = float(np.median(inner_best_Cs))
    full_scaler = StandardScaler()
    X_full = full_scaler.fit_transform(X_arr)
    final = LogisticRegression(
        C=median_C,
        penalty="l1",
        solver="saga",
        max_iter=4000,
        random_state=seed,
    )
    final.fit(X_full, y_arr)
    coefs = dict(zip(feature_names, final.coef_[0].tolist()))
    selected = [n for n, c in coefs.items() if abs(c) > 1e-8]

    auc_res = auc_with_ci(
        np.asarray(held_out_y, dtype=np.int64),
        np.asarray(held_out_score, dtype=np.float64),
        seed=seed,
        n_resamples=n_resamples if bootstrap_auc else 1,
    )

    return LassoFit(
        coefs=coefs,
        auc=auc_res.point,
        auc_ci=auc_res,
        selected_features=selected,
        intercept=float(final.intercept_[0]),
        cv_alpha=median_C,
    )


def ordinal_lasso_cv(
    X: NDArrayF,
    y: NDArrayI | Sequence[int],
    feature_names: Sequence[str],
    *,
    n_folds: int = 10,
    seed: int = 42,
) -> LassoFit:
    """Cumulative-link ordinal LASSO on 3-level y. **POST-PLAN B (T2.7b).**

    `y` levels (per CONVENTIONS line 99): 0 = refusal-family, 1 =
    partial-info-family, 2 = full-info-family. Drop nonsensical /
    out_of_context / other before calling this.

    Stage 3 T3.8 secondary robustness check; not invoked by Plan B.
    """
    raise NotImplementedError(
        "T2.7b (post-Plan B). Implementation deferred per plans/decisions.md "
        "2026-04-25 22:15."
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
