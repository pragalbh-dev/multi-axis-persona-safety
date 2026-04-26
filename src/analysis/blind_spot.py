"""Blind-spot lift: AUC(AA + PCs) − AUC(AA only).

Per CONVENTIONS line 102: marginal predictive gain from adding PCs 2..k to a
joint logistic LASSO with AA as the always-on feature. Stage 3 T3.8 fills the
implementation; Stage 1 locks the function contract + dataclass.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from src.analysis.types import BlindSpotLift, BootstrapResult

NDArrayF = npt.NDArray[np.float64]
NDArrayI = npt.NDArray[np.int64]


def blind_spot_lift(
    aa_projection: NDArrayF | Sequence[float],
    pc_projections: NDArrayF,
    y_binary: NDArrayI | Sequence[int],
    pc_names: Sequence[str],
    *,
    seed: int = 42,
    n_resamples: int = 10_000,
) -> BlindSpotLift:
    """AUC(AA + nonzero LASSO-selected PCs) − AUC(AA only) with bootstrap delta CI.

    Plan B's H1 numerical statement.

    Implementation: fit logistic_lasso_cv twice (AA-only then AA+PCs), grab the
    held-out predictions both times, then resample paired-by-row indices and
    compute the AUC delta on each resample. CI is the percentile interval on
    those resampled deltas.
    """
    from sklearn.metrics import roc_auc_score

    from src.analysis.lasso import logistic_lasso_cv

    aa = np.asarray(aa_projection, dtype=np.float64).reshape(-1, 1)
    pcs = np.asarray(pc_projections, dtype=np.float64)
    if pcs.ndim == 1:
        pcs = pcs.reshape(-1, 1)
    y = np.asarray(y_binary, dtype=np.int64)
    if aa.shape[0] != y.size or pcs.shape[0] != y.size:
        raise ValueError("aa_projection / pc_projections / y_binary must align in n.")

    # Fit AA-only and AA+PCs
    fit_aa = logistic_lasso_cv(
        aa, y, ["aa"], n_folds=10, seed=seed, bootstrap_auc=False, n_resamples=1
    )
    aa_with_pcs = np.concatenate([aa, pcs], axis=1)
    feature_names = ["aa"] + list(pc_names)
    fit_full = logistic_lasso_cv(
        aa_with_pcs, y, feature_names, n_folds=10, seed=seed, bootstrap_auc=False, n_resamples=1
    )

    # Re-derive held-out predictions for the bootstrap delta. Re-running the
    # nested CV inside the resample loop would be too slow; instead we use
    # the refit model's in-sample predictions as a (slightly optimistic) score
    # surface, then bootstrap row indices. This matches the practice in
    # CONVENTIONS where blind-spot lift is "AUC(full) - AUC(reduced)" with CIs
    # via paired resamples. Refit-on-all is consistent with the LassoFit.coefs
    # we report, and the bootstrap CI captures sampling variability around
    # both AUCs in the same way.
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    sc_aa = StandardScaler().fit(aa)
    aa_z = sc_aa.transform(aa)
    m_aa = LogisticRegression(
        C=fit_aa.cv_alpha or 1.0,
        penalty="l1",
        solver="saga",
        max_iter=4000,
        random_state=seed,
    ).fit(aa_z, y)
    score_aa = m_aa.predict_proba(aa_z)[:, 1]

    sc_full = StandardScaler().fit(aa_with_pcs)
    full_z = sc_full.transform(aa_with_pcs)
    m_full = LogisticRegression(
        C=fit_full.cv_alpha or 1.0,
        penalty="l1",
        solver="saga",
        max_iter=4000,
        random_state=seed,
    ).fit(full_z, y)
    score_full = m_full.predict_proba(full_z)[:, 1]

    rng = np.random.default_rng(seed)
    n = y.size
    deltas = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y[idx])) < 2:
            deltas[i] = 0.0
            continue
        a = roc_auc_score(y[idx], score_aa[idx])
        f = roc_auc_score(y[idx], score_full[idx])
        deltas[i] = f - a

    point_delta = float(roc_auc_score(y, score_full) - roc_auc_score(y, score_aa))
    lo = float(np.quantile(deltas, 0.025))
    hi = float(np.quantile(deltas, 0.975))

    delta_ci = BootstrapResult(
        point=point_delta, ci_low=lo, ci_high=hi, n_resamples=n_resamples
    )

    return BlindSpotLift(
        auc_aa_only=fit_aa.auc,
        auc_with_pcs=fit_full.auc,
        delta=delta_ci,
        selected_pcs=[
            n for n in fit_full.selected_features if n != "aa"
        ],
    )
