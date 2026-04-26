"""Tests for Plan B-critical analysis utilities.

Validates that:
  - logistic_lasso_cv recovers signal on a separable synthetic dataset
  - blind_spot_lift returns a positive delta when extra features carry signal
  - cohens_d gives a known value for textbook input
"""

from __future__ import annotations

import numpy as np

from src.analysis.blind_spot import blind_spot_lift
from src.analysis.effect_size import cohens_d, cohens_d_point
from src.analysis.lasso import logistic_lasso_cv


def test_logistic_lasso_recovers_signal() -> None:
    rng = np.random.default_rng(42)
    n = 400
    # 3-feature synthetic: features 0+1 informative, 2 noise
    X = rng.normal(size=(n, 3))
    logits = 1.5 * X[:, 0] - 1.2 * X[:, 1] + 0.0 * X[:, 2]
    p = 1 / (1 + np.exp(-logits))
    y = (rng.uniform(size=n) < p).astype(int)
    fit = logistic_lasso_cv(
        X, y, ["a", "b", "noise"], n_folds=5, seed=42, n_resamples=200
    )
    # Held-out AUC should be well above chance
    assert fit.auc > 0.75
    # Informative features selected; noise probably dropped (LASSO at moderate C)
    assert "a" in fit.selected_features
    assert "b" in fit.selected_features
    # Coef signs match (a positive, b negative) up to sign of refit
    assert fit.coefs["a"] > 0
    assert fit.coefs["b"] < 0


def test_blind_spot_lift_positive_when_pcs_help() -> None:
    rng = np.random.default_rng(0)
    n = 300
    aa = rng.normal(size=n)
    pc2 = rng.normal(size=n)
    pc3 = rng.normal(size=n)
    # y is driven mainly by pc2 + pc3, only slightly by aa
    logits = 0.3 * aa + 1.5 * pc2 - 1.0 * pc3
    p = 1 / (1 + np.exp(-logits))
    y = (rng.uniform(size=n) < p).astype(int)

    lift = blind_spot_lift(
        aa,
        np.stack([pc2, pc3], axis=1),
        y,
        ["pc2", "pc3"],
        seed=0,
        n_resamples=500,
    )
    # PCs add real signal beyond AA → delta should be positive
    assert lift.auc_with_pcs > lift.auc_aa_only
    assert lift.delta.point > 0.05
    # Both pcs should be selected
    assert "pc2" in lift.selected_pcs
    assert "pc3" in lift.selected_pcs


def test_blind_spot_lift_zero_when_pcs_are_noise() -> None:
    rng = np.random.default_rng(1)
    n = 300
    aa = rng.normal(size=n)
    pc_noise = rng.normal(size=(n, 2))
    logits = 1.5 * aa  # only AA matters
    p = 1 / (1 + np.exp(-logits))
    y = (rng.uniform(size=n) < p).astype(int)

    lift = blind_spot_lift(
        aa, pc_noise, y, ["pc2", "pc3"], seed=1, n_resamples=500
    )
    # Lift should be near zero (CI may include zero)
    assert abs(lift.delta.point) < 0.05
    assert lift.delta.ci_low <= 0.0 <= lift.delta.ci_high


def test_cohens_d_textbook() -> None:
    # Known: groups with means 1.0 and 0.0, both SD=1.0 → d ≈ 1.0
    rng = np.random.default_rng(0)
    a = rng.normal(loc=1.0, scale=1.0, size=200)
    b = rng.normal(loc=0.0, scale=1.0, size=200)
    d = cohens_d_point(a, b)
    assert 0.7 < d < 1.3  # noise-tolerant

    res = cohens_d(a, b, n_resamples=500, seed=0)
    assert abs(res.point - d) < 1e-9
    assert res.ci_low < res.point < res.ci_high


def test_cohens_d_zero_when_equal() -> None:
    rng = np.random.default_rng(0)
    a = rng.normal(size=100)
    b = rng.normal(size=100)
    d = cohens_d_point(a, b)
    assert abs(d) < 0.5
