"""Analysis result types.

These dataclasses are the contract between `src.analysis.*` (stats functions)
and the experiment scripts that consume them. Lock at Stage 1 T1.5 so
downstream agents can reach for the right fields without re-litigating shape.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

CorrelationStat = Literal["pearson", "point_biserial", "kendall", "rank_biserial"]


@dataclass
class BootstrapResult:
    """A point estimate plus a 95% BCa CI (10K resamples by default)."""

    point: float
    ci_low: float
    ci_high: float
    n_resamples: int = 10000

    def __str__(self) -> str:
        return f"{self.point:.4f} [{self.ci_low:.4f}, {self.ci_high:.4f}]"


@dataclass
class CorrelationResult:
    stat: CorrelationStat
    r: float
    p: float
    q_fdr: float | None = None
    n: int = 0


@dataclass
class LassoFit:
    """Output of nested 10-fold CV logistic LASSO (Stage 3 T3.8).

    `coefs` is keyed by feature name (e.g. {"aa": 0.71, "pc2": 0.0, ...}).
    `selected_features` is the subset with nonzero coefficients.
    """

    coefs: dict[str, float]
    auc: float
    auc_ci: BootstrapResult
    selected_features: list[str]
    intercept: float = 0.0
    cv_alpha: float = 0.0


@dataclass
class BlindSpotLift:
    """`AUC(AA + selected PCs) − AUC(AA only)` with bootstrap BCa CI on the delta."""

    auc_aa_only: float
    auc_with_pcs: float
    delta: BootstrapResult
    selected_pcs: list[str] = field(default_factory=list)
