"""Statistical analysis. See `src/README.md` for the module map."""

from src.analysis.blind_spot import blind_spot_lift
from src.analysis.bootstrap import bca_ci, bca_ci_difference
from src.analysis.correlation import bh_fdr, kendall_tau, pearson_with_ci, point_biserial
from src.analysis.lasso import auc_with_ci, logistic_lasso_cv, ordinal_lasso_cv
from src.analysis.types import BlindSpotLift, BootstrapResult, CorrelationResult, LassoFit

__all__ = [
    "BlindSpotLift",
    "BootstrapResult",
    "CorrelationResult",
    "LassoFit",
    "auc_with_ci",
    "bca_ci",
    "bca_ci_difference",
    "bh_fdr",
    "blind_spot_lift",
    "kendall_tau",
    "logistic_lasso_cv",
    "ordinal_lasso_cv",
    "pearson_with_ci",
    "point_biserial",
]
