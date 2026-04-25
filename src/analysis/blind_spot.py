"""Blind-spot lift: AUC(AA + PCs) − AUC(AA only).

Per CONVENTIONS line 102: marginal predictive gain from adding PCs 2..k to a
joint logistic LASSO with AA as the always-on feature. Stage 3 T3.8 fills the
implementation; Stage 1 locks the function contract + dataclass.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from src.analysis.types import BlindSpotLift

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
    """Compute AUC(AA only), AUC(AA + nonzero LASSO-selected PCs), delta CI.

    Args:
        aa_projection: shape (n,). Per-prompt projection onto the Assistant
            Axis at the extraction layer.
        pc_projections: shape (n, k). Per-prompt projections onto PC2..PCk.
        y_binary: shape (n,). Binarized harm label.
        pc_names: length-k labels matching pc_projections columns (e.g.
            ["pc2", "pc3", ..., "pck"]).
        seed / n_resamples: bootstrap controls.

    Returns: BlindSpotLift with both AUCs, the bootstrap CI on the delta, and
    the list of LASSO-selected PC names.

    Stage 3 T3.8 implementation outline:
    1. Standardize features.
    2. Fit logistic_lasso_cv on `[AA]` only → auc_aa_only.
    3. Fit logistic_lasso_cv on `[AA, PC2, ..., PCk]` → auc_with_pcs +
       selected_pcs.
    4. Bootstrap CI on the delta via paired resamples (use
       bootstrap.bca_ci_difference on the held-out predictions).
    """
    raise NotImplementedError(
        "Stage 3 T3.8 implements blind_spot_lift. Stage 1 ships the contract."
    )
