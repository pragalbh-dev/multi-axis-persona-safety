"""Centered PCA + Marchenko-Pastur threshold for role-vector dimensionality.

Wraps `external/assistant-axis::compute_pca` (which uses sklearn under the hood)
with our preferred return shape and adds the MP threshold helper.

For Plan B, we run PCA on the **paper's released 275-role-vector cache** at the
layer that maximizes cos_sim(PC1, AA). Our own 30-rollout-per-role cache is too
noisy for the actual PC fit; we only use it for τ-calibration (Plan B Step 5).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

# Make external/assistant-axis importable.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_AA_PATH = _PROJECT_ROOT / "external" / "assistant-axis"
if str(_AA_PATH) not in sys.path:
    sys.path.insert(0, str(_AA_PATH))


@dataclass
class PCAResult:
    """Centered PCA fit on role activations at a single layer.

    `components` is `(k, d_model)` — `components[i]` is PC_i+1 (PC1 at index 0).
    `eigenvalues` is `(k,)` — corresponding singular values squared / (n-1).
    `explained_variance_ratio` is `(k,)` summing to 1.
    `mean` is `(d_model,)` — the centering mean (subtract before projecting).
    """

    components: np.ndarray
    eigenvalues: np.ndarray
    explained_variance_ratio: np.ndarray
    mean: np.ndarray
    n_samples: int
    d_model: int

    def project(self, x: np.ndarray | torch.Tensor, k: int | None = None) -> np.ndarray:
        """Project `x` (..., d_model) onto top-`k` PCs. Returns (..., k)."""
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().float().numpy()
        centered = x - self.mean
        comps = self.components if k is None else self.components[:k]
        return centered @ comps.T


def fit_pca(role_activations: np.ndarray | torch.Tensor) -> PCAResult:
    """Center + fit PCA on `(n, d_model)` role activations.

    Note: `compute_pca` from upstream applies sklearn PCA which centers by
    default but doesn't expose the mean cleanly. We re-implement the centered
    SVD directly here so we can store the mean for later projections.
    """
    if isinstance(role_activations, torch.Tensor):
        role_activations = role_activations.detach().cpu().float().numpy()
    if role_activations.ndim != 2:
        raise ValueError(
            f"role_activations must be (n, d_model); got {role_activations.shape}"
        )
    n, d = role_activations.shape
    mean = role_activations.mean(axis=0)
    centered = role_activations - mean

    # Thin SVD; full_matrices=False so V has shape (min(n,d), d).
    _u, s, vt = np.linalg.svd(centered, full_matrices=False)
    eigenvalues = (s**2) / max(n - 1, 1)
    total = eigenvalues.sum()
    explained = eigenvalues / total if total > 0 else np.zeros_like(eigenvalues)

    return PCAResult(
        components=vt,
        eigenvalues=eigenvalues,
        explained_variance_ratio=explained,
        mean=mean,
        n_samples=n,
        d_model=d,
    )


def marchenko_pastur_threshold(d: int, n: int, sigma: float = 1.0) -> float:
    """Upper edge λ+ of the MP distribution.

    γ = d/n (advisory: role vectors are correlated, not iid, so MP is a soft
    bound; the paper's convention is the top PCs explaining ≥70% variance,
    we report both).

    λ+ = σ² (1 + √γ)²
    """
    if n <= 0:
        raise ValueError("n must be positive")
    gamma = d / n
    return float(sigma**2 * (1 + np.sqrt(gamma)) ** 2)


def num_pcs_above_mp(eigenvalues: np.ndarray, d: int, n: int, sigma: float = 1.0) -> int:
    """Count of eigenvalues exceeding the MP upper edge."""
    threshold = marchenko_pastur_threshold(d, n, sigma)
    return int((eigenvalues > threshold).sum())


def num_pcs_for_variance(explained_variance_ratio: np.ndarray, target: float = 0.7) -> int:
    """Smallest k such that sum(explained[:k]) >= target. Paper convention 0.7."""
    cum = np.cumsum(explained_variance_ratio)
    idx = np.searchsorted(cum, target) + 1
    return int(min(idx, len(explained_variance_ratio)))
