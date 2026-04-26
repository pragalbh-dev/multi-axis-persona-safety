"""Per-prompt projection + cosine-similarity helpers.

Used by:
  - Plan B Step 7b: project each (prompt, response) activation onto AA + PC1..PC10
                    to fill `aa_projection` + `pc_projections` columns of details.parquet.
  - Plan B Step 2: cos_sim(PC1, AA) per layer for L* selection.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import torch


def _to_np(x: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().float().numpy()
    return np.asarray(x).astype(np.float32)


def cos_sim(a: np.ndarray | torch.Tensor, b: np.ndarray | torch.Tensor) -> float:
    """Cosine similarity between two 1-D vectors."""
    av = _to_np(a)
    bv = _to_np(b)
    av = av.flatten()
    bv = bv.flatten()
    na = float(np.linalg.norm(av))
    nb = float(np.linalg.norm(bv))
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(av, bv) / (na * nb))


def project_onto_direction(
    activations: np.ndarray | torch.Tensor,
    direction: np.ndarray | torch.Tensor,
    *,
    normalize: bool = True,
) -> np.ndarray:
    """Project `(n, d)` activations onto a `(d,)` direction. Returns `(n,)`.

    With `normalize=True` (default) the direction is L2-normalized first so
    the projection is the scalar component along the unit direction.
    """
    A = _to_np(activations)
    v = _to_np(direction).flatten()
    if A.ndim == 1:
        A = A[None, :]
    if normalize:
        n = np.linalg.norm(v)
        if n > 0:
            v = v / n
    return A @ v


def per_layer_cos_sim(
    pc1_per_layer: np.ndarray | torch.Tensor,
    aa_per_layer: np.ndarray | torch.Tensor,
) -> np.ndarray:
    """Per-layer cos_sim between PC1 and AA. Both inputs `(n_layers, d_model)`.

    Returns `(n_layers,)` array.
    """
    P = _to_np(pc1_per_layer)
    A = _to_np(aa_per_layer)
    if P.shape != A.shape:
        raise ValueError(f"shape mismatch: PC1 {P.shape} vs AA {A.shape}")
    out = np.zeros(P.shape[0], dtype=np.float32)
    for layer in range(P.shape[0]):
        out[layer] = cos_sim(P[layer], A[layer])
    return out


def argmax_cos_sim_layer(
    pc1_per_layer: np.ndarray | torch.Tensor,
    aa_per_layer: np.ndarray | torch.Tensor,
    candidate_layers: Iterable[int] | None = None,
) -> tuple[int, float]:
    """Return `(L*, cos_sim_at_Lstar)` — the layer with max cos_sim(PC1, AA).

    Optionally restricted to `candidate_layers` (e.g. middle 50-90% per CONVENTIONS).
    """
    sims = per_layer_cos_sim(pc1_per_layer, aa_per_layer)
    if candidate_layers is None:
        idx = int(np.argmax(sims))
    else:
        cands = list(candidate_layers)
        sub = sims[cands]
        idx = cands[int(np.argmax(sub))]
    return idx, float(sims[idx])


def normalize_to_norm(v: np.ndarray | torch.Tensor, target_norm: float) -> np.ndarray:
    """Scale `v` to have L2 norm == `target_norm` (paper convention for steering vectors).

    Used to scale steering directions to the lmsys-chat-1m mean residual-stream norm
    at the target layer (CONVENTIONS "Steering-vector norm convention").
    """
    arr = _to_np(v)
    n = np.linalg.norm(arr)
    if n == 0:
        return arr
    return arr * (target_norm / n)
