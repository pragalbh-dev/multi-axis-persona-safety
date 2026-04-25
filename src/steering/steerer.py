"""Steering / capping wrapper around `external/assistant-axis` ActivationSteering.

We do not reimplement the hook math. The upstream `ActivationSteering` already
supports addition / ablation / mean-ablation / capping with multi-vector
multi-layer composition, and auto-discovers `model.layers` vs
`language_model.layers` (handles Gemma 4 multimodal). Stage 1 T1.3 = thin
wrappers + factories that translate `SteeringConfig` (Stage 1 type) into the
upstream constructor.

Capping order at the same layer is irrelevant when added PCs are orthogonal
to PC1 / AA by construction (per CONVENTIONS line 90). For the multi-axis
defense in Stage 6, `multi_axis_cap` verifies orthogonality at the capping
layer before applying — so any order is safe.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator, Sequence
from contextlib import ExitStack, contextmanager
from pathlib import Path

import torch
import torch.nn as nn

from src.steering.types import SteeringConfig

# Make the vendored upstream importable. external/assistant-axis is the source
# of truth; we never modify it.
_REPO = Path(__file__).resolve().parents[2]
_UPSTREAM = _REPO / "external" / "assistant-axis"
if str(_UPSTREAM) not in sys.path:
    sys.path.insert(0, str(_UPSTREAM))

from assistant_axis.steering import ActivationSteering  # noqa: E402

__all__ = [
    "ActivationSteering",
    "from_config",
    "cap_and_steer",
    "multi_axis_cap",
    "verify_orthogonality",
]


def _expand_layer_ranges(
    layer_indices: Sequence[int] | Sequence[tuple[int, int]],
    intervention_type: str,
) -> list[int]:
    """For capping, ranges are expanded to one int per layer; otherwise pass-through."""
    if intervention_type != "capping":
        return [int(x) for x in layer_indices]  # type: ignore[arg-type]
    out: list[int] = []
    for entry in layer_indices:
        if isinstance(entry, int):
            out.append(entry)
        else:
            start, end = int(entry[0]), int(entry[1])
            out.extend(range(start, end + 1))
    return out


def from_config(model: nn.Module, cfg: SteeringConfig) -> ActivationSteering:
    """Build an upstream `ActivationSteering` from our `SteeringConfig`.

    Per-vector layer expansion: when capping with `layer_indices` containing
    `(start, end)` ranges, this function broadcasts each (vector, threshold)
    pair across every layer in the range. The upstream class would otherwise
    require us to repeat each entry by hand.
    """
    n_vec = len(cfg.vectors)

    if cfg.intervention_type == "capping":
        # For each vector, expand its layer entry into per-layer ints, then
        # broadcast the (vector, threshold) pair across those layers.
        expanded_vectors: list[torch.Tensor] = []
        expanded_layers: list[int] = []
        expanded_taus: list[float] = []
        for i in range(n_vec):
            entry = cfg.layer_indices[i] if i < len(cfg.layer_indices) else cfg.layer_indices[0]
            layers = _expand_layer_ranges([entry], "capping")  # type: ignore[arg-type]
            for L in layers:
                expanded_vectors.append(cfg.vectors[i])
                expanded_layers.append(L)
                expanded_taus.append(cfg.cap_thresholds[i])
        return ActivationSteering(
            model=model,
            steering_vectors=expanded_vectors,
            coefficients=[1.0] * len(expanded_vectors),  # unused for capping
            layer_indices=expanded_layers,
            intervention_type="capping",
            positions=cfg.positions,
            cap_thresholds=expanded_taus,
        )

    # Non-capping modes: layer_indices is a flat int list, one per vector.
    flat_layers = [int(x) for x in cfg.layer_indices]  # type: ignore[arg-type]
    return ActivationSteering(
        model=model,
        steering_vectors=cfg.vectors,
        coefficients=cfg.coefficients or [1.0] * n_vec,
        layer_indices=flat_layers,
        intervention_type=cfg.intervention_type,
        positions=cfg.positions,
        mean_activations=cfg.mean_activations,
    )


@contextmanager
def cap_and_steer(
    model: nn.Module,
    cap_cfg: SteeringConfig,
    steer_cfg: SteeringConfig,
) -> Iterator[tuple[ActivationSteering, ActivationSteering]]:
    """Compose a capping + steering intervention as nested context managers.

    Stage 4 T4.5 ("AA-capped + PC2-steered") and Stage 6 T6.x ("multi-axis cap
    while running an attack") both want this. Hooks register independently;
    PyTorch fires them in registration order, so cap fires before steer at
    every layer where both apply. That ordering is intentional: cap projects
    the residual stream below τ first, then steering adds its perturbation on
    top. The composed forward pass thus matches what the paper does in §5.
    """
    if cap_cfg.intervention_type != "capping":
        raise ValueError(
            f"cap_cfg.intervention_type must be 'capping', got {cap_cfg.intervention_type!r}"
        )

    with ExitStack() as stack:
        cap = stack.enter_context(from_config(model, cap_cfg))
        steer = stack.enter_context(from_config(model, steer_cfg))
        yield cap, steer


def verify_orthogonality(
    vectors: Sequence[torch.Tensor], tol: float = 0.1
) -> tuple[bool, torch.Tensor]:
    """Pairwise cosine-similarity check on a set of intervention directions.

    Returns `(all_below_tol, sim_matrix)` where `sim_matrix[i, j]` is
    `|<v_i, v_j>| / (||v_i|| ||v_j||)` (off-diagonal). Used by `multi_axis_cap`
    before it builds the steerer — Stage 6 T6.1 contract requires orthogonality
    at every capping layer, otherwise we fall back to deterministic capping
    order (PC1 → PC2 → PC3) and log to `decisions.md`.
    """
    n = len(vectors)
    sims = torch.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                sims[i, j] = 1.0
                continue
            vi = vectors[i].float().flatten()
            vj = vectors[j].float().flatten()
            sims[i, j] = (vi @ vj).abs() / (vi.norm() * vj.norm() + 1e-8)
    off_diag = sims - torch.eye(n)
    return bool(off_diag.abs().max().item() <= tol), sims


def multi_axis_cap(
    model: nn.Module,
    axes: Sequence[tuple[torch.Tensor, tuple[int, int], float]],
    *,
    positions: str = "all",
    orthogonality_tol: float = 0.1,
) -> ActivationSteering:
    """Build a multi-axis capping steerer for Stage 6 T6.1.

    Args:
        axes: list of (direction_vector, (layer_start, layer_end), tau).
            Each direction is capped at every layer in its range with its tau.
        positions: "all" (default, paper convention) or "last".
        orthogonality_tol: max off-diagonal |cos_sim|; if any pair exceeds
            this threshold the user is responsible for picking a deterministic
            order. We surface a warning via stderr but DO NOT fail — the caller
            (Stage 6 T6.1) handles the fallback.
    """
    vectors = [t for t, _, _ in axes]
    ranges = [r for _, r, _ in axes]
    taus = [tau for _, _, tau in axes]

    ok, _ = verify_orthogonality(vectors, tol=orthogonality_tol)
    if not ok:
        import warnings

        warnings.warn(
            "multi_axis_cap directions exceed orthogonality_tol; "
            "Stage 6 T6.1 fallback to deterministic capping order is required.",
            stacklevel=2,
        )

    cfg = SteeringConfig(
        vectors=vectors,
        intervention_type="capping",
        layer_indices=ranges,
        cap_thresholds=taus,
        positions=positions,  # type: ignore[arg-type]
    )
    return from_config(model, cfg)
