"""Steering / capping config types.

The actual hook implementation lives in
`external/assistant-axis/assistant_axis/steering.py::ActivationSteering`. We
keep our wrapper in `src.steering.steerer` and pass it `SteeringConfig`
instances built from `ExperimentConfig.steering`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import torch

InterventionType = Literal["addition", "ablation", "capping", "mean_ablation"]
Positions = Literal["all", "last"]


@dataclass
class SteeringConfig:
    """One operation against the model: add v, ablate v, or cap projection onto v.

    For `intervention_type="capping"`, `layer_indices` is a list of layer
    ranges (each a `(start, end)` inclusive tuple) and `cap_thresholds` is a
    list of τ values, one per vector. For other types, `layer_indices` is a
    flat list of single-layer ints.
    """

    vectors: list[torch.Tensor]
    intervention_type: InterventionType
    layer_indices: list[int] | list[tuple[int, int]]
    coefficients: list[float] = field(default_factory=list)
    cap_thresholds: list[float] = field(default_factory=list)
    positions: Positions = "all"
    mean_activations: list[torch.Tensor] | None = None

    def __post_init__(self) -> None:
        n = len(self.vectors)
        if n == 0:
            raise ValueError("SteeringConfig requires at least one vector")
        if self.coefficients and len(self.coefficients) != n:
            raise ValueError("coefficients length must match vectors")
        if self.intervention_type == "capping":
            if len(self.cap_thresholds) != n:
                raise ValueError("cap_thresholds length must match vectors when capping")
            for entry in self.layer_indices:
                if not (isinstance(entry, tuple | list) and len(entry) == 2):
                    raise ValueError(
                        "layer_indices entries must be (start, end) ranges when capping"
                    )
        else:
            for entry in self.layer_indices:
                if not isinstance(entry, int):
                    raise ValueError(
                        "layer_indices must be ints when intervention_type is not 'capping'"
                    )
        if self.intervention_type == "mean_ablation" and self.mean_activations is None:
            raise ValueError("mean_ablation requires mean_activations")
