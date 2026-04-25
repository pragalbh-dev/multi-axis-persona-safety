"""Visualization shared types.

Every figure has a static (matplotlib PDF + PNG, paper) and an interactive
(plotly JSON, dashboard) form. `make_figure` in `figures.py` returns both;
`FigureSpec` records where they land.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

FigureKind = Literal[
    "persona_space_3d",  # Fig 1 / Viz 1
    "safety_heatmap",  # Fig 2 / Viz 2
    "steering_comparator",  # Fig 3 / Viz 3
    "layer_signal",  # Fig 4 / Viz 4
    "persona_arithmetic",  # Fig 5 / Viz 5
    "multi_axis_pareto",  # Fig 6 (Stage 6 deliverable)
]


@dataclass
class FigureSpec:
    """Where a figure lives + which experiment produced it."""

    name: str  # human label, e.g. "Persona Space Explorer"
    kind: FigureKind
    source_exp: str  # e.g. "results/exp1_pca_decomposition"
    static_path: Path  # PDF / PNG for paper
    interactive_path: Path  # JSON for dashboard
    fig_number: int | None = None  # paper-side numbering
