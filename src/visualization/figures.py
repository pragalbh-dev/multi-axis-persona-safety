"""Figure rendering — paper static + dashboard interactive in one call.

Every paper figure has a matplotlib (PDF + PNG) form for inclusion and a
plotly (JSON) form for the dashboard. `make_figure` returns both; callers
write the static form to `figures/{name}.{pdf,png}` and the interactive form
to `figures/{name}.json`.

Stage 1 T1.5 ships interface stubs per FigureKind; Stages 3-6 fill the
specific renderers as their experiments produce data.
"""

from __future__ import annotations

from typing import Any

from src.visualization.types import FigureKind, FigureSpec

# Figure registry: FigureKind -> (paper Fig number, source experiment, viz #)
FIGURE_REGISTRY: dict[FigureKind, dict[str, Any]] = {
    "persona_space_3d": {
        "fig_number": 1,
        "source_exp": "exp1_pca_decomposition",
        "viz_id": 1,
        "stage": 3,
        "renderer": "_render_persona_space_3d",
    },
    "safety_heatmap": {
        "fig_number": 2,
        "source_exp": "exp2_safety_relevance",
        "viz_id": 2,
        "stage": 3,
        "renderer": "_render_safety_heatmap",
    },
    "steering_comparator": {
        "fig_number": 3,
        "source_exp": "exp3_orthogonal_steering",
        "viz_id": 3,
        "stage": 4,
        "renderer": "_render_steering_comparator",
    },
    "layer_signal": {
        "fig_number": 4,
        "source_exp": "exp1_pca_decomposition",
        "viz_id": 4,
        "stage": 6,
        "renderer": "_render_layer_signal",
    },
    "persona_arithmetic": {
        "fig_number": 5,
        "source_exp": "exp5_composition",
        "viz_id": 5,
        "stage": 5,
        "renderer": "_render_persona_arithmetic",
    },
    "multi_axis_pareto": {
        "fig_number": 6,
        "source_exp": "exp6_multi_axis_defense",
        "viz_id": None,  # Multi-axis Pareto is paper-only; not in Viz 1-5 interactive set.
        "stage": 6,
        "renderer": "_render_multi_axis_pareto",
    },
}


def make_figure(spec: FigureSpec, data: Any) -> tuple[Any, Any]:
    """Render `data` per `spec.kind`. Returns `(matplotlib_fig, plotly_fig)`.

    Stage 1 ships the dispatch contract. Stage-specific renderers (Stage 3-6)
    fill in the per-kind drawing logic.
    """
    raise NotImplementedError(
        f"Renderer for FigureKind={spec.kind!r} is a Stage 3-6 deliverable. "
        f"Stage 1 ships the registry + dispatcher contract."
    )


def figure_paths(name: str, results_dir: str) -> tuple[str, str, str]:
    """Canonical output paths under `{results_dir}/figures/`.

    Returns `(pdf_path, png_path, plotly_json_path)`.
    """
    return (
        f"{results_dir}/figures/{name}.pdf",
        f"{results_dir}/figures/{name}.png",
        f"{results_dir}/figures/{name}.json",
    )
