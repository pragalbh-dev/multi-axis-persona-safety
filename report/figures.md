# Figure Registry

Source-of-truth: `src/visualization/figures.py::FIGURE_REGISTRY`. This file is
the human-readable mirror used during paper-writing.

| Fig | Name | Viz # | Stage | Source experiment | Renderer (Stage 3-6 fills) |
| --- | --- | --- | --- | --- | --- |
| 1 | Persona Space Explorer (3D PCA) | Viz 1 | 3 | `results/exp1_pca_decomposition/` | `_render_persona_space_3d` |
| 2 | Safety Heatmap Across PCs | Viz 2 | 3 | `results/exp2_safety_relevance/` | `_render_safety_heatmap` |
| 3 | Steering Effect Comparator | Viz 3 | 4 | `results/exp3_orthogonal_steering/` + `exp4_blind_spot/` | `_render_steering_comparator` |
| 4 | Layer-by-Layer Persona Signal | Viz 4 | 6 | `results/exp1_pca_decomposition/` (+ ablations) | `_render_layer_signal` |
| 5 | Persona Arithmetic | Viz 5 | 5 | `results/exp5_composition/` | `_render_persona_arithmetic` |
| 6 | Multi-Axis Defense Pareto | (paper-only) | 6 | `results/exp6_multi_axis_defense/` | `_render_multi_axis_pareto` |

Notes:

- Viz 1-5 ship interactive Plotly forms in the dashboard (Stage 8). Fig 6
  (Pareto curve) is paper-only because the dashboard's Persona Steering
  Playground (Viz 6) covers it experientially via the defense toggle.
- Static (matplotlib) outputs land at
  `results/expN_name/figures/{fig_name}.{pdf,png}`; interactive (plotly JSON)
  at `results/expN_name/figures/{fig_name}.json`. Path helper:
  `src.visualization.figures.figure_paths(name, results_dir)`.
- Each figure caption in `paper.md` should cite the exact `metrics.json` keys
  it visualizes, so a future reviewer can audit numbers from JSON without
  rerunning code.
