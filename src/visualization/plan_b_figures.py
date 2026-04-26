"""Plan B's 3 ad-hoc figures (matplotlib + Plotly).

These are NOT folded into the paper's FIGURE_REGISTRY (which maps to Fig 1..6
with different semantics). Plan B writes them directly to
`results/plan_b_gemma2_27b/figures/` for the fellowship-deadline writeup.
A post-deadline cleanup task can promote `blind_spot_summary` into
`fig5_blind_spot` of the registry once the schema fits.

Three figures:
  1. harm_rate_per_condition  — bar chart (the money plot)
  2. scree_plot               — eigenspectrum of role-vector PCA + MP threshold
  3. blind_spot_summary       — text card + dot of LASSO blind-spot lift
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np


def _ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def render_harm_rate_per_condition(
    per_condition: Mapping[str, Mapping[str, float]],
    out_dir: str | Path,
    *,
    title: str = "Harm rate per condition (Gemma 2 27B, DAN, n=500)",
) -> tuple[Path, Path]:
    """Bar chart with BCa CI error bars.

    Args:
        per_condition: {condition_id: {harm_rate, ci_low, ci_high, n_total}}
            condition_id ordering preserved from caller.
        out_dir: output directory (figures/ under Plan B results dir).

    Returns: (png_path, html_path)
    """
    import matplotlib.pyplot as plt
    import plotly.graph_objects as go

    out_dir = _ensure_dir(out_dir)
    conds = list(per_condition.keys())
    rates = [per_condition[c]["harm_rate"] for c in conds]
    lows = [per_condition[c]["harm_rate"] - per_condition[c]["ci_low"] for c in conds]
    highs = [per_condition[c]["ci_high"] - per_condition[c]["harm_rate"] for c in conds]

    # matplotlib (PNG)
    fig, ax = plt.subplots(figsize=(max(6, len(conds) * 0.7), 5))
    bars = ax.bar(conds, rates, yerr=[lows, highs], capsize=4, color="#4C72B0")
    # Highlight the baseline + AA-cap
    for i, c in enumerate(conds):
        if c == "baseline":
            bars[i].set_color("#666666")
        elif c == "aa_capped":
            bars[i].set_color("#55A868")
        elif c.startswith("random_"):
            bars[i].set_color("#C44E52")
    ax.set_ylabel("Harm rate (DAN)")
    ax.set_title(title)
    ax.set_ylim(0.0, max(1.0, max(per_condition[c]["ci_high"] for c in conds) + 0.05))
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    png_path = out_dir / "harm_rate_per_condition.png"
    fig.savefig(png_path, dpi=150)
    plt.close(fig)

    # Plotly (HTML)
    err_arr_low = np.array(lows)
    err_arr_high = np.array(highs)
    fig_p = go.Figure(
        data=[
            go.Bar(
                x=conds,
                y=rates,
                error_y=dict(
                    type="data",
                    symmetric=False,
                    array=err_arr_high,
                    arrayminus=err_arr_low,
                ),
                marker_color=[
                    "#666666"
                    if c == "baseline"
                    else "#55A868"
                    if c == "aa_capped"
                    else "#C44E52"
                    if c.startswith("random_")
                    else "#4C72B0"
                    for c in conds
                ],
                hovertemplate=(
                    "<b>%{x}</b><br>harm_rate=%{y:.3f}<br>CI=[%{customdata[0]:.3f},%{customdata[1]:.3f}]"
                    "<extra></extra>"
                ),
                customdata=[
                    [per_condition[c]["ci_low"], per_condition[c]["ci_high"]] for c in conds
                ],
            )
        ]
    )
    fig_p.update_layout(
        title=title,
        xaxis_title="condition",
        yaxis_title="harm rate (DAN)",
        template="plotly_white",
    )
    html_path = out_dir / "harm_rate_per_condition.html"
    fig_p.write_html(html_path, include_plotlyjs="cdn")
    return png_path, html_path


def render_scree_plot(
    explained_variance_ratio: np.ndarray,
    *,
    out_dir: str | Path,
    n_samples: int,
    d_model: int,
    title: str = "Role-space PCA scree (Gemma 2 27B, layer L*)",
    top_k: int = 30,
) -> tuple[Path, Path]:
    """Eigenspectrum + cumulative variance + MP threshold."""
    import matplotlib.pyplot as plt
    import plotly.graph_objects as go

    from src.analysis.pca import marchenko_pastur_threshold

    out_dir = _ensure_dir(out_dir)

    k = min(top_k, len(explained_variance_ratio))
    xs = np.arange(1, k + 1)
    var = explained_variance_ratio[:k]
    cum = np.cumsum(var)

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.bar(xs, var, color="#4C72B0", alpha=0.8, label="explained variance")
    ax1.set_xlabel("PC index")
    ax1.set_ylabel("explained variance ratio")
    ax2 = ax1.twinx()
    ax2.plot(xs, cum, color="#C44E52", marker="o", label="cumulative")
    ax2.set_ylabel("cumulative")
    ax2.set_ylim(0, 1.05)
    ax1.set_title(title)
    ax1.axhline(
        marchenko_pastur_threshold(d_model, n_samples) / max(explained_variance_ratio.sum(), 1e-9),
        color="grey",
        linestyle="--",
        linewidth=1,
        label="MP threshold",
    )
    ax1.legend(loc="upper right")
    plt.tight_layout()
    png_path = out_dir / "scree_plot.png"
    fig.savefig(png_path, dpi=150)
    plt.close(fig)

    fig_p = go.Figure()
    fig_p.add_trace(go.Bar(x=xs, y=var, name="explained variance", marker_color="#4C72B0"))
    fig_p.add_trace(
        go.Scatter(
            x=xs,
            y=cum,
            name="cumulative",
            mode="lines+markers",
            yaxis="y2",
            line=dict(color="#C44E52"),
        )
    )
    fig_p.update_layout(
        title=title,
        xaxis_title="PC index",
        yaxis=dict(title="explained variance ratio"),
        yaxis2=dict(title="cumulative", overlaying="y", side="right", range=[0, 1.05]),
        template="plotly_white",
    )
    html_path = out_dir / "scree_plot.html"
    fig_p.write_html(html_path, include_plotlyjs="cdn")
    return png_path, html_path


def render_blind_spot_summary(
    *,
    aa_cap_delta_pp: float,
    pc2_recovery_pp: float,
    pc3_recovery_pp: float,
    random_recovery_pp_max: float,
    blind_spot_auc_delta: float,
    blind_spot_ci_low: float,
    blind_spot_ci_high: float,
    out_dir: str | Path,
) -> tuple[Path, Path]:
    """Text card + dot plot of the H1 numerical claim."""
    import matplotlib.pyplot as plt
    import plotly.graph_objects as go

    out_dir = _ensure_dir(out_dir)

    summary = (
        f"AA-capping reduces harm rate by {aa_cap_delta_pp:.1f} pp.\n"
        f"+PC2 (λ=+2) recovers {pc2_recovery_pp:.1f} pp.\n"
        f"+PC3 (λ=+2) recovers {pc3_recovery_pp:.1f} pp.\n"
        f"5 random baselines (λ=+2) max recovery: {random_recovery_pp_max:.1f} pp.\n\n"
        f"Per-prompt LASSO blind-spot AUC delta = {blind_spot_auc_delta:.3f} "
        f"[95% CI {blind_spot_ci_low:.3f}, {blind_spot_ci_high:.3f}]"
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.text(0.05, 0.7, summary, fontsize=12, verticalalignment="top", family="monospace")
    # Dot plot of the recoveries
    cats = ["AA-cap Δ", "+PC2 recovery", "+PC3 recovery", "+random max"]
    vals = [aa_cap_delta_pp, pc2_recovery_pp, pc3_recovery_pp, random_recovery_pp_max]
    colors = ["#55A868", "#4C72B0", "#4C72B0", "#C44E52"]
    ax.scatter(vals, np.arange(len(vals)) + 0.5, c=colors, s=120, transform=ax.transAxes)
    for i, (cat, val) in enumerate(zip(cats, vals)):
        ax.text(0.55, 0.05 + i * 0.05, f"{cat}: {val:+.1f} pp", fontsize=10, transform=ax.transAxes)
    ax.set_axis_off()
    ax.set_title("Plan B blind-spot summary (Gemma 2 27B)")
    plt.tight_layout()
    png_path = out_dir / "blind_spot_summary.png"
    fig.savefig(png_path, dpi=150)
    plt.close(fig)

    fig_p = go.Figure(
        data=[
            go.Bar(
                x=cats,
                y=vals,
                marker_color=colors,
                hovertemplate="<b>%{x}</b>: %{y:+.1f} pp<extra></extra>",
            )
        ]
    )
    fig_p.add_annotation(
        text=(
            f"LASSO blind-spot AUC delta = {blind_spot_auc_delta:.3f} "
            f"[95% CI {blind_spot_ci_low:.3f}, {blind_spot_ci_high:.3f}]"
        ),
        x=0.5,
        y=1.12,
        xref="paper",
        yref="paper",
        showarrow=False,
    )
    fig_p.update_layout(
        title="Plan B — blind-spot summary (Gemma 2 27B, DAN)",
        yaxis_title="harm-rate change (percentage points)",
        template="plotly_white",
    )
    html_path = out_dir / "blind_spot_summary.html"
    fig_p.write_html(html_path, include_plotlyjs="cdn")
    return png_path, html_path
