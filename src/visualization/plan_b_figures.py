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
    refusal_rate_per_condition: dict[str, float] | None = None,
    auc_aa_only: float | None = None,
    auc_with_pcs: float | None = None,
    selected_pcs: list[str] | None = None,
) -> tuple[Path, Path]:
    """Two-panel summary: behavioural-bypass (refusal-rate) + statistical lift (AUC).

    The original "harm-rate recovery" framing breaks at coherence-safe λ where PC
    attacks bypass the cap's *behaviour* (refusal pattern) without restoring
    information-level harm. So:
      Panel 1 — refusal-keyword incidence per condition (the bypass signal).
      Panel 2 — LASSO AUC: AA only vs AA+PC1..PC10 with CI on delta.

    `refusal_rate_per_condition` keys are condition_ids; values are 0..1.
    If unprovided, falls back to the legacy harm-rate-recovery bar plot.
    """
    import matplotlib.pyplot as plt
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    out_dir = _ensure_dir(out_dir)
    has_refusal = bool(refusal_rate_per_condition)

    # --- matplotlib version (PNG) ---------------------------------------------
    fig, axes = plt.subplots(1, 2 if has_refusal else 1, figsize=(12, 5) if has_refusal else (8, 5))
    if has_refusal:
        ax_ref, ax_auc = axes
        # Panel 1: refusal rate per condition
        order = [
            ("baseline", "baseline", "#999999"),
            ("aa_capped", "AA-cap", "#55A868"),
            ("aa_capped_pc2_pos0p25", "+PC2 (λ=0.25)", "#4C72B0"),
            ("aa_capped_pc3_pos0p25", "+PC3 (λ=0.25)", "#4C72B0"),
            ("aa_capped_random_0_pos0p25", "+random₀", "#C44E52"),
            ("aa_capped_random_1_pos0p25", "+random₁", "#C44E52"),
        ]
        cats_r = [lbl for k, lbl, _ in order if k in refusal_rate_per_condition]
        vals_r = [refusal_rate_per_condition[k] * 100 for k, _, _ in order if k in refusal_rate_per_condition]
        cols_r = [c for k, _, c in order if k in refusal_rate_per_condition]
        bars = ax_ref.bar(cats_r, vals_r, color=cols_r)
        ax_ref.set_ylabel("refusal-keyword rate (%)")
        ax_ref.set_title("Behavioural bypass — refusal-rate per condition")
        ax_ref.set_ylim(0, 100)
        for b, v in zip(bars, vals_r):
            ax_ref.text(b.get_x() + b.get_width() / 2, v + 2, f"{v:.0f}%", ha="center", fontsize=10)
        ax_ref.tick_params(axis="x", rotation=30)
        for tick in ax_ref.get_xticklabels():
            tick.set_horizontalalignment("right")

        # Panel 2: AUC bars + CI
        ax_auc.bar(["AA only", "AA + PC1..PC10"], [auc_aa_only or 0.0, auc_with_pcs or 0.0],
                   color=["#999999", "#4C72B0"])
        ax_auc.set_ylabel("AUC (binary harm prediction)")
        ax_auc.set_ylim(0.5, 1.0)
        ax_auc.set_title(
            f"Statistical signal — LASSO AUC\nΔ={blind_spot_auc_delta:+.3f} "
            f"[{blind_spot_ci_low:+.3f}, {blind_spot_ci_high:+.3f}]"
        )
        for i, v in enumerate([auc_aa_only or 0.0, auc_with_pcs or 0.0]):
            ax_auc.text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=11)
        if selected_pcs:
            ax_auc.annotate(
                f"selected PCs: {', '.join(selected_pcs)}",
                xy=(0.5, -0.18), xycoords="axes fraction", ha="center", fontsize=9,
                color="#555",
            )
        plt.suptitle(
            f"Plan B — blind-spot signal at coherence-safe λ (AA-cap Δharm = {aa_cap_delta_pp:+.1f} pp)",
            fontsize=12, y=1.02,
        )
    else:
        ax = axes
        # legacy fallback
        cats = ["AA-cap Δ", "+PC2 Δharm", "+PC3 Δharm", "+random Δharm"]
        vals = [aa_cap_delta_pp, pc2_recovery_pp, pc3_recovery_pp, random_recovery_pp_max]
        colors = ["#55A868", "#4C72B0", "#4C72B0", "#C44E52"]
        ax.bar(cats, vals, color=colors)
        ax.axhline(0, color="#333", linewidth=0.5)
        ax.set_ylabel("Δ harm-rate vs cap-only (pp)")
        ax.set_title("Plan B blind-spot summary (Gemma 2 27B)")
    plt.tight_layout()
    png_path = out_dir / "blind_spot_summary.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- plotly version (HTML) -------------------------------------------------
    if has_refusal:
        fig_p = make_subplots(
            rows=1, cols=2,
            subplot_titles=(
                "Behavioural bypass — refusal-rate per condition",
                f"Statistical signal — LASSO AUC (Δ={blind_spot_auc_delta:+.3f} "
                f"[{blind_spot_ci_low:+.3f}, {blind_spot_ci_high:+.3f}])",
            ),
            column_widths=[0.6, 0.4],
        )
        order = [
            ("baseline", "baseline", "#999999"),
            ("aa_capped", "AA-cap", "#55A868"),
            ("aa_capped_pc2_pos0p25", "+PC2 (λ=0.25)", "#4C72B0"),
            ("aa_capped_pc3_pos0p25", "+PC3 (λ=0.25)", "#4C72B0"),
            ("aa_capped_random_0_pos0p25", "+random₀", "#C44E52"),
            ("aa_capped_random_1_pos0p25", "+random₁", "#C44E52"),
        ]
        cats_r = [lbl for k, lbl, _ in order if k in refusal_rate_per_condition]
        vals_r = [refusal_rate_per_condition[k] * 100 for k, _, _ in order if k in refusal_rate_per_condition]
        cols_r = [c for k, _, c in order if k in refusal_rate_per_condition]
        fig_p.add_trace(
            go.Bar(x=cats_r, y=vals_r, marker_color=cols_r,
                   text=[f"{v:.0f}%" for v in vals_r], textposition="outside",
                   hovertemplate="<b>%{x}</b>: %{y:.1f}%<extra></extra>",
                   showlegend=False),
            row=1, col=1,
        )
        fig_p.update_yaxes(title_text="refusal-keyword rate (%)", range=[0, 110], row=1, col=1)
        fig_p.add_trace(
            go.Bar(x=["AA only", "AA + PC1..PC10"],
                   y=[auc_aa_only or 0.0, auc_with_pcs or 0.0],
                   marker_color=["#999999", "#4C72B0"],
                   text=[f"{auc_aa_only or 0.0:.3f}", f"{auc_with_pcs or 0.0:.3f}"],
                   textposition="outside",
                   hovertemplate="<b>%{x}</b>: AUC=%{y:.3f}<extra></extra>",
                   showlegend=False),
            row=1, col=2,
        )
        fig_p.update_yaxes(title_text="AUC (binary harm prediction)", range=[0.5, 1.05], row=1, col=2)
        fig_p.update_layout(
            title=f"Plan B — blind-spot signal at coherence-safe λ "
                  f"(AA-cap Δharm = {aa_cap_delta_pp:+.1f} pp on Gemma 2 27B, DAN, n=500/condition)",
            template="plotly_white", margin=dict(t=80, b=60),
        )
        if selected_pcs:
            fig_p.add_annotation(
                text=f"selected PCs: {', '.join(selected_pcs)}",
                x=1.0, y=-0.25, xref="x2 domain", yref="y2 domain",
                showarrow=False, xanchor="right", font=dict(size=10, color="#555"),
            )
    else:
        cats = ["AA-cap Δ", "+PC2 Δharm", "+PC3 Δharm", "+random Δharm"]
        vals = [aa_cap_delta_pp, pc2_recovery_pp, pc3_recovery_pp, random_recovery_pp_max]
        colors = ["#55A868", "#4C72B0", "#4C72B0", "#C44E52"]
        fig_p = go.Figure(
            data=[go.Bar(x=cats, y=vals, marker_color=colors,
                         hovertemplate="<b>%{x}</b>: %{y:+.1f} pp<extra></extra>")]
        )
        fig_p.add_annotation(
            text=(f"LASSO blind-spot AUC delta = {blind_spot_auc_delta:.3f} "
                  f"[95% CI {blind_spot_ci_low:.3f}, {blind_spot_ci_high:.3f}]"),
            x=0.5, y=1.12, xref="paper", yref="paper", showarrow=False,
        )
        fig_p.update_layout(
            title="Plan B — blind-spot summary (Gemma 2 27B, DAN)",
            yaxis_title="Δ harm-rate vs cap-only (pp)",
            template="plotly_white",
        )

    html_path = out_dir / "blind_spot_summary.html"
    fig_p.write_html(html_path, include_plotlyjs="cdn")
    return png_path, html_path
