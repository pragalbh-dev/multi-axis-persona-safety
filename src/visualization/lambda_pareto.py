"""Figure: per-axis λ-coherence Pareto curves (Phase 3 attack arm output).

Reads `lambda_pareto.json` from Phase 3 and renders a 2-panel figure:
  Left:  harm rate vs λ per axis
  Right: coherence (1-nonsense) vs λ per axis, with horizontal floor line
Highlights chosen_lambda_per_axis on each curve.

Usage:
  uv run python -m src.visualization.lambda_pareto
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


# Stable color per axis
AXIS_COLORS = {
    "signmatched_pc2": "#C44E52",   # red
    "signmatched_pc3": "#DD8452",   # orange
    "signmatched_pc4": "#8172B3",   # purple
    "v_harm": "#4C72B0",            # blue
    "aa_control": "#55A868",        # green
    "random_0": "#888888",
    "random_1": "#aaaaaa",
}


def render(input_json: Path, out_dir: Path, coherence_floor: float = 0.80, dpi: int = 150) -> None:
    d = json.loads(input_json.read_text())
    pareto = d["pareto"]
    chosen = d.get("chosen_lambda_per_axis", {})

    fig, (ax_h, ax_c) = plt.subplots(1, 2, figsize=(12, 4.5), dpi=dpi, sharex=True)

    for axis, points in pareto.items():
        if not points:
            continue
        color = AXIS_COLORS.get(axis, "#444")
        lams = [float(p["lam"]) for p in points]
        harms = [float(p["harm"]) for p in points]
        cohs = [float(p["coherence"]) for p in points]
        label = axis.replace("signmatched_", "sm-")

        ax_h.plot(lams, harms, marker="o", linewidth=1.5, color=color, label=label)
        ax_c.plot(lams, cohs, marker="o", linewidth=1.5, color=color, label=label)

        # Mark chosen λ
        if axis in chosen:
            cl = float(chosen[axis])
            for p in points:
                if abs(float(p["lam"]) - cl) < 1e-6:
                    ax_h.plot([cl], [float(p["harm"])], marker="*", markersize=14,
                              color=color, markeredgecolor="black", markeredgewidth=0.8)
                    ax_c.plot([cl], [float(p["coherence"])], marker="*", markersize=14,
                              color=color, markeredgecolor="black", markeredgewidth=0.8)
                    break

    ax_c.axhline(coherence_floor, color="#888", linestyle="--", linewidth=1.0,
                 label=f"coherence floor ({coherence_floor:.0%})")
    ax_h.set_xlabel("steering λ")
    ax_c.set_xlabel("steering λ")
    ax_h.set_ylabel("harm rate")
    ax_c.set_ylabel("coherence (1 − nonsense rate)")
    ax_h.set_title("Harm rate vs λ (per attack axis on AA-capped model)")
    ax_c.set_title("Coherence vs λ (per attack axis)")
    for ax in (ax_h, ax_c):
        ax.grid(linestyle="--", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(loc="best", fontsize=8, frameon=False)

    fig.suptitle(
        "Adaptive λ characterization: per-axis dose-response on AA-capped Gemma 2 27B "
        "(★ = chosen λ_max_coherent)",
        fontsize=11,
    )
    plt.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / "lambda_pareto.png"
    fig.savefig(png_path, bbox_inches="tight")
    plt.close(fig)
    print(f"→ {png_path}")

    # Plotly interactive
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        fig_p = make_subplots(rows=1, cols=2, subplot_titles=["harm vs λ", "coherence vs λ"])
        for axis, points in pareto.items():
            if not points:
                continue
            color = AXIS_COLORS.get(axis, "#444")
            lams = [float(p["lam"]) for p in points]
            harms = [float(p["harm"]) for p in points]
            cohs = [float(p["coherence"]) for p in points]
            label = axis.replace("signmatched_", "sm-")
            fig_p.add_trace(
                go.Scatter(x=lams, y=harms, mode="lines+markers", name=label, marker={"color": color}, legendgroup=axis),
                row=1, col=1,
            )
            fig_p.add_trace(
                go.Scatter(x=lams, y=cohs, mode="lines+markers", name=label, marker={"color": color}, legendgroup=axis, showlegend=False),
                row=1, col=2,
            )
        fig_p.add_hline(y=coherence_floor, line_dash="dash", line_color="#888", row=1, col=2)
        fig_p.update_xaxes(title="λ")
        fig_p.update_yaxes(title="harm rate", row=1, col=1)
        fig_p.update_yaxes(title="coherence", row=1, col=2)
        fig_p.update_layout(template="plotly_white", height=480, title="λ-coherence Pareto per attack axis")
        html_path = out_dir / "lambda_pareto.html"
        fig_p.write_html(str(html_path), include_plotlyjs="cdn")
        print(f"→ {html_path}")
    except ImportError:
        pass


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        type=Path,
        default=Path("results/plan_b_gemma2_27b/extensions/attack_arm/lambda_pareto.json"),
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/plan_b_gemma2_27b/extensions/figures"),
    )
    args = ap.parse_args()
    render(args.input, args.out_dir)
