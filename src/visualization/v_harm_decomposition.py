"""Figure: v_harm decomposition onto AA + role-PCs.

Reads `harm_direction_merged.json` (canonical Ext A on n=2207 merged baseline).
Renders a single-panel bar chart with cos_sim(v_harm, axis) for axis ∈
{AA, PC1, PC2, ..., PC10}, color-coded by sign and with the argmax axis
highlighted. Saved as PNG + interactive HTML.

Usage:
  uv run python -m src.visualization.v_harm_decomposition
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def render(input_json: Path, out_dir: Path, dpi: int = 150) -> None:
    d = json.loads(input_json.read_text())
    cos_aa = float(d["cos_sim_v_harm_aa"])
    cos_pcs = list(d["cos_sim_v_harm_pcs"])
    argmax_name = d["argmax_axis_name"]
    n_total = int(d["n_total"])
    n_harm = int(d["n_harm"])

    labels = ["AA"] + [f"PC{i}" for i in range(1, len(cos_pcs) + 1)]
    values = [cos_aa] + cos_pcs

    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=dpi)
    bars = []
    for label, v in zip(labels, values):
        is_argmax = label == argmax_name
        color = "#C44E52" if v < 0 else "#55A868"  # red=anti-aligned, green=aligned
        edge = "black" if is_argmax else "none"
        lw = 2.0 if is_argmax else 0.0
        b = ax.bar(label, v, color=color, edgecolor=edge, linewidth=lw)
        bars.append(b)

    ax.axhline(0.0, color="#888", linewidth=0.8)
    ax.set_ylabel("cos_sim(v_harm, axis)")
    ax.set_title(
        f"v_harm decomposition onto AA + role-PCA basis\n"
        f"merged DAN+Shah baseline (n={n_total}, harm={n_harm}); argmax axis: {argmax_name}",
        fontsize=11,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", rotation=0)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    # Annotate values near bars
    for label, v in zip(labels, values):
        offset = 0.02 if v >= 0 else -0.02
        va = "bottom" if v >= 0 else "top"
        ax.text(label, v + offset, f"{v:+.2f}", ha="center", va=va, fontsize=8)

    # Footer note
    ax.text(
        0.02, -0.18,
        (
            f"AUC ⟨h, v_harm⟩ = {d['single_dir_auc_v_harm']:.3f}   |   "
            f"AUC ⟨h, AA⟩ = {d['single_dir_auc_aa']:.3f}   |   "
            f"residual outside top-10 PCs: {d['residual_outside_top_k']['10']:.3f}   |   "
            f"red = anti-aligned with v_harm (sign=−1 attack), "
            f"green = aligned (sign=+1 attack)"
        ),
        transform=ax.transAxes,
        fontsize=8,
        color="#555",
    )
    plt.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / "v_harm_decomposition.png"
    fig.savefig(png_path, bbox_inches="tight")
    plt.close(fig)
    print(f"→ {png_path}")

    # Plotly version for interactive figure
    try:
        import plotly.graph_objects as go

        colors = ["#C44E52" if v < 0 else "#55A868" for v in values]
        edge_widths = [2.0 if l == argmax_name else 0.0 for l in labels]
        fig_p = go.Figure(
            data=[
                go.Bar(
                    x=labels,
                    y=values,
                    marker={
                        "color": colors,
                        "line": {"color": "black", "width": edge_widths},
                    },
                    text=[f"{v:+.3f}" for v in values],
                    textposition="outside",
                )
            ]
        )
        fig_p.update_layout(
            title=(
                f"v_harm decomposition (n={n_total}, harm={n_harm}); argmax = {argmax_name}<br>"
                f"<sub>AUC v_harm={d['single_dir_auc_v_harm']:.3f} vs AA={d['single_dir_auc_aa']:.3f}; "
                f"red = anti-aligned (sign=−1 attack), green = aligned (sign=+1)</sub>"
            ),
            yaxis_title="cos_sim(v_harm, axis)",
            template="plotly_white",
            height=480,
            margin={"l": 60, "r": 30, "t": 80, "b": 60},
        )
        fig_p.add_hline(y=0, line_color="#888", line_width=0.8)
        html_path = out_dir / "v_harm_decomposition.html"
        fig_p.write_html(str(html_path), include_plotlyjs="cdn")
        print(f"→ {html_path}")
    except ImportError:
        print("plotly not installed; skipping HTML")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        type=Path,
        default=Path("results/plan_b_gemma2_27b/extensions/harm_direction_merged.json"),
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/plan_b_gemma2_27b/extensions/figures"),
    )
    args = ap.parse_args()
    render(args.input, args.out_dir)
