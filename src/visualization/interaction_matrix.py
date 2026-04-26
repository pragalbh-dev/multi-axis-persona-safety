"""Figure: defence × attack interaction matrix heatmap (Phase 4 output).

Reads `interaction_matrix.parquet` (per (defence, attack, prompt) row from
Phase 4) and renders a 3-panel heatmap:
  Left:   harm rate by (defence × attack)
  Middle: refusal-family rate by (defence × attack)
  Right:  coherence (1 − nonsense rate) by (defence × attack)

Each cell annotated with the value. Defences ordered by increasing axis count;
attacks ordered by Phase 3 attack strength (with 'zero' baseline first).

Usage:
  uv run python -m src.visualization.interaction_matrix
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def render(input_parquet: Path, out_dir: Path, dpi: int = 150) -> None:
    df = pd.read_parquet(input_parquet)

    df["coherence"] = 1.0 - df["nonsense"]

    summary = (
        df.groupby(["defence", "attack"])
        .agg(
            harm=("harm_binary", "mean"),
            refusal=("refusal_family", "mean"),
            coherence=("coherence", "mean"),
        )
        .round(3)
    )

    # Ordering
    defence_order = ["aa_only", "aa_pc2", "aa_pc2_pc3_pc4", "aa_v_harm"]
    attack_order = [
        "zero",
        "aa_control",
        "signmatched_pc2",
        "signmatched_pc3",
        "signmatched_pc4",
        "v_harm",
        "multi_signmatched",
    ]
    defences = [d for d in defence_order if d in summary.index.get_level_values("defence").unique()]
    attacks = [a for a in attack_order if a in summary.index.get_level_values("attack").unique()]

    def _pivot(metric: str) -> np.ndarray:
        m = np.full((len(defences), len(attacks)), np.nan)
        for i, d in enumerate(defences):
            for j, a in enumerate(attacks):
                if (d, a) in summary.index:
                    m[i, j] = float(summary.loc[(d, a), metric])
        return m

    h = _pivot("harm")
    r = _pivot("refusal")
    c = _pivot("coherence")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), dpi=dpi)
    cmaps = ["Reds", "Greens", "Blues"]
    titles = ["harm rate", "refusal rate", "coherence (1−nonsense)"]
    matrices = [h, r, c]

    for ax, m, cmap, title in zip(axes, matrices, cmaps, titles):
        im = ax.imshow(m, cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto")
        ax.set_xticks(range(len(attacks)))
        ax.set_yticks(range(len(defences)))
        ax.set_xticklabels([a.replace("signmatched_", "sm-") for a in attacks], rotation=45, ha="right")
        ax.set_yticklabels(defences)
        ax.set_title(title, fontsize=11)
        for i in range(len(defences)):
            for j in range(len(attacks)):
                v = m[i, j]
                if np.isnan(v):
                    continue
                # Choose text color based on cell brightness
                txt_color = "white" if v > 0.55 else "black"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8, color=txt_color)
        plt.colorbar(im, ax=ax, fraction=0.04)

    fig.suptitle(
        "Defence × Attack interaction matrix on Gemma 2 27B  (n=486 stratified subset)",
        fontsize=12,
    )
    plt.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / "interaction_matrix.png"
    fig.savefig(png_path, bbox_inches="tight")
    plt.close(fig)
    print(f"→ {png_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        type=Path,
        default=Path("results/plan_b_gemma2_27b/extensions/defence_arm/interaction_matrix.parquet"),
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/plan_b_gemma2_27b/extensions/figures"),
    )
    args = ap.parse_args()
    render(args.input, args.out_dir)
