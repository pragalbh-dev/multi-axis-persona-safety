"""Figure: cross-dataset baseline harm rate (DAN vs Shah) on Gemma 2 27B.

Computed from the merged baseline (Plan B 500 DAN + Phase 1 extended 1707).

Usage:
  uv run python -m src.visualization.cross_dataset_baseline
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def render(plan_b_details: Path, ext_details: Path, out_dir: Path, dpi: int = 150) -> None:
    pb = pd.read_parquet(plan_b_details)
    pb_baseline = pb[pb["condition_id"] == "baseline"].copy()
    pb_baseline["dataset"] = pb_baseline.get("dataset", "dan")
    ext = pd.read_parquet(ext_details)
    merged = pd.concat([pb_baseline[["dataset", "harm_binary"]], ext[["dataset", "harm_binary"]]], ignore_index=True)

    summary = (
        merged.groupby("dataset")
        .agg(n=("harm_binary", "size"), harm=("harm_binary", "mean"))
        .round(4)
    )
    print(summary)

    # 95% CI via Wilson interval (cleaner than normal approx for small p)
    def _wilson_ci(k: int, n: int) -> tuple[float, float]:
        from math import sqrt
        if n == 0:
            return 0.0, 0.0
        p = k / n
        z = 1.96
        denom = 1 + z * z / n
        center = (p + z * z / (2 * n)) / denom
        half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
        return max(0.0, center - half), min(1.0, center + half)

    cis = {}
    for ds, row in summary.iterrows():
        n = int(row["n"])
        k = int(row["harm"] * n)
        cis[ds] = _wilson_ci(k, n)

    fig, ax = plt.subplots(figsize=(7, 4), dpi=dpi)
    datasets = list(summary.index)
    rates = [float(summary.loc[d, "harm"]) for d in datasets]
    ns = [int(summary.loc[d, "n"]) for d in datasets]
    cis_lo = [cis[d][0] for d in datasets]
    cis_hi = [cis[d][1] for d in datasets]
    yerr = np.array([
        [r - lo for r, lo in zip(rates, cis_lo)],
        [hi - r for r, hi in zip(rates, cis_hi)],
    ])
    colors = ["#C44E52" if r > 0.10 else "#DD8452" for r in rates]

    bars = ax.bar(datasets, rates, color=colors, yerr=yerr, capsize=6,
                  error_kw={"linewidth": 1.0, "ecolor": "#444"})
    for d, r, n in zip(datasets, rates, ns):
        ax.text(d, r + 0.005, f"{r:.1%} (n={n})", ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("baseline harm rate")
    ax.set_title("Cross-dataset baseline harm on Gemma 2 27B (no cap, no steer)")
    ax.set_ylim(0, max(cis_hi) + 0.04)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    ax.text(
        0.5, -0.15,
        f"Wilson 95% CI shown. Total: n={sum(ns)} ({sum(int(summary.loc[d, 'harm']) * summary.loc[d, 'n'] for d in datasets):.0f} harm). "
        "Shah's lower rate reflects shorter, more direct prompt style vs DAN's elaborate jailbreaks.",
        transform=ax.transAxes, ha="center", fontsize=8, color="#555",
    )
    plt.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / "cross_dataset_baseline.png"
    fig.savefig(png_path, bbox_inches="tight")
    plt.close(fig)
    print(f"→ {png_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan-b-details", type=Path, default=Path("results/plan_b_gemma2_27b/details.parquet"))
    ap.add_argument("--ext-details", type=Path, default=Path("results/plan_b_gemma2_27b/extensions/baseline_extended.parquet"))
    ap.add_argument("--out-dir", type=Path, default=Path("results/plan_b_gemma2_27b/extensions/figures"))
    args = ap.parse_args()
    render(args.plan_b_details, args.ext_details, args.out_dir)
