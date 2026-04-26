"""Smoke tests for Plan B figure renderers."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.visualization.plan_b_figures import (
    render_blind_spot_summary,
    render_harm_rate_per_condition,
    render_scree_plot,
)


def test_render_harm_rate_per_condition(tmp_path: Path) -> None:
    per_cond = {
        "baseline": {"harm_rate": 0.42, "ci_low": 0.36, "ci_high": 0.49, "n_total": 500},
        "aa_capped": {"harm_rate": 0.10, "ci_low": 0.07, "ci_high": 0.14, "n_total": 500},
        "pc2_pos2": {"harm_rate": 0.22, "ci_low": 0.17, "ci_high": 0.27, "n_total": 500},
        "random_0_pos2": {"harm_rate": 0.11, "ci_low": 0.07, "ci_high": 0.15, "n_total": 500},
    }
    png, html = render_harm_rate_per_condition(per_cond, tmp_path)
    assert png.exists() and png.stat().st_size > 0
    assert html.exists() and html.stat().st_size > 0


def test_render_scree_plot(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    var = rng.dirichlet(np.ones(20)).astype(np.float64)
    var = np.sort(var)[::-1]
    png, html = render_scree_plot(
        var, out_dir=tmp_path, n_samples=275, d_model=4608, top_k=15
    )
    assert png.exists()
    assert html.exists()


def test_render_blind_spot_summary(tmp_path: Path) -> None:
    png, html = render_blind_spot_summary(
        aa_cap_delta_pp=-32.0,
        pc2_recovery_pp=12.5,
        pc3_recovery_pp=8.1,
        random_recovery_pp_max=2.4,
        blind_spot_auc_delta=0.041,
        blind_spot_ci_low=0.013,
        blind_spot_ci_high=0.072,
        out_dir=tmp_path,
    )
    assert png.exists()
    assert html.exists()
