"""Tests for src.evaluation.safety aggregation logic.

Synthetic data; no GPU, no model loading.
"""

from __future__ import annotations

import pandas as pd

from src.evaluation.safety import (
    HARM_BINARY_POSITIVE,
    HARM_LABELS_9CAT,
    aggregate_harm_rate,
    binarize_harm,
    derive_harm_binary_column,
    eval_safety,
    eval_safety_per_condition,
)


def test_binarize_harm_all_labels() -> None:
    for label in HARM_LABELS_9CAT:
        expected = 1 if label in HARM_BINARY_POSITIVE else 0
        assert binarize_harm(label) == expected


def _synth(n: int, n_harm: int, dataset: str, condition_id: str) -> pd.DataFrame:
    rows = []
    for i in range(n):
        is_harm = i < n_harm
        label = "enough_info" if is_harm else "refusal"
        rows.append({
            "prompt_id": f"{dataset}_{condition_id}_{i}",
            "dataset": dataset,
            "condition_id": condition_id,
            "harm_label_primary": label,
        })
    return pd.DataFrame(rows)


def test_eval_safety_per_dataset() -> None:
    df = pd.concat([
        _synth(40, 16, "dan", "baseline"),
        _synth(40, 8, "dan", "aa_capped"),
    ])
    df = derive_harm_binary_column(df)
    out = eval_safety(df, judge_cfg=None, judge_prompt_path="", datasets=["dan"])  # type: ignore[arg-type]
    assert "dan" in out
    # 24/80 harm = 0.30
    assert abs(out["dan"].harm_rate - 0.30) < 1e-9
    assert out["dan"].n_total == 80
    assert out["dan"].n_harm == 24
    # CI bounds bracket point estimate
    assert out["dan"].bca_ci_low <= out["dan"].harm_rate <= out["dan"].bca_ci_high


def test_eval_safety_per_condition() -> None:
    df = pd.concat([
        _synth(50, 25, "dan", "baseline"),
        _synth(50, 10, "dan", "aa_capped"),
        _synth(50, 30, "dan", "pc2_pos"),
    ])
    df = derive_harm_binary_column(df)
    out = eval_safety_per_condition(df)
    assert ("dan", "baseline") in out
    assert ("dan", "aa_capped") in out
    assert ("dan", "pc2_pos") in out
    assert abs(out[("dan", "baseline")].harm_rate - 0.50) < 1e-9
    assert abs(out[("dan", "aa_capped")].harm_rate - 0.20) < 1e-9
    assert abs(out[("dan", "pc2_pos")].harm_rate - 0.60) < 1e-9


def test_aggregate_handles_degenerate() -> None:
    # all 1s
    s = pd.Series([1, 1, 1, 1, 1])
    r = aggregate_harm_rate(s)
    assert r is not None
    assert r.harm_rate == 1.0
    assert r.bca_ci_low == 1.0
    # all 0s
    s = pd.Series([0, 0, 0, 0, 0])
    r = aggregate_harm_rate(s)
    assert r is not None
    assert r.harm_rate == 0.0


def test_aggregate_skips_when_empty() -> None:
    s = pd.Series([], dtype=float)
    assert aggregate_harm_rate(s) is None
