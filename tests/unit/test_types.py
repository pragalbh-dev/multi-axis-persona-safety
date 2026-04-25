"""Stage 1 T1.1 — happy-path instantiation + safetensors round-trip."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.analysis import BlindSpotLift, BootstrapResult, CorrelationResult, LassoFit
from src.evaluation import (
    PER_PROMPT_COLUMNS,
    CapabilityResult,
    EvalResult,
    JudgeConfig,
    SafetyResult,
)
from src.extraction import ActivationCache, ExtractionConfig
from src.steering import SteeringConfig
from src.visualization import FigureSpec


def test_per_prompt_columns_locked() -> None:
    # Stage 1 T1.4 contract — adding a column is a schema migration.
    assert "prompt_id" in PER_PROMPT_COLUMNS
    assert "harm_binary" in PER_PROMPT_COLUMNS
    assert "aa_projection" in PER_PROMPT_COLUMNS
    assert "pc_projections" in PER_PROMPT_COLUMNS
    assert len(PER_PROMPT_COLUMNS) == 20


def test_extraction_config_minimal() -> None:
    cfg = ExtractionConfig(
        model_id="gemma_2_27b", layers=[22], hook_point="blocks.{L}.hook_resid_post"
    )
    assert cfg.token_aggregation == "mean_response"
    assert cfg.dtype == "bf16"


def test_activation_cache_roundtrip(tmp_path: Path) -> None:
    t = torch.arange(48, dtype=torch.float32).reshape(4, 12)
    cache = ActivationCache(
        model_id="gemma_2_27b",
        dataset="extraction_questions",
        layer=22,
        tensor=t,
        prompt_ids=[f"p{i}" for i in range(4)],
    )
    stem = tmp_path / "L22"
    cache.save(stem)
    assert (stem.with_suffix(".safetensors")).is_file()
    assert (stem.with_suffix(".meta.json")).is_file()
    loaded = ActivationCache.load(stem)
    assert torch.equal(loaded.tensor, t)
    assert loaded.prompt_ids == cache.prompt_ids
    assert loaded.layer == 22
    assert loaded.token_aggregation == "mean_response"


def test_steering_config_capping_requires_thresholds() -> None:
    with pytest.raises(ValueError, match="cap_thresholds"):
        SteeringConfig(
            vectors=[torch.zeros(8)],
            intervention_type="capping",
            layer_indices=[(2, 5)],
        )


def test_steering_config_capping_requires_range() -> None:
    with pytest.raises(ValueError, match="ranges"):
        SteeringConfig(
            vectors=[torch.zeros(8)],
            intervention_type="capping",
            layer_indices=[3],  # int, not (start, end)
            cap_thresholds=[0.5],
        )


def test_steering_config_addition_requires_int_layers() -> None:
    with pytest.raises(ValueError, match="ints"):
        SteeringConfig(
            vectors=[torch.zeros(8)],
            intervention_type="addition",
            layer_indices=[(2, 5)],  # range, but mode is non-capping
            coefficients=[1.0],
        )


def test_safety_result_construction() -> None:
    import pandas as pd

    df = pd.DataFrame({"prompt_id": ["p1"], "harm_binary": [1]})
    res = SafetyResult(
        dataset="dan",
        rows=df,
        n_total=1,
        n_harm=1,
        harm_rate=1.0,
        bca_ci_low=0.5,
        bca_ci_high=1.0,
    )
    assert res.asr == res.harm_rate == 1.0


def test_capability_result_construction() -> None:
    import pandas as pd

    res = CapabilityResult(
        benchmark="ifeval",
        rows=pd.DataFrame({"score": [0.7]}),
        n=1,
        score=0.7,
        bca_ci_low=0.5,
        bca_ci_high=0.9,
    )
    assert res.benchmark == "ifeval"


def test_eval_result_default_is_empty() -> None:
    er = EvalResult()
    assert er.safety == {} and er.capability == {}


def test_judge_config_default() -> None:
    jc = JudgeConfig(hf_id="Qwen/Qwen3.6-27B")
    assert jc.tensor_parallel_size == 2
    assert jc.max_output_len == 128


def test_bootstrap_result_str() -> None:
    br = BootstrapResult(point=0.7, ci_low=0.6, ci_high=0.8, n_resamples=10000)
    s = str(br)
    assert "0.7000" in s


def test_lasso_fit_stub() -> None:
    fit = LassoFit(
        coefs={"aa": 1.0, "pc2": 0.0},
        auc=0.85,
        auc_ci=BootstrapResult(point=0.85, ci_low=0.8, ci_high=0.9),
        selected_features=["aa"],
    )
    assert fit.selected_features == ["aa"]


def test_blind_spot_lift_stub() -> None:
    lift = BlindSpotLift(
        auc_aa_only=0.7,
        auc_with_pcs=0.74,
        delta=BootstrapResult(point=0.04, ci_low=0.01, ci_high=0.07),
    )
    assert lift.auc_with_pcs > lift.auc_aa_only


def test_correlation_result_stub() -> None:
    cr = CorrelationResult(stat="point_biserial", r=0.4, p=0.001, n=200)
    assert cr.stat == "point_biserial"


def test_figure_spec_construction(tmp_path: Path) -> None:
    fs = FigureSpec(
        name="persona_space",
        kind="persona_space_3d",
        source_exp="results/exp1_pca_decomposition",
        static_path=tmp_path / "fig.pdf",
        interactive_path=tmp_path / "fig.json",
        fig_number=1,
    )
    assert fs.kind == "persona_space_3d"
