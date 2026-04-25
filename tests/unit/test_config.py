"""Stage 1 T1.6.5 — pydantic v2 ExperimentConfig validation.

Covers: template loads, unknown model_id rejected, TP enum, layer-range bound,
half-set eval-size pair rejected, dataset enum, eval-size + hook-point resolution.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.utils.config import ExperimentConfig, load_experiment_config


def _base() -> dict:
    return {
        "experiment_id": "_t_",
        "model_id": "gemma_2_27b",
        "output_dir": "_t_",
    }


def test_template_loads() -> None:
    cfg = load_experiment_config("configs/experiment_template.yaml")
    assert cfg.dtype == "bf16"
    assert cfg.tensor_parallel == 4
    assert cfg.datasets == ["dan", "shah_reconstructed"]


def test_unknown_model_id_rejected() -> None:
    with pytest.raises(ValidationError, match="subjects.yaml"):
        ExperimentConfig(**{**_base(), "model_id": "not-a-real-id"})


def test_unknown_judge_id_rejected() -> None:
    with pytest.raises(ValidationError, match="subjects.yaml"):
        ExperimentConfig(**{**_base(), "judge_primary_id": "nope"})


def test_tensor_parallel_enum() -> None:
    with pytest.raises(ValidationError, match="must be 1, 2, or 4"):
        ExperimentConfig(**{**_base(), "tensor_parallel": 3})


def test_extraction_layer_out_of_range() -> None:
    with pytest.raises(ValidationError, match="out of range"):
        ExperimentConfig(**{**_base(), "extraction_layer": 999})


def test_extraction_layer_in_range_ok() -> None:
    # gemma_2 has 46 layers per configs/model_hooks.yaml
    cfg = ExperimentConfig(**{**_base(), "extraction_layer": 22})
    assert cfg.extraction_layer == 22


def test_partial_eval_sizes_rejected() -> None:
    with pytest.raises(ValidationError, match="both null"):
        ExperimentConfig(**{**_base(), "max_input_len": 256, "max_output_len": None})


def test_unknown_dataset_rejected() -> None:
    with pytest.raises(ValidationError):
        ExperimentConfig(**{**_base(), "datasets": ["unknown_dataset"]})


def test_unknown_capability_bench_rejected() -> None:
    with pytest.raises(ValidationError):
        ExperimentConfig(**{**_base(), "capability_benchmarks": ["unknown_bench"]})


def test_hook_point_resolves() -> None:
    cfg = ExperimentConfig(**{**_base(), "extraction_layer": 22})
    assert cfg.resolved_hook_point() == "blocks.22.hook_resid_post"
    assert cfg.resolved_hook_point(layer=10) == "blocks.10.hook_resid_post"


def test_eval_sizes_resolve_from_yaml() -> None:
    cfg = ExperimentConfig(**_base())
    assert cfg.resolved_eval_sizes("ifeval") == (256, 1024)
    assert cfg.resolved_eval_sizes("gsm8k_1000") == (256, 512)
