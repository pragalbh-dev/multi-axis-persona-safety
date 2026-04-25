"""Pydantic v2 schema for experiment configs.

Every src/experiments/exp{N}_{name}.py script loads its YAML config through
`load_experiment_config(path)`, which returns a validated `ExperimentConfig`.
Validators reach into the locked configs in `configs/` (subjects.yaml,
model_hooks.yaml, eval_sizes.yaml) so the per-experiment YAML stays small and
overrides are constrained.

Source-of-truth pointers:
- configs/subjects.yaml             — model + judge entries (TP, dtype, kwargs)
- configs/model_hooks.yaml          — n_layers + hook paths per family
- configs/eval_sizes.yaml           — (dataset, family) -> max_in / max_out
- configs/paper_capping_ranges.yaml — Tier-1 verbatim capping ranges
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml  # type: ignore[import-untyped]
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = REPO_ROOT / "configs"

DType = Literal["bf16", "fp16", "fp8"]
TokenAggregation = Literal["mean_response", "last", "all"]
SteeringMode = Literal["none", "addition", "ablation", "capping", "mean_ablation"]
Positions = Literal["all", "last"]
SafetyDataset = Literal["dan", "shah_reconstructed"]
CapabilityBenchmark = Literal["ifeval", "mmlu_pro", "gsm8k", "eq_bench"]


# ── Locked-config loaders (memoized at module level) ──────────────────────
def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data: dict[str, Any] = yaml.safe_load(f)
        return data


_SUBJECTS_CACHE: dict[str, Any] | None = None
_HOOKS_CACHE: dict[str, Any] | None = None
_EVAL_SIZES_CACHE: dict[str, Any] | None = None


def load_subjects() -> dict[str, Any]:
    global _SUBJECTS_CACHE
    if _SUBJECTS_CACHE is None:
        _SUBJECTS_CACHE = _load_yaml(CONFIGS_DIR / "subjects.yaml")
    return _SUBJECTS_CACHE


def load_model_hooks() -> dict[str, Any]:
    global _HOOKS_CACHE
    if _HOOKS_CACHE is None:
        _HOOKS_CACHE = _load_yaml(CONFIGS_DIR / "model_hooks.yaml")
    return _HOOKS_CACHE


def load_eval_sizes() -> dict[str, Any]:
    global _EVAL_SIZES_CACHE
    if _EVAL_SIZES_CACHE is None:
        _EVAL_SIZES_CACHE = _load_yaml(CONFIGS_DIR / "eval_sizes.yaml")
    return _EVAL_SIZES_CACHE


def model_family_for(model_id: str) -> str:
    """Map a subjects.yaml key to a model_hooks.yaml family key.

    The mapping is by prefix: gemma_2_27b -> gemma_2, qwen_3_32b -> qwen_3,
    gemma_4_31b -> gemma_4, qwen_3_6_27b -> qwen_3_6.
    """
    families = load_model_hooks()
    # longest-prefix match so "qwen_3_6" beats "qwen_3"
    for fam in sorted(families.keys(), key=len, reverse=True):
        if model_id.startswith(fam):
            return fam
    raise ValueError(f"No model family matches model_id={model_id!r}")


# ── Sub-models ────────────────────────────────────────────────────────────
class SteeringSubConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: SteeringMode = "none"
    vectors: list[str] = Field(default_factory=list)
    coefficients: list[float] = Field(default_factory=list)
    cap_thresholds: list[float] = Field(default_factory=list)
    # int per layer (steering) OR list[int] per range (capping); validated below
    layers: list[Any] = Field(default_factory=list)
    positions: Positions = "all"
    capping_config_ref: str | None = None

    @model_validator(mode="after")
    def _check_shape(self) -> SteeringSubConfig:
        if self.mode == "none":
            return self
        n = len(self.vectors)
        if n == 0:
            raise ValueError(f"steering.mode={self.mode} requires at least one vector")
        if self.coefficients and len(self.coefficients) != n:
            raise ValueError("steering.coefficients length must match vectors")
        if self.mode == "capping":
            if len(self.cap_thresholds) != n:
                raise ValueError("steering.cap_thresholds length must match vectors")
            for entry in self.layers:
                if not isinstance(entry, list) or len(entry) != 2:
                    raise ValueError(
                        "steering.layers entries must be [start, end] ranges when mode=capping"
                    )
        else:
            for entry in self.layers:
                if not isinstance(entry, int):
                    raise ValueError("steering.layers must be ints when mode != capping")
        return self


# ── Top-level config ──────────────────────────────────────────────────────
class ExperimentConfig(BaseModel):
    """Validated experiment config. Round-trip-stable via .model_dump()."""

    model_config = ConfigDict(extra="forbid")

    # Identity
    experiment_id: str
    seed: int = 42

    # Subject model
    model_id: str
    dtype: DType = "bf16"
    tensor_parallel: int = 4
    data_parallel: int = 1
    trust_remote_code: bool | None = None
    attention_backend: str | None = None
    chat_template_kwargs: dict[str, Any] | None = None

    # Extraction
    extraction_layer: int | None = None
    hook_point: str = ""
    token_aggregation: TokenAggregation = "mean_response"

    # Eval sizing
    max_input_len: int | None = None
    max_output_len: int | None = None
    batch_size: int | None = None

    # Judges
    judge_primary_id: str = "qwen_3_6_27b"
    judge_crosscheck_id: str = "gemma_4_31b"
    crosscheck_subset_size: int = 200
    judge_prompt_path: str = "configs/judge_prompt.yaml"
    role_expression_prompt_path: str = "configs/role_expression_prompt.yaml"

    # Datasets
    datasets: list[SafetyDataset] = Field(
        default_factory=lambda: list[SafetyDataset](["dan", "shah_reconstructed"])
    )
    capability_benchmarks: list[CapabilityBenchmark] = Field(
        default_factory=lambda: list[CapabilityBenchmark](
            ["ifeval", "mmlu_pro", "gsm8k", "eq_bench"]
        )
    )

    # Steering
    steering: SteeringSubConfig = Field(default_factory=SteeringSubConfig)

    # Output / resume
    output_dir: str
    resume_from_manifest: str | None = None
    fresh: bool = False

    # ── Validators ────────────────────────────────────────────────────────
    @field_validator("tensor_parallel")
    @classmethod
    def _check_tp(cls, v: int) -> int:
        if v not in {1, 2, 4}:
            raise ValueError(f"tensor_parallel must be 1, 2, or 4 (got {v})")
        return v

    @field_validator("model_id")
    @classmethod
    def _check_model_id(cls, v: str) -> str:
        if not v:
            raise ValueError("model_id is required")
        subjects = load_subjects()
        if v not in subjects:
            raise ValueError(
                f"model_id={v!r} not in configs/subjects.yaml (have {list(subjects)!r})"
            )
        return v

    @field_validator("judge_primary_id", "judge_crosscheck_id")
    @classmethod
    def _check_judge_id(cls, v: str) -> str:
        subjects = load_subjects()
        if v not in subjects:
            raise ValueError(
                f"judge id={v!r} not in configs/subjects.yaml (have {list(subjects)!r})"
            )
        return v

    @model_validator(mode="after")
    def _check_extraction_layer(self) -> ExperimentConfig:
        if self.extraction_layer is None:
            return self
        family = model_family_for(self.model_id)
        n_layers = int(load_model_hooks()[family]["n_layers"])
        if not (0 <= self.extraction_layer < n_layers):
            raise ValueError(
                f"extraction_layer={self.extraction_layer} out of range "
                f"[0, {n_layers}) for family {family}"
            )
        return self

    @model_validator(mode="after")
    def _check_eval_sizes_pair(self) -> ExperimentConfig:
        # Either both null (auto-lookup) or both set (override).
        a, b = self.max_input_len, self.max_output_len
        if (a is None) ^ (b is None):
            raise ValueError(
                "max_input_len and max_output_len must be both null (auto) or both set"
            )
        return self

    @model_validator(mode="after")
    def _check_self_preference(self) -> ExperimentConfig:
        # If subject == cross-check judge, the cross-check pass must be skipped.
        # This is enforced by the driver, but we surface a warning-shaped invariant.
        if self.model_id == self.judge_crosscheck_id and self.crosscheck_subset_size > 0:
            # Not an error; the driver will skip. But we record it in a hook so
            # callers can detect the no-op. (Pydantic doesn't have warnings; drivers
            # check the equality themselves.)
            pass
        return self

    # ── Resolvers (post-validation helpers; do not mutate the model) ─────
    def resolved_hook_point(self, layer: int | None = None) -> str:
        """Render the hook path for the resolved family."""
        if self.hook_point:
            return self.hook_point.format(L=layer if layer is not None else self.extraction_layer)
        family = model_family_for(self.model_id)
        hooks = load_model_hooks()[family]
        backend = hooks.get("preferred_backend", "transformer_lens")
        key = (
            "tl_hook_post_mlp_resid" if backend == "transformer_lens" else "nnsight_post_mlp_resid"
        )
        template: str = hooks[key]
        L = layer if layer is not None else self.extraction_layer
        if L is None:
            raise ValueError("resolved_hook_point requires layer or extraction_layer set")
        return template.replace("{L}", str(L))

    def resolved_eval_sizes(self, dataset: str) -> tuple[int, int]:
        """Return (max_input_len, max_output_len), inheriting from eval_sizes.yaml if null.

        Lookup key in eval_sizes.yaml is `f"{dataset}::{model_id}"` (full
        subject id, not family), matching the keys produced by Stage 0's
        scripts/token_distribution_audit.py.
        """
        if self.max_input_len is not None and self.max_output_len is not None:
            return self.max_input_len, self.max_output_len
        sizes = load_eval_sizes()
        key = f"{dataset}::{self.model_id}"
        entries = sizes.get("entries", {})
        if key not in entries:
            raise KeyError(
                f"No eval_sizes.yaml entry for {key!r}. Either add one or set "
                f"max_input_len/max_output_len explicitly in the experiment config."
            )
        e = entries[key]
        return int(e["max_input_len"]), int(e["max_output_len"])


# ── Public loaders ────────────────────────────────────────────────────────
def load_experiment_config(path: str | Path) -> ExperimentConfig:
    """Load and validate an experiment config from a YAML file."""
    p = Path(path)
    if not p.is_absolute():
        p = REPO_ROOT / p
    raw = _load_yaml(p)
    return ExperimentConfig.model_validate(raw)


def dump_experiment_config(cfg: ExperimentConfig, path: str | Path) -> None:
    """Dump a validated config back to YAML (used by init_results_dir)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg.model_dump(mode="json"), f, sort_keys=False)
