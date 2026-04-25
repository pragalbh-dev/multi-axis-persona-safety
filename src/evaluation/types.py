"""Evaluation result types.

Stage 1 T1.4 locks the per-prompt details.parquet row schema (see
`PER_PROMPT_COLUMNS`) so downstream stages and Viz 6 can target a fixed shape.

Re-exports `JudgeConfig` from `src.evaluation.judge_batch` for convenience.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from src.evaluation.judge_batch import JudgeConfig

__all__ = [
    "JudgeConfig",
    "PER_PROMPT_COLUMNS",
    "SafetyResult",
    "CapabilityResult",
    "EvalResult",
]


# Locked Stage 1 T1.4. Every safety / capability eval writes a parquet whose
# columns are a SUPERSET of these. Drivers raise if a required column is
# missing at write-time.
PER_PROMPT_COLUMNS: list[str] = [
    "prompt_id",
    "dataset",  # "dan" | "shah_reconstructed" | capability id
    "condition_id",  # str hash of (model_id, steering_mode, lambda, defense_config)
    "model_id",
    "steering_mode",  # "none" | "addition" | "ablation" | "capping" | "mean_ablation"
    "lambda",  # float; coefficient for steering, NaN for none/capping
    "defense_config",  # str | null (e.g. "aa_cap_p25" or "multi_axis_aa+pc2+pc3")
    "layer",  # int; extraction layer used for projections
    "input_text",
    "response_text",
    "response_tokens",  # int (token count of response_text)
    "aa_projection",  # float
    "pc_projections",  # JSON-serialized list[float] (PC1..PCk)
    "harm_label_primary",
    "harm_label_crosscheck",  # str | null (cross-check skipped or self-pref)
    "harm_binary",  # int 0/1
    "capability_score",  # float | null (only set when row originates from a capability bench)
    "seed",
    "git_sha",
    "run_iso",
]


@dataclass
class SafetyResult:
    dataset: str
    rows: pd.DataFrame
    n_total: int
    n_harm: int
    harm_rate: float
    bca_ci_low: float
    bca_ci_high: float

    @property
    def asr(self) -> float:
        """Attack success rate is harm_rate; alias for paper-style reporting."""
        return self.harm_rate


@dataclass
class CapabilityResult:
    benchmark: str  # "ifeval" | "mmlu_pro" | "gsm8k" | "eq_bench"
    rows: pd.DataFrame
    n: int
    score: float
    bca_ci_low: float
    bca_ci_high: float


@dataclass
class EvalResult:
    safety: dict[str, SafetyResult] = field(default_factory=dict)
    capability: dict[str, CapabilityResult] = field(default_factory=dict)
    details_path: Path | None = None
    manifest_path: Path | None = None
