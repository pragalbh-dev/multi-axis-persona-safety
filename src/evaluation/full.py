"""Phased experiment driver — orchestrates subject → safety → capability → cross-check.

Stage 1 T1.4 contract. Stage 2 T2.6 wires it together. Per CONVENTIONS
"Serving topology", each phase loads the right model on all 4 GPUs, does its
work, and tears down before the next phase loads.

Self-preference rule: when `cfg.model_id == cfg.judge_crosscheck_id` (i.e.
Gemma 4 31B is the subject AND the cross-check judge), skip the cross-check
phase. The driver enforces this so call sites don't have to.
"""

from __future__ import annotations

import pandas as pd

from src.evaluation.types import EvalResult
from src.utils.config import ExperimentConfig

# The order in which a fresh experiment runs; each phase tears down before next loads.
PHASES: list[str] = [
    "subject_rollouts",  # load subject -> generate responses -> stash to parquet
    "primary_judge",  # load Qwen 3.6-27B -> classify -> append harm_label_primary
    "capability",  # load subject -> run capability benchmarks -> append capability_score
    "cross_check_judge",  # load Gemma 4 31B -> classify 200-sample subset -> append harm_label_crosscheck
]


def eval_full(cfg: ExperimentConfig, *, prompts_df: pd.DataFrame) -> EvalResult:
    """Run the full phased pipeline for one experiment.

    Args:
        cfg: validated experiment config (already passed through
            `load_experiment_config`).
        prompts_df: per-prompt frame containing at least `prompt_id`,
            `dataset`, `input_text`. Steering/capping config from `cfg.steering`
            is applied during the subject phase.

    Returns: EvalResult with safety + capability sub-results, paths to the
    written `details.parquet` and `manifest.json`. Side-effects: writes the
    full per-prompt parquet matching `PER_PROMPT_COLUMNS` to `cfg.output_dir`.

    Stage 2 T2.6 deliverable. Stage 1 locks the contract.

    Implementation outline (Stage 2):
    1. `init_results_dir(cfg)` -> `(out_dir, manifest)` and resume detection.
    2. Phase: subject_rollouts.
        - Load subject vLLM instance (TP=4 bf16 from subjects.yaml).
        - For each (prompt, condition) cell, generate with optional
          `cap_and_steer` context.
        - Stash rows to `out_dir / 'rollouts.parquet'` with the columns the
          downstream phases need.
        - Tear down vLLM instance, free VRAM.
    3. Phase: primary_judge.
        - Load Qwen 3.6-27B (`judge_primary_id`) via `run_judge_batch`.
        - Append `harm_label_primary` and `harm_binary`.
        - Tear down.
    4. Phase: capability (if `cfg.capability_benchmarks` is non-empty).
        - Reload subject; run each benchmark; append capability rows.
        - Tear down.
    5. Phase: cross_check_judge — skipped if subject == cross-check.
        - Load Gemma 4 31B; classify the 200-sample subset; append
          `harm_label_crosscheck`.
        - Tear down.
    6. Compute per-dataset aggregates (`SafetyResult.harm_rate` + BCa CI) and
       per-benchmark `CapabilityResult.score`.
    7. Write `details.parquet` (final superset of `PER_PROMPT_COLUMNS`) and
       `metrics.json` (aggregates).
    8. Mark manifest done; write back.
    """
    raise NotImplementedError(
        "eval_full is a Stage 2 T2.6 deliverable. Stage 1 ships the phased contract."
    )


def should_run_crosscheck(cfg: ExperimentConfig) -> bool:
    """Cross-check phase is skipped when the subject is the cross-check judge."""
    return cfg.model_id != cfg.judge_crosscheck_id and cfg.crosscheck_subset_size > 0
