"""Capability eval — IFEval / MMLU Pro / GSM8k / EQ-Bench.

Stage 1 T1.4 contract. Stage 3 T3.5.5 (capability baselines) is the first
real consumer; Stage 4-6 reuse it for steering/capping deltas.

Per benchmark we pick `(max_input_len, max_output_len)` from
`configs/eval_sizes.yaml` keyed `f"{dataset}::{model_id}"`. Eval dataset
parquets live at `data/eval/{ifeval,mmlu_pro_1400,gsm8k_1000,eq_bench}/`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from src.evaluation.types import CapabilityResult

Benchmark = Literal["ifeval", "mmlu_pro", "gsm8k", "eq_bench"]


def eval_capability(
    model: Any,
    benchmark: Benchmark,
    *,
    model_id: str,
    eval_data_root: str | Path = "data/eval",
    max_input_len: int | None = None,
    max_output_len: int | None = None,
    seed: int = 42,
) -> CapabilityResult:
    """Run one capability benchmark and return rows + aggregate score.

    Stage 3 T3.5.5 deliverable. Stage 1 ships the dispatch contract.

    Args:
        model: vLLM `LLM` instance (subject already loaded by the phased driver).
        benchmark: which benchmark to run.
        model_id: subjects.yaml key — used to resolve eval-size defaults.
        eval_data_root: where the dataset parquets live.
        max_input_len / max_output_len: optional override; default looked up
            from `configs/eval_sizes.yaml`.
        seed: Sampling seed.

    Returns: CapabilityResult with `rows` (parquet-shaped per-prompt frame),
    `n` (sample size), `score` (aggregate metric per benchmark), BCa CI.

    Per-benchmark scoring (paper-aligned):
    - IFEval: instruction-following pass rate (strict + loose averaged).
    - MMLU Pro: multiple-choice accuracy on the 1,400-sample subset.
    - GSM8k: exact-match on numeric answer (extracted from #### marker).
    - EQ-Bench: continuous score per the upstream rubric on validation split.

    Stage 3 T3.5.5 is the implementer; this stub raises NotImplementedError.
    """
    raise NotImplementedError(
        f"eval_capability for {benchmark!r} is a Stage 3 T3.5.5 deliverable. "
        f"Stage 1 ships the dispatch contract."
    )


def benchmark_dataset_path(benchmark: Benchmark, eval_data_root: str | Path) -> Path:
    """Where each benchmark's parquet lives. Matches Stage 0 download layout."""
    root = Path(eval_data_root)
    return {
        "ifeval": root / "ifeval" / "prompts.parquet",
        "mmlu_pro": root / "mmlu_pro_1400" / "prompts.parquet",
        "gsm8k": root / "gsm8k_1000" / "prompts.parquet",
        "eq_bench": root / "eq_bench" / "prompts.parquet",
    }[benchmark]
