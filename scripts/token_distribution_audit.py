"""Token-distribution audit across eval datasets + subject/judge tokenizers.

Stage 0 T0.2. Per CONVENTIONS: max_input_len and max_output_len per task must
come from empirical distribution of tokenized prompts, not guesses.

For each (subject_or_judge_tokenizer × eval_dataset) we report p50/p95/p99
*input* token length on a 200-sample subset. max_input_len is set to next
power-of-2 above p99 (rounded generously). max_output_len is task-semantic
(set below per task).

Writes configs/eval_sizes.yaml keyed (dataset, model_family).

Usage:
    uv run python scripts/token_distribution_audit.py
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils import env  # noqa: F401

import numpy as np  # noqa: E402
import yaml  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = PROJECT_ROOT / "data" / "eval"

# Subject + judge tokenizer sources. Use the *base* (unquantized) tokenizer —
# fp8 checkpoints share the same tokenizer as the base model; saves a gated
# download for our judge (Qwen 3.6-27B FP8 ships tokenizer anyway, but base is
# what our code will default to at text-input time).
TOKENIZERS = {
    "gemma_2_27b": "google/gemma-2-27b-it",       # gated; license accepted
    "qwen_3_32b": "Qwen/Qwen3-32B",
    "gemma_4_31b": "google/gemma-4-31B-it",       # needs trust_remote_code (config only)
    "qwen_3_6_27b": "Qwen/Qwen3.6-27B",
}

# Task-semantic output budgets (pre-audit; refined later if needed).
# Rationale: enough headroom to finish without truncation but not so much
# it wastes KV cache memory.
OUTPUT_LEN_BY_TASK = {
    "persona_jailbreak": 1024,       # long reasoning often elicited
    "ifeval": 1024,                  # instruction-following can be long
    "mmlu_pro_1400": 256,            # MC answer + brief chain-of-thought
    "gsm8k_1000": 512,               # multi-step arithmetic
    "eq_bench": 512,                 # emotional reasoning, moderate length
    "extraction_questions": 512,     # paper's default, matches generation.py
    "judge_9cat": 128,               # single-word label + brief rationale
    "judge_role_expression": 8,      # single digit 0-3
}

# Eval dataset prompt-extraction map. Shah persona_jailbreak absent.
DATASETS = {
    "ifeval": ("prompt", None),
    "mmlu_pro_1400": ("question", "options"),
    "gsm8k_1000": ("question", None),
    "eq_bench": ("prompt", None),
}

SAMPLE_N = 200
SEED = 42


def _load_prompts(name: str) -> list[str]:
    jsonl = EVAL_DIR / name / "prompts.jsonl"
    rows = [json.loads(line) for line in jsonl.read_text().splitlines() if line.strip()]
    prompt_key, options_key = DATASETS[name]
    out = []
    for r in rows:
        p = r.get(prompt_key, "")
        # MMLU-Pro: append options inline — this is what a batch prompt looks like
        if options_key and r.get(options_key):
            opts = r[options_key]
            if isinstance(opts, list):
                letters = "ABCDEFGHIJ"
                p = f"{p}\n\n" + "\n".join(f"{letters[i]}. {o}" for i, o in enumerate(opts))
        out.append(p)
    return out


def _load_extraction_questions() -> list[str]:
    data = json.loads((PROJECT_ROOT / "data" / "paper_artifacts" / "extraction_questions.json").read_text())
    return [q["question"] for q in data]


def _percentiles(lens: list[int]) -> dict:
    arr = np.array(lens)
    return {
        "n": len(arr),
        "min": int(arr.min()),
        "p50": int(np.percentile(arr, 50)),
        "p95": int(np.percentile(arr, 95)),
        "p99": int(np.percentile(arr, 99)),
        "max": int(arr.max()),
    }


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p *= 2
    return p


def main() -> None:
    rng = random.Random(SEED)
    # Load all tokenizers up front — fail fast if any auth issue.
    tokenizers = {}
    for family, hf_id in TOKENIZERS.items():
        print(f"loading tokenizer: {family} ← {hf_id}")
        tokenizers[family] = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)

    # Gather prompt sources
    sources: dict[str, list[str]] = {}
    for ds in DATASETS:
        prompts = _load_prompts(ds)
        if len(prompts) > SAMPLE_N:
            idx = sorted(rng.sample(range(len(prompts)), SAMPLE_N))
            prompts = [prompts[i] for i in idx]
        sources[ds] = prompts
    sources["extraction_questions"] = _load_extraction_questions()[:SAMPLE_N]

    # Tokenize every (dataset × tokenizer)
    config: dict = {"seed": SEED, "sample_n": SAMPLE_N, "tokenizers": TOKENIZERS, "entries": {}}
    for ds, prompts in sources.items():
        for family, tok in tokenizers.items():
            lens = [len(tok.encode(p, add_special_tokens=False)) for p in prompts]
            pct = _percentiles(lens)
            max_in = _next_pow2(pct["p99"] + 32)  # small buffer above p99
            out_len = OUTPUT_LEN_BY_TASK.get(ds, 512)
            key = f"{ds}::{family}"
            config["entries"][key] = {
                "dataset": ds,
                "model_family": family,
                "input_percentiles": pct,
                "max_input_len": max_in,
                "max_output_len": out_len,
                "max_model_len": max_in + out_len,
            }
            print(f"  {key}: p50={pct['p50']:4d} p95={pct['p95']:4d} p99={pct['p99']:4d} "
                  f"-> max_in={max_in}  out={out_len}  total={max_in+out_len}")

    out_path = PROJECT_ROOT / "configs" / "eval_sizes.yaml"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(config, sort_keys=False))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
