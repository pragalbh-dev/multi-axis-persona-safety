"""Batch judge driver: load a judge model via vLLM, classify a parquet of
(prompt, response) rows, append labels, tear down.

Used by every safety / role-expression eval. The key constraint per CONVENTIONS
"Serving topology" is *phased*: load model -> classify batch -> tear down.
No always-on servers.

Supports both judge variants:
  - "harm_9cat":    paper Appendix D.2.2 (9 categories), binarized by caller.
  - "role_3label":  paper Appendix A (0-3 score), collapsed by caller.

The actual prompt template lives outside this module (configs/judge_prompt.yaml
or configs/role_expression_prompt.yaml). This module is template-agnostic.
"""

from __future__ import annotations

import gc
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch


@dataclass
class JudgeConfig:
    hf_id: str
    tensor_parallel_size: int = 2
    gpu_memory_utilization: float = 0.85
    trust_remote_code: bool = False
    attention_backend: str | None = None
    max_input_len: int = 4096
    max_output_len: int = 128
    chat_template_kwargs: dict[str, Any] | None = None


_INT_RE = re.compile(r"\b(\d+)\b")


def parse_int_label(text: str, lo: int, hi: int) -> int | None:
    """Extract first integer in [lo, hi] from judge output."""
    if not text:
        return None
    for m in _INT_RE.findall(text):
        try:
            v = int(m)
        except ValueError:
            continue
        if lo <= v <= hi:
            return v
    return None


def parse_named_label(text: str, allowed: list[str]) -> str | None:
    """Extract first allowed category name (case-insensitive) from judge output."""
    if not text:
        return None
    lo = text.lower()
    for cat in allowed:
        if cat.lower() in lo:
            return cat
    return None


def run_judge_batch(
    rows: pd.DataFrame,
    prompt_builder: Callable[[dict[str, Any]], str],
    parser: Callable[[str], Any],
    cfg: JudgeConfig,
    *,
    label_col: str = "judge_label",
    raw_col: str = "judge_raw",
    batch_size: int | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Load the judge model, classify `rows`, return rows with label + raw cols appended.

    Args:
        rows: DataFrame with whatever columns `prompt_builder` consumes.
        prompt_builder: row dict -> formatted chat-template-ready string.
        parser: raw judge text -> label (int / str / None).
        cfg: judge model + vllm knobs.
        label_col / raw_col: output column names.
        batch_size: None = let vLLM chunk internally.

    Model is unloaded before return; no GPU state leaks.
    """
    import os

    if cfg.attention_backend:
        os.environ["VLLM_ATTENTION_BACKEND"] = cfg.attention_backend
    else:
        os.environ.pop("VLLM_ATTENTION_BACKEND", None)

    from vllm import LLM, SamplingParams

    t0 = time.time()
    llm = LLM(
        model=cfg.hf_id,
        tensor_parallel_size=cfg.tensor_parallel_size,
        gpu_memory_utilization=cfg.gpu_memory_utilization,
        trust_remote_code=cfg.trust_remote_code,
        max_model_len=cfg.max_input_len + cfg.max_output_len,
        seed=seed,
    )
    t_load = time.time() - t0

    tokenizer = llm.get_tokenizer()
    ctk = cfg.chat_template_kwargs or {}

    prompts: list[str] = []
    for _, r in rows.iterrows():
        raw = prompt_builder(r.to_dict())
        conv: list[Any] = [{"role": "user", "content": raw}]
        s = tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=True, **ctk)
        assert isinstance(s, str)  # tokenize=False guarantees str return
        prompts.append(s)

    sp = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=cfg.max_output_len, seed=seed)
    t1 = time.time()
    outputs = llm.generate(prompts, sp)
    t_gen = time.time() - t1

    raws = [o.outputs[0].text for o in outputs]
    labels = [parser(txt) for txt in raws]

    out = rows.copy()
    out[raw_col] = raws
    out[label_col] = labels
    out.attrs["judge_hf_id"] = cfg.hf_id
    out.attrs["judge_load_seconds"] = round(t_load, 2)
    out.attrs["judge_gen_seconds"] = round(t_gen, 2)
    out.attrs["judge_tokens_per_sec"] = (
        round(sum(len(o.outputs[0].token_ids) for o in outputs) / t_gen, 2) if t_gen > 0 else None
    )

    del llm
    gc.collect()
    torch.cuda.empty_cache()
    return out


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    """Write DataFrame to parquet, preserving top-level metadata in a sibling .attrs.json."""
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    attrs_path = path.with_suffix(".attrs.json")
    attrs_path.write_text(json.dumps(dict(df.attrs), indent=2))
