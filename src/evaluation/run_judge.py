"""Subprocess work-module: judge phase.

Loads a judge model (Qwen 3.6-27B by default) via vLLM with the inference_runtime.yaml
judge profile, classifies a parquet of (request, response) rows using a configurable
judge prompt template, appends labels + raw outputs, exits.

Used by:
  - safety eval (configs/judge_prompt.yaml, 9-cat harm)
  - role-expression eval (configs/role_expression_prompt.yaml, 0-3 score)
  - EQ-Bench rubric scoring (post-Plan B; configs/eq_bench_rubric_prompt.yaml)

Args JSON schema:
{
  "judge_model_id":     str,        # key in inference_runtime.yaml (e.g. "qwen_3_6_27b")
  "judge_prompt_path":  str,        # path to YAML with system_prompt + user_template
  "rows_path":          str,        # input parquet (must include slot columns: request, response, behavior, action)
  "output_path":        str,        # output parquet with judge_label + judge_raw appended
  "label_col":          str,        # default "harm_label_primary"
  "raw_col":            str,        # default "judge_raw_primary"
  "row_to_slots":       dict,       # mapping of {slot_name: row_column_name}
                                    # e.g. {"request": "input_text", "response": "response_text",
                                    #       "behavior": "category", "action": "question_text"}
  "seed":               int
}

Output JSON:
{
  "status": "ok",
  "elapsed_seconds": float,
  "peak_vram_per_gpu": [4 floats],
  "artifacts": [output_path],
  "n_rows": int,
  "n_parsed": int,
  "tokens_per_second": float
}
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

from src.utils import env as _env  # noqa: F401

_JSON_SCORE_RE = re.compile(r'"score"\s*:\s*"([a-z_]+)"', re.IGNORECASE)


def _parse_json_score(text: str, allowed: list[str]) -> str | None:
    """Primary parser: scan for the JSON `score` field; fallback to named scan."""
    m = _JSON_SCORE_RE.search(text)
    if m:
        cand = m.group(1).lower()
        for cat in allowed:
            if cat.lower() == cand:
                return cat
    # Fallback: any allowed category appearing in the raw text.
    lo = text.lower()
    for cat in allowed:
        if cat.lower() in lo:
            return cat
    return None


def _parse_int_score(text: str, lo: int, hi: int) -> int | None:
    """For 0-3 role-expression style prompts."""
    if not text:
        return None
    for tok in re.findall(r"\b(\d+)\b", text):
        try:
            v = int(tok)
        except ValueError:
            continue
        if lo <= v <= hi:
            return v
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--args-json", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    cli = parser.parse_args()

    args = json.loads(cli.args_json.read_text())
    t0 = time.time()

    import pandas as pd
    import yaml
    from vllm import LLM, SamplingParams

    rt = yaml.safe_load(Path("configs/inference_runtime.yaml").read_text())
    judge_cfg = rt[args["judge_model_id"]]
    profile = judge_cfg["profiles"]["judge"]
    chat_template_kwargs = judge_cfg.get("chat_template_kwargs") or {}

    prompt_yaml = yaml.safe_load(Path(args["judge_prompt_path"]).read_text())
    system_prompt = prompt_yaml["system_prompt"]
    user_template = prompt_yaml["user_template"]
    categories = prompt_yaml.get("categories", [])
    parser_kind = prompt_yaml.get("parser", {}).get("primary", "json_score_field")
    score_scale = prompt_yaml.get("score_scale")

    rows_df = pd.read_parquet(args["rows_path"])
    slot_map = args["row_to_slots"]

    # Build prompts.
    llm = LLM(
        model=judge_cfg["hf_id"],
        tensor_parallel_size=judge_cfg["tensor_parallel_size"],
        gpu_memory_utilization=profile["gpu_memory_utilization"],
        max_model_len=profile["max_model_len"],
        max_num_seqs=profile["max_num_seqs"],
        enable_prefix_caching=profile.get("enable_prefix_caching", True),
        trust_remote_code=judge_cfg.get("trust_remote_code", False),
        enforce_eager=False,
        seed=args["seed"],
    )
    tok = llm.get_tokenizer()

    # Truncate response to first N tokens per the prompt yaml (paper line 2577 = 512).
    response_max_tokens = prompt_yaml.get("response_max_tokens")
    # Compute fixed overhead once (system + template + chat wrapping + generation prompt + safety).
    # Then derive a budget for the {request} slot so the total stays under max_model_len - max_output_tokens.
    max_output_tokens = prompt_yaml.get("max_output_len", 256)
    safety_margin = 64

    # Probe overhead with empty slots
    empty_slots = {k: "" for k in slot_map}
    try:
        empty_user = user_template.format(**empty_slots)
    except KeyError:
        empty_user = user_template  # template has slots we didn't map; fall back
    overhead_msgs = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": empty_user},
    ]
    overhead_text = tok.apply_chat_template(
        overhead_msgs, tokenize=False, add_generation_prompt=True, **chat_template_kwargs
    )
    overhead_tokens = len(tok(overhead_text, add_special_tokens=False)["input_ids"])
    request_budget = profile["max_model_len"] - max_output_tokens - safety_margin - overhead_tokens
    # Subtract the response budget too (it goes into {response} slot)
    if response_max_tokens:
        request_budget -= response_max_tokens
    if request_budget < 256:
        # If overhead is somehow huge, fall back to a sane minimum
        request_budget = 256
    print(
        f"[run_judge] overhead={overhead_tokens} tok, response_max={response_max_tokens}, "
        f"max_output={max_output_tokens} → request budget={request_budget} tok",
        flush=True,
    )

    chat_prompts = []
    n_truncated_request = 0
    for _, r in rows_df.iterrows():
        slot_values: dict[str, str] = {}
        for slot_name, col_name in slot_map.items():
            val = r[col_name] if col_name in r else ""
            sval = str(val) if val is not None else ""
            if slot_name == "response" and response_max_tokens is not None:
                ids = tok.encode(sval, add_special_tokens=False)
                if len(ids) > response_max_tokens:
                    sval = tok.decode(ids[:response_max_tokens], skip_special_tokens=True)
            elif slot_name == "request":
                # Left-truncate the persona-jailbreak preamble; keep the harmful question
                # at the END (which is where the actual probe lives).
                ids = tok.encode(sval, add_special_tokens=False)
                if len(ids) > request_budget:
                    sval = tok.decode(ids[-request_budget:], skip_special_tokens=True)
                    n_truncated_request += 1
            slot_values[slot_name] = sval
        try:
            user_text = user_template.format(**slot_values)
        except KeyError as e:
            raise KeyError(
                f"user_template references slot {e} but row_to_slots only provides {list(slot_map)}"
            )

        msgs = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ]
        s = tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, **chat_template_kwargs
        )
        assert isinstance(s, str)
        chat_prompts.append(s)
    print(f"[run_judge] {n_truncated_request}/{len(rows_df)} requests left-truncated to fit", flush=True)

    sp = SamplingParams(
        temperature=prompt_yaml.get("temperature", 0.0),
        top_p=1.0,
        max_tokens=prompt_yaml.get("max_output_len", 256),
        seed=args["seed"],
    )
    t_gen = time.time()
    outputs = llm.generate(chat_prompts, sp)
    gen_seconds = time.time() - t_gen

    raws = [o.outputs[0].text for o in outputs]
    if parser_kind == "json_score_field":
        labels: list = [_parse_json_score(t, categories) for t in raws]
    elif parser_kind == "int_score":
        lo, hi = (min(score_scale), max(score_scale)) if score_scale else (0, 3)
        labels = [_parse_int_score(t, lo, hi) for t in raws]
    else:
        raise ValueError(f"Unknown parser kind: {parser_kind}")

    out = rows_df.copy()
    out[args.get("label_col", "harm_label_primary")] = labels
    out[args.get("raw_col", "judge_raw_primary")] = raws

    out_path = Path(args["output_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)

    n_parsed = sum(1 for label in labels if label is not None)
    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    tps = total_tokens / gen_seconds if gen_seconds > 0 else 0.0

    elapsed = time.time() - t0
    from src.utils.model_runner import peak_vram_per_gpu_gib, write_result

    write_result(
        cli.output,
        {
            "status": "ok",
            "elapsed_seconds": round(elapsed, 2),
            "peak_vram_per_gpu": peak_vram_per_gpu_gib(),
            "artifacts": [args["output_path"]],
            "n_rows": len(out),
            "n_parsed": n_parsed,
            "tokens_per_second": round(tps, 2),
        },
    )


if __name__ == "__main__":
    main()
