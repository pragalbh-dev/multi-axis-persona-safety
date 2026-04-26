"""Subprocess work-module: GPT-5.5 (OpenAI API) judge labelling.

Async client with concurrency=100 default (configurable to 200 via args).
Used for T2.4.5 — labelling 200 (prompt, response) pairs as pseudo-ground-truth
to validate our self-hosted Qwen 3.6-27B primary judge.

OPENAI_API_KEY loaded from .env via src.utils.env import.

Args JSON schema:
{
  "rows_path":          str,        # parquet of (prompt_id, dataset, condition_id, ...)
  "judge_prompt_path":  str,        # configs/judge_prompt.yaml
  "row_to_slots":       dict,       # {slot_name: column_name}
  "model":              str,        # "gpt-5.5" (or override for testing)
  "max_concurrent":     int,        # default 100
  "budget_usd":         float,      # default 15.0; hard-stop on exceed
  "input_price_per_mtok":  float,   # for cost tracking; user provides current pricing
  "output_price_per_mtok": float,
  "output_path":        str,        # output parquet
  "seed":               int
}

Output JSON:
{
  "status": "ok"|"budget_exceeded",
  "elapsed_seconds": float,
  "n_completed": int,
  "n_total": int,
  "cost_usd": float,
  "artifacts": [output_path]
}
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import time
from pathlib import Path

from src.utils import env as _env  # noqa: F401  loads .env (OPENAI_API_KEY)

_JSON_SCORE_RE = re.compile(r'"score"\s*:\s*"([a-z_]+)"', re.IGNORECASE)


def _parse_json_score(text: str, allowed: list[str]) -> str | None:
    if not text:
        return None
    m = _JSON_SCORE_RE.search(text)
    if m:
        cand = m.group(1).lower()
        for cat in allowed:
            if cat.lower() == cand:
                return cat
    lo = text.lower()
    for cat in allowed:
        if cat.lower() in lo:
            return cat
    return None


async def _label_one(
    client,
    sem: asyncio.Semaphore,
    *,
    model: str,
    system_prompt: str,
    user_text: str,
    seed: int,
    json_schema: dict | None,
    max_output_tokens: int,
    is_reasoning_model: bool,
) -> tuple[str | None, str, dict]:
    """One async judge call. Returns (raw_text, raw_text, usage_dict)."""
    from openai import APIError, RateLimitError

    backoff = 0.5
    for attempt in range(6):
        try:
            async with sem:
                kwargs: dict = {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_text},
                    ],
                    "seed": seed,
                }
                # Reasoning models (gpt-5.x, o1, o3): use max_completion_tokens, no temperature.
                # Non-reasoning models: use max_tokens, temperature=0.
                if is_reasoning_model:
                    kwargs["max_completion_tokens"] = max_output_tokens
                else:
                    kwargs["max_tokens"] = max_output_tokens
                    kwargs["temperature"] = 0
                if json_schema is not None:
                    kwargs["response_format"] = {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "harm_eval",
                            "schema": json_schema,
                            "strict": True,
                        },
                    }
                resp = await client.chat.completions.create(**kwargs)
                msg = resp.choices[0].message.content or ""
                usage = {
                    "input_tokens": resp.usage.prompt_tokens,
                    "output_tokens": resp.usage.completion_tokens,
                }
                return msg, msg, usage
        except RateLimitError:
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 16.0)
        except APIError:
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 16.0)
    return None, "", {"input_tokens": 0, "output_tokens": 0}


async def _run(args: dict) -> dict:
    import pandas as pd
    import yaml
    from openai import AsyncOpenAI

    prompt_yaml = yaml.safe_load(Path(args["judge_prompt_path"]).read_text())
    system_prompt = prompt_yaml["system_prompt"]
    user_template = prompt_yaml["user_template"]
    categories = prompt_yaml.get("categories", [])
    response_max_chars = prompt_yaml.get("response_max_tokens", 512) * 4  # rough char/token

    rows_df = pd.read_parquet(args["rows_path"])
    slot_map = args["row_to_slots"]

    # JSON schema for response_format strict mode (OpenAI requires `additionalProperties: false`).
    json_schema = {
        "type": "object",
        "properties": {
            "analysis": {"type": "string"},
            "score": {"type": "string", "enum": categories},
        },
        "required": ["analysis", "score"],
        "additionalProperties": False,
    }

    client = AsyncOpenAI()
    max_concurrent = int(args.get("max_concurrent", 100))
    sem = asyncio.Semaphore(max_concurrent)
    seed = args.get("seed", 42)
    model = args.get("model", "gpt-5.5")
    # Reasoning models (gpt-5.x, o1, o3) use max_completion_tokens and burn
    # tokens on hidden chain-of-thought. Non-reasoning (gpt-4.x) uses max_tokens
    # and temperature=0.
    is_reasoning = any(model.startswith(p) for p in ("gpt-5", "o1", "o3"))
    # Generous output budget — judge analysis is ~6 sentences + JSON wrapping;
    # reasoning models burn extra on hidden CoT. Caller can override via args.
    max_output_tokens = int(args.get("max_output_tokens", 2048 if is_reasoning else 512))

    in_price = float(args.get("input_price_per_mtok", 0.0))
    out_price = float(args.get("output_price_per_mtok", 0.0))
    budget_usd = float(args.get("budget_usd", 15.0))

    user_texts: list[str] = []
    for _, r in rows_df.iterrows():
        slot_values: dict[str, str] = {}
        for slot_name, col_name in slot_map.items():
            val = r[col_name] if col_name in r else ""
            sval = str(val) if val is not None else ""
            if slot_name == "response":
                # Coarse char-level truncation to match the prompt's "first N tokens" guidance.
                sval = sval[:response_max_chars]
            slot_values[slot_name] = sval
        user_texts.append(user_template.format(**slot_values))

    # Schedule.
    tasks = [
        asyncio.create_task(
            _label_one(
                client,
                sem,
                model=model,
                system_prompt=system_prompt,
                user_text=user_text,
                seed=seed,
                json_schema=json_schema,
                max_output_tokens=max_output_tokens,
                is_reasoning_model=is_reasoning,
            )
        )
        for user_text in user_texts
    ]

    results: list[tuple[str | None, str, dict]] = [None] * len(tasks)  # type: ignore[list-item]
    completed = 0
    cost_usd = 0.0
    budget_exceeded = False

    out_path = Path(args["output_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for fut in asyncio.as_completed(tasks):
        msg, raw, usage = await fut
        idx = completed
        completed += 1
        results[idx] = (msg, raw, usage)
        cost_usd += (
            usage["input_tokens"] * in_price / 1_000_000
            + usage["output_tokens"] * out_price / 1_000_000
        )
        if cost_usd >= budget_usd:
            budget_exceeded = True
            for t in tasks:
                t.cancel()
            break

        # Stash partial every 25
        if completed % 25 == 0:
            partial = rows_df.iloc[:completed].copy()
            partial["gpt55_raw_output"] = [r[1] for r in results[:completed]]
            partial["gpt55_label"] = [
                _parse_json_score(r[1], categories) for r in results[:completed]
            ]
            partial.to_parquet(out_path, index=False)

    # Fill cancelled remainders with None
    for i in range(len(results)):
        if results[i] is None:
            results[i] = (None, "", {"input_tokens": 0, "output_tokens": 0})

    rows_out = rows_df.copy()
    rows_out["gpt55_raw_output"] = [r[1] for r in results]
    rows_out["gpt55_label"] = [_parse_json_score(r[1], categories) for r in results]
    rows_out["gpt55_input_tokens"] = [r[2]["input_tokens"] for r in results]
    rows_out["gpt55_output_tokens"] = [r[2]["output_tokens"] for r in results]
    rows_out.to_parquet(out_path, index=False)

    n_parsed = sum(1 for r in results if r[0] is not None)
    return {
        "completed": completed,
        "n_parsed": n_parsed,
        "cost_usd": round(cost_usd, 4),
        "budget_exceeded": budget_exceeded,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--args-json", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    cli = parser.parse_args()

    args = json.loads(cli.args_json.read_text())
    t0 = time.time()

    result = asyncio.run(_run(args))
    elapsed = time.time() - t0

    from src.utils.model_runner import write_result

    write_result(
        cli.output,
        {
            "status": "budget_exceeded" if result["budget_exceeded"] else "ok",
            "elapsed_seconds": round(elapsed, 2),
            "peak_vram_per_gpu": [],  # API call, no local GPU
            "artifacts": [args["output_path"]],
            "n_completed": result["completed"],
            "n_parsed": result["n_parsed"],
            "cost_usd": result["cost_usd"],
        },
    )


if __name__ == "__main__":
    main()
