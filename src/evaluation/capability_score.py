"""Per-benchmark scorers for Phase E capability eval.

Inputs are post-rollout DataFrames (one row per prompt) with columns:
  - prompt_id, dataset, response_text  (always)
  - bench-specific gold columns (instruction_id_list/kwargs for IFEval,
    answer for GSM8k, reference_answer_fullscale for EQ-Bench).

Each scorer returns:
  {
    "n": int,
    "score": float,         # headline metric, normalized to [0, 1]
    "extra": {...},         # bench-specific breakdown
  }
"""

from __future__ import annotations

import re
from typing import Any

import pandas as pd

# ── IFEval ─────────────────────────────────────────────────────────────────────


def score_ifeval(df: pd.DataFrame) -> dict[str, Any]:
    """Run josejg/instruction_following_eval scorer.

    Returns prompt-level + instruction-level strict/loose accuracy. Headline
    metric = mean of (inst_strict, inst_loose) — matches paper convention.

    Reads `instruction_id_list_json` + `kwargs_json` (JSON-stringified to
    survive parquet round-trip without pandas-induced schema unioning).
    """
    import json as _json

    from instruction_following_eval import evaluate_instruction_following

    examples: list[dict] = []
    responses: list[str] = []
    for i, (_, r) in enumerate(df.iterrows()):
        instruction_id_list = _json.loads(r["instruction_id_list_json"])
        kwargs = _json.loads(r["kwargs_json"])
        ex = {
            "key": i,
            "prompt": r.get("input_text", ""),
            "instruction_id_list": instruction_id_list,
            "kwargs": kwargs,
        }
        examples.append(ex)
        responses.append(str(r["response_text"]))
    metrics = evaluate_instruction_following(examples, responses)
    headline = 0.5 * (metrics["inst_level_strict_accuracy"] + metrics["inst_level_loose_accuracy"])
    return {
        "n": int(len(df)),
        "score": float(headline),
        "extra": {
            "prompt_level_strict_accuracy": float(metrics["prompt_level_strict_accuracy"]),
            "prompt_level_loose_accuracy": float(metrics["prompt_level_loose_accuracy"]),
            "inst_level_strict_accuracy": float(metrics["inst_level_strict_accuracy"]),
            "inst_level_loose_accuracy": float(metrics["inst_level_loose_accuracy"]),
        },
    }


# ── GSM8k ──────────────────────────────────────────────────────────────────────

_GSM8K_FINAL = re.compile(r"####\s*(-?[\d,]+(?:\.\d+)?)")
_LAST_NUM_RE = re.compile(r"(-?\d[\d,]*(?:\.\d+)?)")


def _extract_gsm8k_gold(answer: str) -> str | None:
    m = _GSM8K_FINAL.search(answer)
    if not m:
        return None
    return m.group(1).replace(",", "").rstrip(".")


def _extract_gsm8k_pred(response: str) -> str | None:
    """Pull the most likely final numeric answer from a model response.

    Heuristic order:
      1. If response contains '#### N' → use it (model emitted the GSM8k marker).
      2. Else: look for '\\boxed{...}' style.
      3. Else: take the LAST number in the response (most generations end with the
         answer).
    Returns the numeric string with commas stripped, or None.
    """
    m = _GSM8K_FINAL.search(response)
    if m:
        return m.group(1).replace(",", "").rstrip(".")
    m_box = re.search(r"\\boxed\{\s*(-?\d[\d,]*(?:\.\d+)?)\s*\}", response)
    if m_box:
        return m_box.group(1).replace(",", "").rstrip(".")
    nums = _LAST_NUM_RE.findall(response)
    if nums:
        return nums[-1].replace(",", "").rstrip(".")
    return None


def score_gsm8k(df: pd.DataFrame) -> dict[str, Any]:
    """Strict numeric exact-match. Both gold and prediction normalized to
    comma-stripped strings; trailing decimal zeros tolerated via float compare
    when both parse cleanly."""
    correct = 0
    n = 0
    parse_fail = 0
    for _, r in df.iterrows():
        n += 1
        gold = _extract_gsm8k_gold(str(r["answer"]))
        pred = _extract_gsm8k_pred(str(r["response_text"]))
        if gold is None:
            parse_fail += 1
            continue
        if pred is None:
            continue
        if pred == gold:
            correct += 1
            continue
        try:
            if abs(float(pred) - float(gold)) < 1e-6:
                correct += 1
        except ValueError:
            pass
    return {
        "n": int(n),
        "score": float(correct / n) if n else 0.0,
        "extra": {
            "n_correct": int(correct),
            "gold_parse_failures": int(parse_fail),
        },
    }


# ── EQ-Bench (v2 fullscale rubric) ─────────────────────────────────────────────

_EMOTION_LINE = re.compile(r"^([A-Za-z][A-Za-z\- ]+?)\s*[:=]\s*(\d+(?:\.\d+)?)\s*$")


def _parse_emotion_scores(response: str) -> dict[str, float]:
    """Parse 'Emotion: N' lines from a model response.

    EQ-Bench format: 4 lines, one per candidate emotion, score 0–10.
    """
    out: dict[str, float] = {}
    for line in response.splitlines():
        line = line.strip()
        if not line:
            continue
        m = _EMOTION_LINE.match(line)
        if not m:
            continue
        emo = m.group(1).strip().lower()
        try:
            score = float(m.group(2))
        except ValueError:
            continue
        out[emo] = score
    return out


def _parse_reference(ref_str: str) -> dict[str, float]:
    """Reference is a stringified Python dict like
    '{"emotion1": "Remorseful", "emotion1_score": 0, ...}'.

    Build {emotion_name_lower: score_float}."""
    import ast

    try:
        d = ast.literal_eval(ref_str)
    except (ValueError, SyntaxError):
        return {}
    out: dict[str, float] = {}
    for i in (1, 2, 3, 4):
        name_key = f"emotion{i}"
        score_key = f"emotion{i}_score"
        if name_key in d and score_key in d:
            try:
                out[str(d[name_key]).strip().lower()] = float(d[score_key])
            except (ValueError, TypeError):
                continue
    return out


def score_eq_bench(df: pd.DataFrame) -> dict[str, Any]:
    """EQ-Bench v2 fullscale: per-prompt parsed-MAE-based score.

    Per-question score = max(0, 1 − sum |pred − ref| / 40), where 40 = 4
    emotions × max 10. If fewer than 4 emotions parsed, score = 0 (penalize
    format failure, matches the upstream rubric's strictness).

    Headline = mean per-question score across the dataset.
    """
    per_q_scores: list[float] = []
    parse_full = 0
    parse_partial = 0
    parse_zero = 0
    for _, r in df.iterrows():
        ref = _parse_reference(str(r["reference_answer_fullscale"]))
        if not ref:
            per_q_scores.append(0.0)
            parse_zero += 1
            continue
        pred = _parse_emotion_scores(str(r["response_text"]))
        matched = 0
        sum_abs_diff = 0.0
        for emo, ref_score in ref.items():
            if emo in pred:
                matched += 1
                sum_abs_diff += abs(pred[emo] - ref_score)
            else:
                sum_abs_diff += 10.0  # missing emotion = max possible diff
        if matched == len(ref):
            parse_full += 1
        elif matched > 0:
            parse_partial += 1
        else:
            parse_zero += 1
        denom = max(1, len(ref)) * 10.0
        s = max(0.0, 1.0 - sum_abs_diff / denom)
        per_q_scores.append(s)
    n = len(per_q_scores)
    headline = float(sum(per_q_scores) / n) if n else 0.0
    return {
        "n": int(n),
        "score": headline,
        "extra": {
            "n_parsed_full_4_of_4": int(parse_full),
            "n_parsed_partial": int(parse_partial),
            "n_parsed_zero": int(parse_zero),
        },
    }


# ── Dispatch ────────────────────────────────────────────────────────────────────


def score_bench(bench: str, df: pd.DataFrame) -> dict[str, Any]:
    if bench == "ifeval":
        return score_ifeval(df)
    if bench == "gsm8k":
        return score_gsm8k(df)
    if bench == "eq_bench":
        return score_eq_bench(df)
    raise ValueError(f"unknown bench: {bench!r}")
