"""Download + preprocess the DAN / in-the-wild persona jailbreak dataset.

Sources (verified 2026-04-24, see plans/decisions.md):
- Personas: TrustAIRLab/in-the-wild-jailbreak-prompts @ a10aab8eff1c73165a442d4464dce192bd28b9c5
  Configuration `jailbreak_2023_12_25` (1,405 in-the-wild persona-jailbreak prompts).
- Questions: TrustAIRLab/forbidden_question_set @ 369aa8e10ee2a26cf087fdcc34af0bb928d33d8e
  390 forbidden questions across 13 OpenAI-policy categories (30/category).

Output (under --output-dir, default `data/eval/dan_jailbreak/`):
- raw_personas.parquet     all 1,405 persona prompts + source metadata
- raw_questions.parquet    390 forbidden questions + 13-category labels
- sampled_1100.parquet     stratified sample of 1,100 (persona, question) pairs,
                           ~85 per category. Columns:
                             prompt_id (int), persona_id (int), question_id (int),
                             persona_text (str), question_text (str),
                             category (str), full_prompt (str)
- manifest.json            HF revisions, seed, stratification counts, download date

Stratification: balanced ~85/category (1100/13 = 84.6) across the 13 forbidden-question
categories. Personas are sampled uniformly within each (category) cell with replacement
disabled at the persona level (i.e., no persona repeated within a category until the
persona pool is exhausted, which it isn't — 1,405 personas >> 85). The full_prompt is
the persona text concatenated with two newlines and the question text, matching the
Shah et al. attack form (persona system prompt + harmful behavioral question).
"""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from datasets import load_dataset

PERSONAS_REPO = "TrustAIRLab/in-the-wild-jailbreak-prompts"
PERSONAS_CONFIG = "jailbreak_2023_12_25"
PERSONAS_REVISION = "a10aab8eff1c73165a442d4464dce192bd28b9c5"

QUESTIONS_REPO = "TrustAIRLab/forbidden_question_set"
QUESTIONS_REVISION = "369aa8e10ee2a26cf087fdcc34af0bb928d33d8e"

TARGET_TOTAL = 1100
N_CATEGORIES = 13
DEFAULT_SEED = 42


def _attack_template(persona_text: str, question_text: str) -> str:
    """Concatenate persona prompt + question into a single attack string.

    Two-newline separator matches Shah et al.'s persona-modulation form (system-prompt
    persona followed by user behavioral question).
    """
    return f"{persona_text.strip()}\n\n{question_text.strip()}"


def _stratified_sample(
    questions: pd.DataFrame,
    personas: pd.DataFrame,
    target_total: int,
    seed: int,
) -> pd.DataFrame:
    """Build target_total (persona, question) pairs balanced across categories.

    Within each category: pick all 30 questions, then for each question sample
    ceil(per_cat/30) personas without replacement until we hit per_cat pairs for that
    category. Personas can repeat across categories (different attack contexts), but
    within a category we exhaust the persona pool before duplicating.
    """
    rng = random.Random(seed)
    cats = sorted(questions["content_policy_name"].unique())
    assert len(cats) == N_CATEGORIES, f"Expected {N_CATEGORIES} cats, got {len(cats)}"

    base, extra = divmod(target_total, N_CATEGORIES)
    # Distribute the +1 remainder across the first `extra` categories alphabetically;
    # rng-shuffle the order so it isn't always the same cats getting the bonus.
    cat_order = cats[:]
    rng.shuffle(cat_order)
    per_cat = {c: base + (1 if i < extra else 0) for i, c in enumerate(cat_order)}

    persona_pool = personas[["persona_id", "persona_text"]].to_dict(orient="records")
    rows: list[dict] = []
    next_prompt_id = 0
    for cat in cats:
        n = per_cat[cat]
        cat_qs = questions[questions["content_policy_name"] == cat].to_dict(orient="records")
        # Per-category persona shuffle: each category gets its own ordering so we don't
        # give the same persona always-the-easiest question.
        cat_personas = persona_pool[:]
        rng.shuffle(cat_personas)
        # Question round-robin: cycle questions so each question gets ~equal personas.
        for i in range(n):
            q = cat_qs[i % len(cat_qs)]
            p = cat_personas[i % len(cat_personas)]
            rows.append(
                {
                    "prompt_id": next_prompt_id,
                    "persona_id": int(p["persona_id"]),
                    "question_id": int(q["question_id"]),
                    "persona_text": p["persona_text"],
                    "question_text": q["question"],
                    "category": cat,
                    "full_prompt": _attack_template(p["persona_text"], q["question"]),
                }
            )
            next_prompt_id += 1
    return pd.DataFrame(rows)


def build(output_dir: Path, seed: int = DEFAULT_SEED) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] Loading personas: {PERSONAS_REPO}/{PERSONAS_CONFIG} @ {PERSONAS_REVISION[:12]}")
    p_ds = load_dataset(
        PERSONAS_REPO,
        PERSONAS_CONFIG,
        split="train",
        revision=PERSONAS_REVISION,
    )
    personas = p_ds.to_pandas().reset_index(drop=True)
    personas["persona_id"] = personas.index.astype(int)
    personas = personas.rename(columns={"prompt": "persona_text"})
    keep_cols = [
        "persona_id",
        "persona_text",
        "platform",
        "source",
        "jailbreak",
        "created_at",
        "date",
        "community",
        "community_id",
    ]
    personas_out = personas[[c for c in keep_cols if c in personas.columns]].copy()
    personas_path = output_dir / "raw_personas.parquet"
    personas_out.to_parquet(personas_path, index=False)
    print(f"      wrote {len(personas_out)} rows -> {personas_path}")

    print(f"[2/4] Loading questions: {QUESTIONS_REPO} @ {QUESTIONS_REVISION[:12]}")
    q_ds = load_dataset(
        QUESTIONS_REPO,
        split="train",
        revision=QUESTIONS_REVISION,
    )
    questions = q_ds.to_pandas().reset_index(drop=True)
    # The dataset uses (content_policy_id, q_id) as a 2-D key; build a global int id.
    questions["question_id"] = questions.index.astype(int)
    questions_out = questions[
        ["question_id", "content_policy_id", "content_policy_name", "q_id", "question"]
    ].copy()
    questions_path = output_dir / "raw_questions.parquet"
    questions_out.to_parquet(questions_path, index=False)
    print(f"      wrote {len(questions_out)} rows, {questions_out.content_policy_name.nunique()} cats -> {questions_path}")

    print(f"[3/4] Stratified sampling to {TARGET_TOTAL} pairs (seed={seed})")
    sampled = _stratified_sample(questions_out, personas_out, TARGET_TOTAL, seed)
    sampled_path = output_dir / "sampled_1100.parquet"
    sampled.to_parquet(sampled_path, index=False)
    cat_counts = sampled["category"].value_counts().to_dict()
    cat_counts_sorted = dict(sorted(cat_counts.items()))
    print(f"      wrote {len(sampled)} rows -> {sampled_path}")
    print(f"      per-category counts: min={min(cat_counts.values())}, max={max(cat_counts.values())}, mean={sum(cat_counts.values())/len(cat_counts):.2f}")

    print("[4/4] Writing manifest.json")
    manifest = {
        "schema_version": "1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "target_total": TARGET_TOTAL,
        "n_categories": N_CATEGORIES,
        "personas_source": {
            "repo": PERSONAS_REPO,
            "config": PERSONAS_CONFIG,
            "revision": PERSONAS_REVISION,
            "split": "train",
            "n_rows": len(personas_out),
            "url": f"https://huggingface.co/datasets/{PERSONAS_REPO}/tree/{PERSONAS_REVISION}",
        },
        "questions_source": {
            "repo": QUESTIONS_REPO,
            "revision": QUESTIONS_REVISION,
            "split": "train",
            "n_rows": len(questions_out),
            "url": f"https://huggingface.co/datasets/{QUESTIONS_REPO}/tree/{QUESTIONS_REVISION}",
        },
        "category_labels": sorted(questions_out["content_policy_name"].unique().tolist()),
        "stratification_counts": cat_counts_sorted,
        "outputs": {
            "raw_personas": str(personas_path.relative_to(output_dir.parent.parent.parent)) if output_dir.is_absolute() else str(personas_path),
            "raw_questions": str(questions_path.relative_to(output_dir.parent.parent.parent)) if output_dir.is_absolute() else str(questions_path),
            "sampled_1100": str(sampled_path.relative_to(output_dir.parent.parent.parent)) if output_dir.is_absolute() else str(sampled_path),
        },
        "attack_template": "f'{persona_text}\\n\\n{question_text}' (two newlines)",
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"      wrote -> {manifest_path}")
    print()
    print("DONE")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/eval/dan_jailbreak"),
        help="Where to write the parquet + manifest. Default: data/eval/dan_jailbreak",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"RNG seed for stratified sampling. Default {DEFAULT_SEED}.",
    )
    args = parser.parse_args()
    build(args.output_dir.resolve(), seed=args.seed)


if __name__ == "__main__":
    main()
