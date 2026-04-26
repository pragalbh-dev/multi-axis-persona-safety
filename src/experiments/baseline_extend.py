"""Phase 1 — Baseline expansion orchestrator.

Extends Plan B's 500-prompt DAN baseline to:
  - 604 unused DAN prompts (those not run in Plan B baseline)
  - 1105 Shah-reconstructed prompts (never run on this model)

Produces a parquet with the same schema as Plan B details.parquet (plus a
`dataset` column ∈ {dan, shah}) so it concatenates cleanly for Phase 2 Ext A.

Pipeline (parent process never imports torch/vllm/transformers):
  1. build_prompts (CPU)        → rollouts/_extended_prompts.parquet
  2. vllm_baseline (subproc)    → rollouts/baseline_extended.parquet
  3. hf_extract (subproc)       → cache + projections in memory
  4. judge (subproc)            → adds harm_label_primary
  5. assemble (CPU)             → extensions/baseline_extended.parquet

Usage:
  uv run python -m src.experiments.baseline_extend --config configs/baseline_extend.yaml
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import yaml


def _log(msg: str) -> None:
    print(f"[bl_ext {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _mark_done(marker: Path, payload: dict) -> None:
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps(payload, indent=2, default=str))


def step_1_build_prompts(cfg: dict, out_dir: Path) -> Path:
    """CPU: select unused DAN + all Shah, write a unified prompts parquet."""
    import pandas as pd

    marker = out_dir / ".step1.done"
    prompts_path = out_dir / "rollouts" / "_extended_prompts.parquet"
    if marker.exists():
        _log(f"step 1: skipped (marker {marker.name} exists)")
        return prompts_path

    # Plan B baseline prompts to exclude from DAN.
    plan_b = pd.read_parquet(cfg["plan_b_baseline_parquet"])
    used_ids = set(plan_b[plan_b["condition_id"] == "baseline"]["prompt_id"].astype(int).tolist())
    _log(f"step 1: Plan B used {len(used_ids)} DAN prompts; will skip them")

    dan = pd.read_parquet(cfg["dan_path"]).copy()
    dan["dataset"] = "dan"
    dan_unused = dan[~dan["prompt_id"].astype(int).isin(used_ids)].copy()
    _log(f"step 1: DAN unused = {len(dan_unused)} of {len(dan)}")

    shah = pd.read_parquet(cfg["shah_path"]).copy()
    shah["dataset"] = "shah"
    # Shah's prompt_id collides with DAN (both 0..). Disambiguate via dataset col;
    # downstream uses (dataset, prompt_id) jointly. To keep activation-cache prompt_ids
    # globally unique, rename Shah ids to a `shah_<id>` string and DAN to `dan_<id>`.
    dan_unused["prompt_id"] = dan_unused["prompt_id"].astype(int).map(lambda i: f"dan_{i}")
    shah["prompt_id"] = shah["prompt_id"].astype(int).map(lambda i: f"shah_{i}")

    # Unified core schema (DAN has 7 cols, Shah has 13 — keep core).
    core_cols = ["prompt_id", "persona_id", "question_id", "persona_text", "question_text", "category", "full_prompt", "dataset"]
    combined = pd.concat(
        [dan_unused[core_cols], shah[core_cols]],
        ignore_index=True,
    )
    combined["input_text"] = combined["full_prompt"]
    combined["condition_id"] = "baseline"

    # Drop outliers that exceed vLLM long-profile context (max_model_len=8192).
    # Plan B's 500-prompt sample max was 22.6k chars (~6.5k tokens, fits). The
    # extended set has a few 50k-char DAN personas that overflow. Char threshold
    # 24000 ≈ 7000 tokens with Gemma tokenizer (conservative; max_new_tokens=256
    # leaves ~7800-token budget). Logged + dropped, not silently truncated.
    char_limit = 24000
    too_long = combined["input_text"].str.len() > char_limit
    n_dropped = int(too_long.sum())
    if n_dropped:
        dropped_ids = combined.loc[too_long, ["prompt_id", "dataset"]].to_dict("records")
        _log(
            f"step 1: DROPPING {n_dropped} prompts whose input_text > {char_limit} chars "
            f"(would overflow vLLM long-profile max_model_len=8192). IDs: {dropped_ids}"
        )
        combined = combined[~too_long].reset_index(drop=True)

    prompts_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(prompts_path, index=False)
    _log(
        f"step 1: wrote {len(combined)} prompts "
        f"({(combined.dataset == 'dan').sum()} DAN + {(combined.dataset == 'shah').sum()} Shah) "
        f"→ {prompts_path}"
    )
    _mark_done(marker, {
        "n_total": int(len(combined)),
        "n_dan": int((combined.dataset == "dan").sum()),
        "n_shah": int((combined.dataset == "shah").sum()),
        "n_dropped_long": n_dropped,
        "prompts_path": str(prompts_path),
    })
    return prompts_path


def step_2_vllm_baseline(cfg: dict, out_dir: Path, prompts_path: Path) -> Path:
    """vLLM long-profile generation on the unified prompts."""
    from src.utils.model_runner import run_in_subprocess

    marker = out_dir / ".step2.done"
    rollouts_path = out_dir / "rollouts" / "baseline_extended_rollouts.parquet"
    if marker.exists():
        _log(f"step 2: skipped (marker {marker.name} exists)")
        return rollouts_path

    res = run_in_subprocess(
        "src.evaluation.run_subject_rollouts",
        {
            "model_id": cfg["model_id"],
            "backend": "vllm",
            "profile": cfg["baseline_profile"],
            "prompts_path": str(prompts_path),
            "output_path": str(rollouts_path),
            "condition_id": "baseline",
            "seed": cfg["seed"],
            "max_new_tokens": cfg["max_new_tokens"],
            "temperature": 0.0,
            "steering": None,
        },
        output_path=out_dir / ".step2.work.json",
        timeout_seconds=7200,
    )
    _log(
        f"step 2: vLLM baseline done in {res['elapsed_seconds']}s, "
        f"{res['n_rows']} rows, {res['tokens_per_second']} tok/s"
    )
    _mark_done(marker, res)
    return rollouts_path


def step_3_hf_extract(cfg: dict, out_dir: Path, rollouts_path: Path) -> Path:
    """HF L*=21 per-prompt activation extraction. Writes safetensors cache."""
    import pandas as pd

    from src.utils.model_runner import run_in_subprocess

    marker = out_dir / ".step3.done"
    cache_dir = "data/cache"
    dataset = "baseline_extended_L21"
    if marker.exists():
        _log(f"step 3: skipped (marker {marker.name} exists)")
        return Path(cache_dir) / "activations" / cfg["model_id"] / dataset / "L21"

    df = pd.read_parquet(rollouts_path)
    rows = []
    for _, r in df.iterrows():
        # Use the same prompt_id namespace as activations cache (globally unique
        # across DAN+Shah by step_1's prefix-rename).
        rows.append({
            "prompt_id": str(r["prompt_id"]) + "::baseline",
            "input_text": r["input_text"],
            "response_text": r["response_text"],
        })

    res = run_in_subprocess(
        "src.extraction.run_extract",
        {
            "model_id": cfg["model_id"],
            "rows": rows,
            "layers": [int(cfg["l_star"])],
            "dataset": dataset,
            "token_aggregation": "mean_response",
            "cache_root": cache_dir,
            "seed": cfg["seed"],
            "max_seq_len": int(cfg["hf_extract_max_input_len"]) + 1024,
            "batch_size": int(cfg["hf_extract_batch_size"]),
        },
        output_path=out_dir / ".step3.work.json",
        timeout_seconds=10800,
    )
    _log(f"step 3: HF extracted {res['n_rows']} per-prompt activations in {res['elapsed_seconds']}s")
    _mark_done(marker, res)
    return Path(cache_dir) / "activations" / cfg["model_id"] / dataset / "L21"


def step_4_judge(cfg: dict, out_dir: Path, rollouts_path: Path) -> Path:
    """Qwen 3.6-27B judge on the extended baseline rollouts."""
    from src.utils.model_runner import run_in_subprocess

    marker = out_dir / ".step4.done"
    judged_path = out_dir / "rollouts" / "baseline_extended_judged.parquet"
    if marker.exists():
        _log(f"step 4: skipped (marker {marker.name} exists)")
        return judged_path

    res = run_in_subprocess(
        "src.evaluation.run_judge",
        {
            "judge_model_id": cfg["judge_model_id"],
            "judge_prompt_path": cfg["judge_prompt_path"],
            "rows_path": str(rollouts_path),
            "output_path": str(judged_path),
            "label_col": "harm_label_primary",
            "raw_col": "judge_raw_primary",
            "row_to_slots": {
                "request": "input_text",
                "response": "response_text",
                "behavior": "category",
                "action": "question_text",
            },
            "seed": cfg["seed"],
        },
        output_path=out_dir / ".step4.work.json",
        timeout_seconds=7200,
    )
    _log(f"step 4: judging done in {res['elapsed_seconds']}s, {res['n_parsed']}/{res['n_rows']} parsed")
    _mark_done(marker, res)
    return judged_path


def step_5_assemble(cfg: dict, out_dir: Path, judged_path: Path, cache_stem: Path) -> Path:
    """Project activations onto AA + PCs, add harm_binary, write final parquet."""
    import numpy as np
    import pandas as pd
    from safetensors.torch import load_file

    from src.evaluation.safety import binarize_harm
    from src.extraction.types import ActivationCache

    marker = out_dir / ".step5.done"
    out_path = out_dir / "baseline_extended.parquet"
    if marker.exists():
        _log(f"step 5: skipped (marker {marker.name} exists)")
        return out_path

    df = pd.read_parquet(judged_path)
    cache = ActivationCache.load(cache_stem)
    acts = cache.tensor.float().numpy()

    plan_b_extr = Path(cfg["plan_b_extraction_dir"])
    aa = load_file(str(plan_b_extr / "aa.safetensors"))["aa_at_lstar"].float().numpy()
    aa_unit = aa / max(np.linalg.norm(aa), 1e-9)
    pcs_d = load_file(str(plan_b_extr / "pcs.safetensors"))
    pcs = pcs_d["pcs_at_lstar"].float().numpy()
    pca_mean = pcs_d["pca_mean_at_lstar"].float().numpy()

    aa_proj = acts @ aa_unit
    centered = acts - pca_mean
    pc_projs = centered @ pcs.T

    df = df.copy()
    df["aa_projection"] = aa_proj
    df["pc_projections"] = [json.dumps(row.tolist()) for row in pc_projs]
    df["layer"] = int(cfg["l_star"])
    df["harm_binary"] = df["harm_label_primary"].map(binarize_harm)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)

    # Quick sanity: per-dataset harm rate
    per_ds = df.groupby("dataset").agg(
        n=("harm_binary", "size"),
        harm=("harm_binary", "mean"),
    ).round(3)
    _log("step 5: assembled extended details")
    _log(f"  per-dataset harm rate:\n{per_ds.to_string()}")
    _log(f"  → {out_path}")
    _mark_done(marker, {
        "n_total": int(len(df)),
        "n_harm": int(df["harm_binary"].sum()),
        "per_dataset_harm": per_ds.to_dict(),
    })
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("configs/baseline_extend.yaml"))
    ap.add_argument("--skip-step", type=int, default=None, help="skip step N and below")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    _log(f"config: {args.config}")
    _log(f"output_dir: {out_dir}")

    skip = args.skip_step or 0

    if skip < 1:
        prompts_path = step_1_build_prompts(cfg, out_dir)
    else:
        prompts_path = out_dir / "rollouts" / "_extended_prompts.parquet"

    if skip < 2:
        rollouts_path = step_2_vllm_baseline(cfg, out_dir, prompts_path)
    else:
        rollouts_path = out_dir / "rollouts" / "baseline_extended_rollouts.parquet"

    if skip < 3:
        cache_stem = step_3_hf_extract(cfg, out_dir, rollouts_path)
    else:
        cache_stem = Path("data/cache") / "activations" / cfg["model_id"] / "baseline_extended_L21" / "L21"

    if skip < 4:
        judged_path = step_4_judge(cfg, out_dir, rollouts_path)
    else:
        judged_path = out_dir / "rollouts" / "baseline_extended_judged.parquet"

    if skip < 5:
        out_path = step_5_assemble(cfg, out_dir, judged_path, cache_stem)
    else:
        out_path = out_dir / "baseline_extended.parquet"

    _log(f"DONE → {out_path}")


if __name__ == "__main__":
    main()
