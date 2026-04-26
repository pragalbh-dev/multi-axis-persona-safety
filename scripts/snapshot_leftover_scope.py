"""Write a snapshot of Plan B's IN-scope identities + everything cut.

Produces under `results/plan_b_gemma2_27b/leftovers/`:

  - `dan_in_500.parquet`         IN-scope: 500 prompt_ids the full Plan B is using
  - `dan_complement_600.parquet` CUT (1100 - 500): the unselected DAN prompts
  - `roles_in_275.parquet`       IN-scope: all 275 paper roles (Plan B uses every role,
                                 just at 30 rollouts/role instead of paper's 300; so the
                                 leftover is "70 more rollouts/role" not "different roles")
  - `scope.yaml`                 declarative: subjects/datasets/conditions/rollouts cut

Design rules:
  - DOES NOT touch the running Plan B pipeline. Reads only from data/eval/, paper artifacts,
    and configs/. Writes only to results/plan_b_gemma2_27b/leftovers/.
  - Uses the same seed (42) as the Plan B sampler so the IN/CUT split is deterministic
    and matches what the live Plan B run is actually generating against — verified by
    re-running the same stratified sampler offline.
  - Idempotent: rerunning overwrites the same files.

Usage:
  uv run python -m scripts.snapshot_leftover_scope
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import yaml

# Make src/ importable
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))


def replicate_plan_b_dan_sample(n_target: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the SAME stratified sampler the Plan B step_5 uses; return (in_500, complement_600)."""
    dan = pd.read_parquet("data/eval/dan_jailbreak/sampled_1100.parquet")
    per_cat = max(1, n_target // dan["category"].nunique())
    sampled = (
        dan.groupby("category", group_keys=False)
        .apply(lambda g: g.sample(n=min(per_cat, len(g)), random_state=seed))
        .reset_index(drop=True)
    )
    if len(sampled) < n_target:
        rest = dan.drop(sampled.index, errors="ignore").sample(
            n=n_target - len(sampled), random_state=seed
        )
        sampled = pd.concat([sampled, rest]).reset_index(drop=True)
    in_scope = sampled.head(n_target).copy()
    in_scope_ids = set(in_scope["prompt_id"].tolist())
    complement = dan[~dan["prompt_id"].isin(in_scope_ids)].copy().reset_index(drop=True)
    return in_scope, complement


def main() -> None:
    cfg = yaml.safe_load(Path("configs/plan_b.yaml").read_text())
    out_dir = Path(cfg["output_dir"]) / "leftovers"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. DAN IN/CUT split
    in_500, complement_600 = replicate_plan_b_dan_sample(
        n_target=cfg["n_dan_prompts"],
        seed=cfg["seed"],
    )
    in_path = out_dir / f"dan_in_{len(in_500)}.parquet"
    cut_path = out_dir / f"dan_complement_{len(complement_600)}.parquet"
    in_500.to_parquet(in_path, index=False)
    complement_600.to_parquet(cut_path, index=False)

    # Per-category breakdown for sanity
    in_cat = in_500["category"].value_counts().to_dict()
    cut_cat = complement_600["category"].value_counts().to_dict()

    # 2. Roles in scope
    role_dir = Path("data/paper_artifacts/assistant_axis_vectors/gemma-2-27b/role_vectors")
    role_names = sorted(p.stem for p in role_dir.iterdir() if p.suffix == ".pt")
    pd.DataFrame({"role_name": role_names}).to_parquet(out_dir / "roles_in_275.parquet", index=False)

    # 3. Declarative scope file enumerating everything cut
    scope = {
        "in_scope": {
            "subjects": ["gemma_2_27b"],
            "datasets": ["dan"],
            "n_dan_prompts": int(cfg["n_dan_prompts"]),
            "n_role_rollouts_per_role": int(cfg["n_role_rollouts_per_role"]),
            "n_default_assistant_rollouts": int(cfg["n_default_assistant_rollouts"]),
            "n_random_baselines": int(cfg["n_random_baselines"]),
            "pc_indices_to_steer": list(cfg["pc_indices_to_steer"]),
            "steering_lambdas": list(cfg["steering_lambdas"]),
            "n_conditions": (
                1  # baseline
                + 1  # aa_capped
                + len(cfg["pc_indices_to_steer"]) * len(cfg["steering_lambdas"])
                + int(cfg["n_random_baselines"])
            ),
            "n_roles": len(role_names),
            "dan_in_path": str(in_path.as_posix()),
            "roles_path": str((out_dir / "roles_in_275.parquet").as_posix()),
        },
        "cut_scope": {
            "subjects": [
                {"key": "qwen_3_32b", "reason": "Plan B time budget; full sweep replays post-deadline"},
                {"key": "gemma_4_31b_thinking_on", "reason": "Plan B time budget"},
                {"key": "gemma_4_31b_thinking_off", "reason": "Plan B time budget"},
                {"key": "llama_3_3_70b", "reason": "70B at bf16 ≈ 140 GB > 128 GB total VRAM; Stage 7 Ext 9 fp8 only"},
            ],
            "datasets": [
                {
                    "key": "shah_reconstructed",
                    "path": "data/eval/reconstructed_jailbreak/sampled_1100.parquet",
                    "reason": "DAN-only for Plan B; Shah-reconstructed is post-deadline replication check",
                },
            ],
            "dan_complement": {
                "n": int(len(complement_600)),
                "path": str(cut_path.as_posix()),
                "reason": f"500 of 1100 stratified for Plan B time budget; remaining {len(complement_600)} are leftover",
            },
            "rollouts_per_role_remaining": {
                "n_per_role_used": int(cfg["n_role_rollouts_per_role"]),
                "paper_n_per_role": 300,
                "leftover_n_per_role": 300 - int(cfg["n_role_rollouts_per_role"]),
                "reason": "Paper used 300/role for τ-calibration; Plan B used 30 for time. PC fit reused paper's pre-computed cache. Leftover would tighten τ percentiles.",
            },
            "stage_2_tasks_deferred": [
                {"task": "T2.5", "reason": "capability eval (4 benchmarks); post-deadline T3.5.5"},
                {"task": "T2.7b", "reason": "ordinal LASSO + per-PC FDR; post-deadline secondary analysis"},
                {"task": "T2.4 cross-check judge", "reason": "Gemma 4 31B-it as 2nd judge on 200-sample subset; deferred"},
            ],
        },
        "in_scope_per_category": {
            "dan_in_500": {str(k): int(v) for k, v in in_cat.items()},
        },
        "cut_per_category": {
            "dan_complement_600": {str(k): int(v) for k, v in cut_cat.items()},
        },
        "follow_up_runs_if_time": [
            {
                "name": "phase_b_dan_complement_gemma2",
                "scope": "remaining 600 DAN prompts × same 11 conditions on Gemma 2 27B",
                "estimated_compute_hours": 6,
                "reuses": "all of Plan B's caches (AA, PCs, lmsys norm, τ-cal, capping range, judge config)",
            },
            {
                "name": "phase_b_qwen_3_32b",
                "scope": "Plan B's 500 DAN × 11 conditions on Qwen 3 32B",
                "estimated_compute_hours": 7,
                "reuses": "judge runtime config; needs fresh extraction + PC fit on Qwen",
            },
            {
                "name": "phase_b_gemma_4_31b_thinking_off",
                "scope": "Plan B's 500 DAN × 11 conditions on Gemma 4 31B (thinking off)",
                "estimated_compute_hours": 7,
                "reuses": "judge runtime config; needs fresh extraction + PC fit on Gemma 4",
            },
            {
                "name": "phase_b_shah_reconstructed_gemma2",
                "scope": "Shah-reconstructed 500 stratified × 11 conditions on Gemma 2 27B",
                "estimated_compute_hours": 6,
                "reuses": "all of Plan B's caches; new dataset only",
            },
            {
                "name": "phase_b_fresh_pc_fit_gemma2",
                "scope": "Generate 300 rollouts/role on all 275 roles (paper's volume), fit PCA on our own activations, verify L* matches paper's L*=21 within ±3 layers",
                "estimated_compute_hours": 4,
                "reuses": "judge config; nothing else (this is the validation pipeline-vs-paper check)",
                "why": "Plan B reused paper's 275-role cache for the PC fit because 30 rollouts/role is too noisy. Fresh fit on 300/role validates our extraction pipeline produces paper-equivalent PCs.",
            },
            {
                "name": "phase_b_lmsys_norm_authentic",
                "scope": "Request lmsys-chat-1m access, redo step 1c with real lmsys prompts (vs current fallback to extraction questions)",
                "estimated_compute_hours": 0.1,
                "reuses": "everything; just replaces step 1c output",
                "why": "Paper-strict steering-vector norm convention. Current fallback uses paper's 240 extraction questions; faithful magnitude but different prompt distribution. ~5 min once lmsys access granted.",
            },
        ],
    }
    scope_path = out_dir / "scope.yaml"
    scope_path.write_text(yaml.safe_dump(scope, default_flow_style=False, sort_keys=False))

    # 4. Pretty summary to stdout
    print(f"Snapshot written to {out_dir}/")
    print(f"  IN  scope: {len(in_500):4d} DAN prompts, {len(role_names)} roles, 1 subject (gemma_2_27b)")
    print(f"  CUT scope: {len(complement_600):4d} DAN prompts (complement)")
    print(f"             3 subjects (qwen_3_32b, gemma_4_31b ON+OFF) + Shah-reconstructed (1100)")
    print(f"             {300 - int(cfg['n_role_rollouts_per_role'])} more rollouts/role for τ-calibration")
    print(f"             T2.5 capability + T2.7b ordinal LASSO + T2.4 cross-check judge")
    print()
    print(f"Follow-up runs ranked by 'time-to-execute if Plan B succeeds early':")
    for run in scope["follow_up_runs_if_time"]:
        print(f"  - {run['name']:42s}  ~{run['estimated_compute_hours']} hr")


if __name__ == "__main__":
    main()
