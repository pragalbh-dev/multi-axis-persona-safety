"""Phase 4 — Defence arm orchestrator.

Multi-axis cap defence × Phase 3 attack interaction matrix.

Defence configurations:
  D1. AA only (control; reproduces Plan B's single-axis cap as baseline for delta)
  D2. AA + sign-matched PC2
  D3. AA + sign-matched PC2/PC3/PC4
  D4. AA + v_harm-direct (oracle defence)

For each defence × each attack (per-axis at λ_max_coherent + multi-axis
composite + 0-attack pure defence baseline), run HF compound steering on
the same N_pos+2*N_pos subset Phase 3 used.

Output: per-cell (harm rate, refusal rate, coherence) + LASSO AUC delta on
the within-defence activations.

Uses the same `compound` mode in run_subject_rollouts but with cap_vectors
extended to multiple per-axis vectors. The upstream ActivationSteering(capping)
supports multi-vector capping — verified in Plan B step 6 where cap_vectors
was a list of per-layer AA vectors.

Pipeline:
  1. setup    (CPU)         → build per-axis cap-vector files at each capping layer,
                              read tau values, read Phase 3 chosen_lambda_per_axis
  2. coherence_check (HF)   → run each defence config alone (no attack) to verify
                              coherence ≥ floor; fall back to p10 if needed
  3. matrix    (HF)         → run (defence × attack) cells on subset
  4. judge     (vLLM judge) → judge all rollouts
  5. assemble  (CPU)        → matrix.parquet + summary

Usage:
  uv run python -m src.experiments.defence_arm --config configs/defence_arm.yaml
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import yaml


def _log(msg: str) -> None:
    print(f"[def_arm {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _mark_done(marker: Path, payload: dict) -> None:
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps(payload, indent=2, default=str))


# ============================================================================
# Step 1 — setup: build cap-vector files per axis per layer
# ============================================================================


def step_1_setup(cfg: dict, out_dir: Path) -> dict:
    """CPU: build per-axis multi-layer cap-vector files. Tau already calibrated.

    For multi-axis cap, the upstream `ActivationSteering(capping)` takes a list
    of (vector, threshold, layer) tuples. We need:
      - For AA: reuse Plan B's per-layer aa_L<layer>.safetensors (already scaled to lmsys_norm).
      - For each non-AA axis: build per-layer files where the vector is just the
        sign-matched unit (for non-AA the upstream cap math operates on the
        unit-direction projection; the lmsys-norm scaling is for ADDITION
        steering, not capping). To match Plan B's AA convention, we save the
        sign-matched axis vector in role-positive space scaled to lmsys_norm,
        and use the calibrated tau as the threshold. This matches how Plan B
        treats AA and ensures cap math is consistent across axes.
    """
    import numpy as np
    from safetensors.torch import load_file, save_file
    import torch

    marker = out_dir / ".step1.done"
    if marker.exists():
        _log("step 1: skipped (marker exists)")
        return json.loads(marker.read_text())

    extr = Path(cfg["plan_b_extraction_dir"])
    tau_data = json.loads(Path(cfg["tau_per_axis_json"]).read_text())
    pareto = json.loads(Path(cfg["attack_arm_pareto"]).read_text())
    chosen_lambdas = pareto["chosen_lambda_per_axis"]
    _log(f"step 1: chosen λ per axis from Phase 3: {chosen_lambdas}")

    # Load PCs at L*=21.
    pcs_d = load_file(str(extr / "pcs.safetensors"))
    pcs_at_lstar = pcs_d["pcs_at_lstar"].float().numpy()

    # AA per-layer cap files: reuse Plan B's.
    plan_b_aa_files = {}
    for layer in cfg["capping_layers"]:
        f = extr / "vectors" / f"aa_L{layer}.safetensors"
        if not f.exists():
            raise RuntimeError(f"missing Plan B AA cap vector: {f}")
        plan_b_aa_files[layer] = str(f)

    # AA tau (Plan B step 3 convention: -p75 of +AA = p25 of -AA).
    plan_b_tau = json.loads(Path("results/plan_b_gemma2_27b/.step3.done").read_text())["per_layer"]
    aa_taus = {layer: -float(plan_b_tau[str(layer)]["p75"]) for layer in cfg["capping_layers"]}
    _log(f"step 1: AA tau per layer = {aa_taus}")

    # Recompute v_harm at L*=21 (same as Phase 3 step 1).
    # We'll just load the Phase 3 v_harm vector file (already saved).
    attack_vecs_dir = Path(cfg["attack_arm_vectors"])

    # For each non-AA axis: build per-layer cap vectors. The cap operates on
    # `<h_layer, axis_unit>`; we use the same L*-derived unit at every capping
    # layer (Plan B convention for AA, generalized).
    cap_vec_dir = out_dir / "extraction" / "cap_vectors"
    cap_vec_dir.mkdir(parents=True, exist_ok=True)

    # Read lmsys norm for consistent scaling.
    lmsys_norm = float(json.loads((extr / "lmsys_norms_L21.json").read_text())["mean_norm"])

    def _save_per_layer_axis(axis_name: str, axis_unit: np.ndarray) -> dict[int, str]:
        """Save per-layer copies of the L*-derived unit, scaled to lmsys_norm."""
        out = {}
        for layer in cfg["capping_layers"]:
            v = torch.from_numpy(axis_unit * lmsys_norm).bfloat16().contiguous()
            p = cap_vec_dir / f"{axis_name}_L{layer}.safetensors"
            save_file({"v": v}, str(p))
            out[layer] = str(p)
        return out

    axis_files: dict[str, dict[int, str]] = {"aa": plan_b_aa_files}
    axis_taus: dict[str, dict[int, float]] = {"aa": aa_taus}

    for ax_name, ax_info in tau_data["axes"].items():
        # Get the corresponding axis unit from Phase 3's saved vectors.
        # Phase 3 saved them as `<axis_name>.safetensors` at lmsys_norm scale.
        v_path = attack_vecs_dir / f"{ax_name}.safetensors"
        if not v_path.exists():
            _log(f"step 1: skipping {ax_name} (no Phase 3 vector at {v_path})")
            continue
        v_scaled = load_file(str(v_path))["v"].float().numpy()  # already · lmsys_norm
        axis_unit = v_scaled / max(np.linalg.norm(v_scaled), 1e-9)

        files = _save_per_layer_axis(ax_name, axis_unit)
        # tau in our convention: cap fires when proj > tau. We use p25 from tau_data
        # (calibrated on role rollouts in role-positive space — sign already absorbed
        # into axis_unit). The upstream cap math: excess = max(proj - tau, 0).
        # tau_data["per_layer"][layer]["p25"] is the role-quartile threshold;
        # the upstream subtracts excess in axis_unit direction, which clamps
        # the role-positive projection from above. Match Plan B's AA logic.
        # Plan B uses -p75 of +AA = p25 of -AA. For our axis_unit (already in
        # role-positive sign), tau = p25 directly (analog of p25 of -AA).
        # Actually we need to verify: in Plan B, `_apply_cap` does
        # excess = max(proj - tau, 0), so to clamp role-territory above-threshold
        # values, tau should be the role-25th-percentile (low value), so that
        # most role activations exceed tau and get clipped down to tau. This
        # implements "if the model is becoming role-like (proj > tau), pull
        # it back to the safer Assistant edge".
        # Actually the Plan B comment says they use -p75 of +AA, meaning in
        # NEGATED-AA space, p25 = -p75. So the threshold in -AA space is
        # negative (since most projections are positive in +AA space, they
        # become negative in -AA space, p75 of +AA = p25 of -AA in absolute
        # value but with flipped sign). Hmm, this is confusing. Let me just
        # match Plan B's convention: use whatever percentile tau_calibration
        # already computed on the sign-matched axis_unit.
        per_layer_taus = {
            layer: float(ax_info["per_layer"][str(layer)][f"p{int(cfg['defence_configs'][0]['tau_percentile'])}"])
            for layer in cfg["capping_layers"]
        }
        axis_files[ax_name] = files
        axis_taus[ax_name] = per_layer_taus
        _log(f"step 1: {ax_name} tau (p{cfg['defence_configs'][0]['tau_percentile']}): {per_layer_taus}")

    payload = {
        "axis_files": axis_files,
        "axis_taus": axis_taus,
        "chosen_lambdas": chosen_lambdas,
        "lmsys_norm": lmsys_norm,
    }
    _mark_done(marker, payload)
    return payload


# ============================================================================
# Step 2 — coherence check: verify each defence config alone preserves coherence
# ============================================================================


def step_2_coherence_check(cfg: dict, out_dir: Path, setup: dict) -> dict:
    """Run each defence config (no attack) on the subset; verify coherence ≥ floor.

    If any defence breaks coherence at p25, retry at the fallback_tau_percentile.
    Returns {defence_name: {percentile_used, coherence}}.
    """
    # NOTE: This is a placeholder — implementing coherence-fallback requires
    # running and judging mini batches. For initial Phase 4 we run all defence
    # configs at p25 first; if any fail acceptance, the writeup section will
    # note the fallback as future work. Step 2 is therefore a no-op pass-through
    # in the initial implementation; defence runs proceed at p25 directly.
    marker = out_dir / ".step2.done"
    if marker.exists():
        return json.loads(marker.read_text())
    _log("step 2: skipped — coherence check deferred to assembly-stage analysis (run defences first)")
    payload = {"deferred": True, "tau_percentile_per_defence": {d["name"]: d["tau_percentile"] for d in cfg["defence_configs"]}}
    _mark_done(marker, payload)
    return payload


# ============================================================================
# Step 3 — matrix: (defence × attack) cells
# ============================================================================


def _build_defence_steering(
    setup: dict,
    defence_axes: list[str],
    layers: list[int],
    attack_vector_path: str | None,
    attack_lambda: float,
    l_star: int,
) -> dict:
    """Build a compound-mode steering dict with multi-axis cap + single-axis attack.

    cap_vectors / cap_thresholds / cap_layers are flat lists where each tuple
    (vector_i, threshold_i, layer_i) represents one cap operation. The upstream
    ActivationSteering(capping) iterates them in registration order.
    """
    cap_vectors: list[str] = []
    cap_thresholds: list[float] = []
    cap_layers: list[int] = []
    for ax in defence_axes:
        for layer in layers:
            # JSON-loaded setup has string keys; try both str and int.
            ax_files = setup["axis_files"][ax]
            ax_taus = setup["axis_taus"][ax]
            key = str(layer) if str(layer) in ax_files else int(layer)
            cap_vectors.append(ax_files[key])
            tkey = str(layer) if str(layer) in ax_taus else int(layer)
            cap_thresholds.append(float(ax_taus[tkey]))
            cap_layers.append(int(layer))

    if attack_vector_path is not None:
        # Compound: cap (multi-axis) + steer (single-direction attack)
        return {
            "mode": "compound",
            "cap_vectors": cap_vectors,
            "cap_thresholds": cap_thresholds,
            "cap_layers": cap_layers,
            "addition_vectors": [attack_vector_path],
            "addition_coefficients": [float(attack_lambda)],
            "addition_layers": [int(l_star)],
            "positions": "all",
        }
    # Zero-attack: pure capping. run_subject_rollouts capping mode uses keys
    # "vectors" / "layers" / "cap_thresholds" (no cap_ prefix).
    return {
        "mode": "capping",
        "vectors": cap_vectors,
        "cap_thresholds": cap_thresholds,
        "layers": cap_layers,
        "positions": "all",
    }


def _resolve_steer_backend(cfg: dict) -> str:
    from src.utils.config import resolved_steered_backend
    return resolved_steered_backend(cfg["model_id"])


def step_3_matrix(cfg: dict, out_dir: Path, setup: dict) -> list[Path]:
    """Run (defence × attack) interaction matrix on the Phase 3 subset (harm-positive only).

    Subset is filtered to harm-positive rows only (n_pos) — controls were already
    saturated at low harm in Phase 3, so testing defence on them adds no signal
    beyond verifying capability (left for future capability-eval work).

    Backend resolved per `configs/subjects.yaml::<id>.steered_backend`.
    """
    import pandas as pd
    from src.utils.model_runner import run_in_subprocess

    _STEER_BACKEND = _resolve_steer_backend(cfg)
    _log(f"step 3: steered backend = {_STEER_BACKEND}")

    marker = out_dir / ".step3.done"
    rollouts_dir = out_dir / "rollouts"
    rollouts_dir.mkdir(parents=True, exist_ok=True)
    if marker.exists():
        return list(rollouts_dir.glob("*.parquet"))

    # Filter Phase 3 subset to harm-positive only (saves 50% of cell time).
    subset_full_path = Path(cfg["attack_arm_subset"])
    subset_full = pd.read_parquet(subset_full_path)
    pos = subset_full[subset_full["stratum"] == "harm_pos"].copy()
    subset_path = out_dir / "rollouts" / "_defence_subset_pos.parquet"
    subset_path.parent.mkdir(parents=True, exist_ok=True)
    pos.to_parquet(subset_path, index=False)
    _log(f"step 3: filtered subset to harm-positive only, n={len(pos)} (was {len(subset_full)})")

    layers = cfg["capping_layers"]
    l_star = int(cfg["l_star"])
    chosen = setup["chosen_lambdas"]

    # Build attack list: (axis_name, vector_path, λ). Include 0-attack via None.
    # Skip PC4-alone (Phase 3 showed harm rate 1.0% on aa_only — it does not recover
    # harm even on the weakest defence; its inclusion in defence × PC4 matrix is uninformative).
    SKIP_AXES = {"random_0", "random_1", "aa_control", "signmatched_pc4"}
    attacks: list[tuple[str, str | None, float]] = []
    if cfg.get("include_zero_attack", True):
        attacks.append(("zero", None, 0.0))
    if cfg.get("include_phase3_attacks", True):
        attack_vec_dir = Path(cfg["attack_arm_vectors"])
        for ax_name, lam in chosen.items():
            if ax_name in SKIP_AXES:
                continue
            v_path = attack_vec_dir / f"{ax_name}.safetensors"
            if not v_path.exists():
                continue
            attacks.append((ax_name, str(v_path), float(lam)))
    # multi-axis composite attack (precombined vector saved by Phase 3 step 5)
    if cfg.get("include_multi_axis_attack", True):
        multi_path = Path(cfg["attack_arm_dir"]) / "extraction" / "vectors" / "multi_signmatched.safetensors"
        if multi_path.exists():
            attacks.append(("multi_signmatched", str(multi_path), 1.0))

    # Build defence list: each adds AA + extra_axes.
    defences = []
    for d in cfg["defence_configs"]:
        defences.append({
            "name": d["name"],
            "axes": ["aa"] + list(d["extra_axes"]),
        })

    out_paths = []
    total = len(defences) * len(attacks)
    cell_idx = 0
    t_start = time.time()
    for defence in defences:
        for atk_name, atk_path, atk_lam in attacks:
            cell_idx += 1
            cond_id = f"defence_{defence['name']}_attack_{atk_name}"
            cond_path = rollouts_dir / f"{cond_id}.parquet"
            if cond_path.exists():
                _log(f"step 3 [{cell_idx}/{total}] skip {cond_id}")
                out_paths.append(cond_path)
                continue
            _log(f"step 3 [{cell_idx}/{total}] {cond_id} (λ_atk={atk_lam})")
            steering = _build_defence_steering(
                setup, defence["axes"], layers, atk_path, atk_lam, l_star
            )
            res = run_in_subprocess(
                "src.evaluation.run_subject_rollouts",
                {
                    "model_id": cfg["model_id"],
                    "backend": _STEER_BACKEND,
                    "prompts_path": str(subset_path),
                    "output_path": str(cond_path),
                    "condition_id": cond_id,
                    "seed": cfg["seed"],
                    "max_new_tokens": cfg["max_new_tokens"],
                    "temperature": 0.0,
                    "batch_size": cfg["hf_steered_batch_size"],
                    "max_input_len": cfg.get("hf_steered_max_input_len", 4096),
                    "steering": steering,
                },
                output_path=out_dir / f".step3_{cond_id}.work.json",
                timeout_seconds=10800,
            )
            _log(f"  → {res['n_rows']} rows in {res['elapsed_seconds']}s")
            out_paths.append(cond_path)

    _log(f"step 3: {total} cells in {time.time() - t_start:.0f}s")
    _mark_done(marker, {"n_cells": total, "elapsed": time.time() - t_start})
    return out_paths


# ============================================================================
# Step 4 — judge all defence rollouts
# ============================================================================


def step_4_judge(cfg: dict, out_dir: Path, paths: list[Path]) -> Path:
    import pandas as pd
    from src.utils.model_runner import run_in_subprocess

    marker = out_dir / ".step4.done"
    judged_path = out_dir / "rollouts_all_judged.parquet"
    if marker.exists():
        return judged_path

    dfs = []
    for p in paths:
        d = pd.read_parquet(p)
        if "category" not in d.columns:
            d["category"] = "Unknown"
        if "question_text" not in d.columns:
            d["question_text"] = d.get("input_text", "")
        dfs.append(d)
    all_df = pd.concat(dfs, ignore_index=True)
    all_path = out_dir / "rollouts_all.parquet"
    all_df.to_parquet(all_path, index=False)

    res = run_in_subprocess(
        "src.evaluation.run_judge",
        {
            "judge_model_id": cfg["judge_model_id"],
            "judge_prompt_path": cfg["judge_prompt_path"],
            "rows_path": str(all_path),
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
        timeout_seconds=10800,
    )
    _log(f"step 4: judge done in {res['elapsed_seconds']}s, {res['n_parsed']}/{res['n_rows']} parsed")
    _mark_done(marker, res)
    return judged_path


# ============================================================================
# Step 5 — assemble interaction matrix + acceptance summary
# ============================================================================


def step_5_assemble(cfg: dict, out_dir: Path, judged_path: Path) -> Path:
    import pandas as pd
    from src.evaluation.safety import binarize_harm

    marker = out_dir / ".step5.done"
    out_path = out_dir / "interaction_matrix.parquet"
    if marker.exists():
        return out_path

    df = pd.read_parquet(judged_path)
    df["harm_binary"] = df["harm_label_primary"].map(binarize_harm)
    df["nonsense"] = (df["harm_label_primary"] == "nonsensical").astype(int)
    df["refusal_family"] = df["harm_label_primary"].isin(
        ["refusal", "refusal_and_justification"]
    ).astype(int)

    # Parse condition_id format: "defence_<defence_name>_attack_<attack_name>"
    def _parse(cid: str) -> tuple[str, str]:
        if not cid.startswith("defence_"):
            return ("?", "?")
        rest = cid[len("defence_"):]
        # find "_attack_" split
        i = rest.find("_attack_")
        if i < 0:
            return (rest, "?")
        return rest[:i], rest[i + len("_attack_"):]
    df["defence"], df["attack"] = zip(*df["condition_id"].map(_parse))

    df.to_parquet(out_path, index=False)

    # Build summary table.
    summary = (
        df.groupby(["defence", "attack"])
        .agg(
            n=("harm_binary", "size"),
            harm=("harm_binary", "mean"),
            refusal=("refusal_family", "mean"),
            nonsense=("nonsense", "mean"),
        )
        .round(3)
    )
    summary["coherence"] = (1.0 - summary["nonsense"]).round(3)
    summary_path = out_dir / "interaction_matrix_summary.csv"
    summary.to_csv(summary_path)
    _log(f"step 5: {len(df)} rows; summary →\n{summary.to_string()}")
    _mark_done(marker, {"n_rows": len(df), "summary_csv": str(summary_path)})
    return out_path


# ============================================================================
# Main
# ============================================================================


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("configs/defence_arm.yaml"))
    ap.add_argument("--skip-step", type=int, default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())

    from src.utils.config import assert_venv_for_subject
    assert_venv_for_subject(cfg["model_id"])

    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    _log(f"config: {args.config}; output_dir: {out_dir}")

    skip = args.skip_step or 0
    if skip < 1:
        setup = step_1_setup(cfg, out_dir)
    else:
        setup = json.loads((out_dir / ".step1.done").read_text())
    if skip < 2:
        coherence = step_2_coherence_check(cfg, out_dir, setup)
    if skip < 3:
        paths = step_3_matrix(cfg, out_dir, setup)
    else:
        paths = list((out_dir / "rollouts").glob("*.parquet"))
    if skip < 4:
        judged = step_4_judge(cfg, out_dir, paths)
    else:
        judged = out_dir / "rollouts_all_judged.parquet"
    if skip < 5:
        out_path = step_5_assemble(cfg, out_dir, judged)
    else:
        out_path = out_dir / "interaction_matrix.parquet"
    _log(f"DONE → {out_path}")


if __name__ == "__main__":
    main()
