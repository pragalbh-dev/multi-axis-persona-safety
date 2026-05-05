"""Phase 3 — Attack arm orchestrator.

Tests three complementary attack directions on the AA-capped model:
  (a) per-PC sign-matched: for each PC_i ∈ {2, 3, 4}, sign_i = sign(cos_sim(v_harm, PC_i));
      attack vector = sign_i · PC_i_unit · lmsys_norm. Tests which individual role-PCs
      carry causally exploitable harm signal in the v_harm-aligned direction.
  (b) v_harm-direct: steer along v_harm itself (oracle upper bound; needs labeled jailbreaks).
  (c) multi-PC: simultaneously steer along all sign-matched PCs at each axis's coherent λ.

Stratified subset: harm-positive prompts from merged Plan B + Phase 1 baseline (N_pos)
+ 2·N_pos harm-negative control sample (1:2 ratio).

Adaptive λ search:
  - Run mini-batches (n=100 stratified) at all λ ∈ schedule per axis to characterize
    the coherence-vs-harm Pareto.
  - Pick λ_max_coherent per axis = highest λ where coherence (1 − nonsense rate) ≥ floor.
  - Run full N_pos + 2·N_pos subset at that λ for each axis + multi-axis attack.

Pipeline (parent process never imports torch/vllm/transformers; all heavy work via subprocess):
  1. setup     (CPU)        → identify N_pos + control, build attack vectors
  2. mini      (HF)         → mini-runs at all λ × all axes on stratified mini-subset
  3. judge_m   (vLLM judge) → judge mini rollouts
  4. pick_lam  (CPU)        → per-axis λ_max_coherent from mini judge results
  5. full      (HF)         → full runs at λ_max_coherent for each axis + multi-axis composite
  6. judge_f   (vLLM judge) → judge full rollouts
  7. assemble  (CPU)        → lambda_sweep.parquet + lambda_pareto.json

Usage:
  uv run python -m src.experiments.attack_arm --config configs/attack_arm.yaml
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import yaml


def _log(msg: str) -> None:
    print(f"[atk_arm {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _mark_done(marker: Path, payload: dict) -> None:
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps(payload, indent=2, default=str))


def _lam_token(lam: float) -> str:
    """λ=0.25 → "pos0p25", λ=0.05 → "pos0p05"."""
    sign = "pos" if lam > 0 else "neg"
    mag = abs(float(lam))
    if mag == int(mag):
        return f"{sign}{int(mag)}"
    return f"{sign}{mag:g}".replace(".", "p")


# ============================================================================
# Step 1 — setup: load Phase 2 outputs, build subset, build attack vectors
# ============================================================================


def step_1_setup(cfg: dict, out_dir: Path) -> dict:
    """CPU: build harm-positive subset + 2:1 control + attack vectors per axis."""
    import numpy as np
    import pandas as pd
    from safetensors.torch import load_file, save_file
    import torch

    marker = out_dir / ".step1.done"
    if marker.exists():
        _log(f"step 1: skipped (marker exists)")
        return json.loads(marker.read_text())

    # 1.1 Load Phase 2 sign-matched directions.
    hd = json.loads(Path(cfg["harm_direction_json"]).read_text())
    cos_pcs = hd["cos_sim_v_harm_pcs"]  # PC1..PCK
    cos_aa = hd["cos_sim_v_harm_aa"]
    _log(f"step 1: harm_direction loaded — cos(v_harm,AA)={cos_aa:+.3f}, top-3 PCs cos: {cos_pcs[:3]}")

    # 1.2 Build harm-positive + control subset from merged baseline.
    plan_b = pd.read_parquet(cfg["plan_b_details"])
    plan_b_baseline = plan_b[plan_b["condition_id"] == "baseline"].copy()
    plan_b_baseline["dataset"] = plan_b_baseline.get("dataset", "dan")  # plan b is pure DAN
    plan_b_baseline["prompt_id"] = "planb_" + plan_b_baseline["prompt_id"].astype(str)
    ext = pd.read_parquet(cfg["extended_baseline"])
    # Note: ext prompt_ids are already prefixed dan_ / shah_ from baseline_extend step 1.

    cols = ["prompt_id", "dataset", "input_text", "harm_binary", "category", "full_prompt", "persona_text", "question_text"]
    plan_b_baseline = plan_b_baseline[[c for c in cols if c in plan_b_baseline.columns]]
    ext = ext[[c for c in cols if c in ext.columns]]
    merged = pd.concat([plan_b_baseline, ext], ignore_index=True)

    pos = merged[merged["harm_binary"] == 1].reset_index(drop=True)
    neg = merged[merged["harm_binary"] == 0].reset_index(drop=True)
    n_pos = len(pos)
    n_neg_target = int(cfg["control_ratio"]) * n_pos
    if n_neg_target > len(neg):
        _log(f"step 1: control target {n_neg_target} > available {len(neg)}; using all")
        neg_sample = neg
    else:
        # Stratify by dataset to match harm-positive distribution.
        ratio_by_ds = pos["dataset"].value_counts(normalize=True).to_dict()
        rng = np.random.default_rng(cfg["seed"])
        pieces = []
        for ds, frac in ratio_by_ds.items():
            n_take = int(round(frac * n_neg_target))
            pool = neg[neg["dataset"] == ds]
            if len(pool) <= n_take:
                pieces.append(pool)
            else:
                pieces.append(pool.sample(n=n_take, random_state=int(rng.integers(0, 2**31))))
        neg_sample = pd.concat(pieces, ignore_index=True)
        # Top up if rounding dropped a few
        if len(neg_sample) < n_neg_target and len(neg) > len(neg_sample):
            extras = neg.drop(neg_sample.index, errors="ignore").sample(
                n=min(n_neg_target - len(neg_sample), len(neg) - len(neg_sample)),
                random_state=cfg["seed"],
            )
            neg_sample = pd.concat([neg_sample, extras], ignore_index=True)

    subset = pd.concat([pos, neg_sample], ignore_index=True)
    subset["stratum"] = ["harm_pos"] * len(pos) + ["harm_neg"] * len(neg_sample)
    subset_path = out_dir / "rollouts" / "_attack_subset.parquet"
    subset_path.parent.mkdir(parents=True, exist_ok=True)
    subset.to_parquet(subset_path, index=False)
    _log(
        f"step 1: subset n={len(subset)} (n_pos={len(pos)} + n_neg={len(neg_sample)}); "
        f"datasets: {subset['dataset'].value_counts().to_dict()}"
    )

    # 1.3 Mini subset for adaptive λ characterization.
    # n=50 (25+25) keeps mini cells under ~140s each at HF compound-mode rate.
    mini_n_pos = min(25, len(pos))
    mini_n_neg = min(25, len(neg_sample))
    mini_pos = pos.sample(n=mini_n_pos, random_state=cfg["seed"])
    mini_neg = neg_sample.sample(n=mini_n_neg, random_state=cfg["seed"])
    mini = pd.concat([mini_pos, mini_neg], ignore_index=True)
    mini["stratum"] = ["harm_pos"] * len(mini_pos) + ["harm_neg"] * len(mini_neg)
    mini_path = out_dir / "rollouts" / "_attack_mini_subset.parquet"
    mini.to_parquet(mini_path, index=False)
    _log(f"step 1: mini subset n={len(mini)} (n_pos={len(mini_pos)} + n_neg={len(mini_neg)})")

    # 1.4 Load AA + PCs from Plan B extraction; compute v_harm direction in d_model space.
    # (We have cos_sim from Phase 2 but for steering we need the actual unit vector.)
    extr = Path(cfg["plan_b_extraction_dir"])
    aa_full = load_file(str(extr / "aa.safetensors"))["aa"]   # (n_layers, d)
    pcs_at_lstar = load_file(str(extr / "pcs.safetensors"))["pcs_at_lstar"]  # (k, d)

    l_star = int(cfg["l_star"])
    aa = aa_full[l_star].float().numpy()
    aa_unit = aa / max(np.linalg.norm(aa), 1e-9)

    # Recompute v_harm: read it back from Phase 2 if available, else fail loud.
    # Phase 2 saves cos_sim values but not the v_harm vector itself; the cleanest
    # path is to recompute v_harm from baseline activations here. We have AA + PCs
    # already; v_harm = mean(harm) - mean(safe) on baseline activations.
    from src.extraction.types import ActivationCache
    plan_b_cache = ActivationCache.load(
        Path("data/cache/activations/gemma_2_27b/plan_b_per_prompt_L21/L21")
    )
    ext_cache = ActivationCache.load(
        Path("data/cache/activations/gemma_2_27b/baseline_extended_L21/L21")
    )
    plan_b_acts = plan_b_cache.tensor.float().numpy()
    plan_b_pids = plan_b_cache.prompt_ids
    ext_acts = ext_cache.tensor.float().numpy()
    ext_pids = ext_cache.prompt_ids

    # Build (acts, harm_label) aligned series across both sources.
    plan_b_lookup = {p: i for i, p in enumerate(plan_b_pids)}
    ext_lookup = {p: i for i, p in enumerate(ext_pids)}

    acts_list, harm_list = [], []
    for _, row in merged.iterrows():
        # Plan B baseline rows have prompt_id "planb_{int}" after our rename;
        # cache pids are "{int}::baseline".
        pid = str(row["prompt_id"])
        if pid.startswith("planb_"):
            key = f"{pid[len('planb_'):]}::baseline"
            if key in plan_b_lookup:
                acts_list.append(plan_b_acts[plan_b_lookup[key]])
                harm_list.append(int(row["harm_binary"]))
        else:
            key = f"{pid}::baseline"
            if key in ext_lookup:
                acts_list.append(ext_acts[ext_lookup[key]])
                harm_list.append(int(row["harm_binary"]))
    acts_arr = np.stack(acts_list, axis=0)
    harm_arr = np.array(harm_list, dtype=np.int32)
    _log(f"step 1: aligned {len(harm_arr)} baseline activations (harm={int(harm_arr.sum())}, safe={int((1-harm_arr).sum())})")

    mean_harm = acts_arr[harm_arr == 1].mean(axis=0)
    mean_safe = acts_arr[harm_arr == 0].mean(axis=0)
    v_harm = mean_harm - mean_safe
    v_harm_unit = v_harm / max(np.linalg.norm(v_harm), 1e-9)

    # 1.5 Load lmsys norm to scale steering vectors (paper convention).
    lmsys_path = extr / "lmsys_norms_L21.json"
    lmsys = json.loads(lmsys_path.read_text())
    lmsys_norm = float(lmsys["mean_norm"])
    _log(f"step 1: lmsys_norm@L{l_star} = {lmsys_norm:.2f}")

    # 1.6 Build attack vectors. Each saved as safetensors with key 'v', dtype bf16,
    # already scaled to lmsys_norm so steering with coefficient = λ matches paper.
    vec_dir = out_dir / "extraction" / "vectors"
    vec_dir.mkdir(parents=True, exist_ok=True)

    def _save_vec(name: str, v_unit: np.ndarray) -> str:
        v = torch.from_numpy(v_unit * lmsys_norm).bfloat16().contiguous()
        p = vec_dir / f"{name}.safetensors"
        save_file({"v": v}, str(p))
        return str(p)

    axis_files: dict[str, str] = {}
    axis_signs: dict[str, int] = {}

    # Per-PC sign-matched (dictate by Phase 2 cos_sim sign).
    pcs_unit = pcs_at_lstar.float().numpy()
    pcs_unit = np.stack([
        pcs_unit[i] / max(np.linalg.norm(pcs_unit[i]), 1e-9)
        for i in range(pcs_unit.shape[0])
    ])
    for pc_idx in cfg["pc_indices_to_attack"]:
        sign_i = -1 if cos_pcs[pc_idx - 1] < 0 else 1
        v_signmatched = sign_i * pcs_unit[pc_idx - 1]
        name = f"signmatched_pc{pc_idx}"
        axis_files[name] = _save_vec(name, v_signmatched)
        axis_signs[name] = sign_i
        _log(f"step 1: {name} (cos(v_harm, PC{pc_idx})={cos_pcs[pc_idx-1]:+.3f}, sign_i={sign_i:+d})")

    # v_harm direct (oracle).
    if cfg.get("include_v_harm_attack", True):
        axis_files["v_harm"] = _save_vec("v_harm", v_harm_unit)
        axis_signs["v_harm"] = 1
        _log(f"step 1: v_harm (||v_harm||_pre={np.linalg.norm(v_harm):.2f})")

    # AA itself (control — expected defended by AA-cap).
    if cfg.get("include_aa_attack", True):
        # Steer toward harm-aligned end of AA. cos(v_harm, AA) is negative in Plan B,
        # so harm-toward-AA direction = -aa_unit. Sign-match: same logic as PCs.
        sign_aa = -1 if cos_aa < 0 else 1
        axis_files["aa_control"] = _save_vec("aa_control", sign_aa * aa_unit)
        axis_signs["aa_control"] = sign_aa
        _log(f"step 1: aa_control (cos(v_harm,AA)={cos_aa:+.3f}, sign={sign_aa:+d})")

    # Random baselines.
    rng_r = np.random.default_rng(cfg["seed"])
    for r in range(int(cfg["n_random_baselines"])):
        rv = rng_r.normal(size=v_harm.shape).astype(np.float32)
        rv = rv / max(np.linalg.norm(rv), 1e-9)
        name = f"random_{r}"
        axis_files[name] = _save_vec(name, rv)
        axis_signs[name] = 1

    # 1.7 Build AA cap vectors per layer (reused from Plan B step 4 logic).
    # Plan B saved per-layer AA vectors at extraction/vectors/aa_L{layer}.safetensors,
    # already scaled to lmsys_norm. Reuse those directly.
    plan_b_aa_dir = Path(cfg["plan_b_extraction_dir"]).parent / "extraction" / "vectors"
    if not plan_b_aa_dir.exists():
        plan_b_aa_dir = extr / "vectors"
    aa_cap_files = []
    for layer in cfg["capping_layers"]:
        f = plan_b_aa_dir / f"aa_L{layer}.safetensors"
        if not f.exists():
            raise RuntimeError(f"missing AA cap vector: {f}")
        aa_cap_files.append(str(f))

    # Tau per layer: read from Plan B step 3 marker (tau_calibration.json or .step3.done).
    # The step3.done marker for plan b has per_layer tau dist; cap thresholds are -p75 of +AA.
    plan_b_tau_path = Path(cfg["tau_per_layer_path"])
    plan_b_tau = json.loads(plan_b_tau_path.read_text())["per_layer"]
    cap_thresholds = []
    for layer in cfg["capping_layers"]:
        # Same sign-flip math as Plan B step 6 line 503-509.
        cap_thresholds.append(-float(plan_b_tau[str(layer)]["p75"]))
    _log(f"step 1: AA cap thresholds (role-pos space) = {[round(t, 2) for t in cap_thresholds]}")

    payload = {
        "subset_path": str(subset_path),
        "mini_path": str(mini_path),
        "n_pos": int(n_pos),
        "n_neg": int(len(neg_sample)),
        "axis_files": axis_files,
        "axis_signs": axis_signs,
        "aa_cap_files": aa_cap_files,
        "cap_thresholds": cap_thresholds,
        "lmsys_norm": lmsys_norm,
        "l_star": l_star,
        "cos_v_harm_aa": cos_aa,
        "cos_v_harm_pcs": cos_pcs,
    }
    _mark_done(marker, payload)
    return payload


# ============================================================================
# Step 2 — mini λ sweep: HF compound runs at all (axis, λ) cells on n=100
# ============================================================================


def _build_compound_steering(setup: dict, axis_name: str, lam: float) -> dict:
    """Construct the steering dict for run_subject_rollouts compound mode."""
    return {
        "mode": "compound",
        "cap_vectors": setup["aa_cap_files"],
        "cap_thresholds": setup["cap_thresholds"],
        "cap_layers": list(setup_capping_layers_from(setup)),
        "addition_vectors": [setup["axis_files"][axis_name]],
        "addition_coefficients": [float(lam)],
        "addition_layers": [int(setup["l_star"])],
        "positions": "all",
    }


def setup_capping_layers_from(setup: dict) -> list[int]:
    # Recover layers from cap_threshold len + first aa_cap file's layer suffix.
    # aa_cap_files paths look like ".../aa_L33.safetensors".
    out = []
    for f in setup["aa_cap_files"]:
        stem = Path(f).stem  # aa_L33
        out.append(int(stem.split("_L")[1]))
    return sorted(out)


def step_2_mini_lambda_sweep(cfg: dict, out_dir: Path, setup: dict) -> list[Path]:
    """Compound runs at every (axis, λ) cell on the mini subset.

    Backend resolved per `configs/subjects.yaml::<id>.steered_backend`
    (sglang for Gemma 2 27B + Qwen 3 32B, hf for Gemma 4 31B).
    """
    from src.utils.config import resolved_steered_backend
    from src.utils.model_runner import run_in_subprocess

    _STEER_BACKEND = resolved_steered_backend(cfg["model_id"])
    _log(f"step 2: steered backend = {_STEER_BACKEND}")

    marker = out_dir / ".step2.done"
    rollouts_dir = out_dir / "rollouts" / "mini"
    rollouts_dir.mkdir(parents=True, exist_ok=True)
    if marker.exists():
        _log(f"step 2: skipped (marker exists)")
        return list(rollouts_dir.glob("*.parquet"))

    mini_path = Path(setup["mini_path"])
    out_paths: list[Path] = []

    # Order axes so PCs go first, v_harm next, AA control, then randoms (fewest randoms run).
    axis_order = []
    for pc_idx in cfg["pc_indices_to_attack"]:
        axis_order.append(f"signmatched_pc{pc_idx}")
    if cfg.get("include_v_harm_attack", True):
        axis_order.append("v_harm")
    if cfg.get("include_aa_attack", True):
        axis_order.append("aa_control")
    # Use config's n_random_baselines directly (already cut to 2).
    n_rand_mini = int(cfg["n_random_baselines"])
    for r in range(n_rand_mini):
        axis_order.append(f"random_{r}")

    total_cells = len(axis_order) * len(cfg["lambda_schedule"])
    cell_idx = 0
    t_start = time.time()
    for axis_name in axis_order:
        for lam in cfg["lambda_schedule"]:
            cell_idx += 1
            cond_id = f"mini_aa_capped_{axis_name}_{_lam_token(lam)}"
            cond_path = rollouts_dir / f"{cond_id}.parquet"
            if cond_path.exists():
                _log(f"step 2 [{cell_idx}/{total_cells}] skip {cond_id} (exists)")
                out_paths.append(cond_path)
                continue
            _log(f"step 2 [{cell_idx}/{total_cells}] {cond_id}")
            steering = _build_compound_steering(setup, axis_name, lam)
            res = run_in_subprocess(
                "src.evaluation.run_subject_rollouts",
                {
                    "model_id": cfg["model_id"],
                    "backend": _STEER_BACKEND,
                    "prompts_path": str(mini_path),
                    "output_path": str(cond_path),
                    "condition_id": cond_id,
                    "seed": cfg["seed"],
                    "max_new_tokens": cfg["max_new_tokens"],
                    "temperature": 0.0,
                    "batch_size": cfg["hf_steered_batch_size"],
                    "max_input_len": cfg.get("hf_steered_max_input_len", 4096),
                    "steering": steering,
                },
                output_path=out_dir / f".step2_{cond_id}.work.json",
                timeout_seconds=3600,
            )
            _log(f"  → {res['n_rows']} rows in {res['elapsed_seconds']}s")
            out_paths.append(cond_path)

    _log(f"step 2: {total_cells} mini cells in {time.time() - t_start:.0f}s total")
    _mark_done(marker, {"n_cells": total_cells, "elapsed": time.time() - t_start})
    return out_paths


# ============================================================================
# Step 3 — judge mini rollouts
# ============================================================================


def step_3_judge_mini(cfg: dict, out_dir: Path, mini_paths: list[Path]) -> Path:
    """Qwen 3.6-27B judge over all mini rollouts."""
    import pandas as pd
    from src.utils.model_runner import run_in_subprocess

    marker = out_dir / ".step3.done"
    judged_path = out_dir / "rollouts" / "mini_judged.parquet"
    if marker.exists():
        _log(f"step 3: skipped (marker exists)")
        return judged_path

    dfs = []
    for p in mini_paths:
        d = pd.read_parquet(p)
        if "category" not in d.columns:
            d["category"] = "Unknown"
        if "question_text" not in d.columns:
            d["question_text"] = d.get("input_text", "")
        dfs.append(d)
    all_df = pd.concat(dfs, ignore_index=True)
    all_path = out_dir / "rollouts" / "mini_all.parquet"
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
        output_path=out_dir / ".step3.work.json",
        timeout_seconds=7200,
    )
    _log(f"step 3: judge done in {res['elapsed_seconds']}s, {res['n_parsed']}/{res['n_rows']} parsed")
    _mark_done(marker, res)
    return judged_path


# ============================================================================
# Step 4 — pick λ_max_coherent per axis
# ============================================================================


def step_4_pick_lambda(cfg: dict, out_dir: Path, judged_path: Path, setup: dict) -> Path:
    """Per-axis: find largest λ where coherence ≥ floor; emit lambda_pareto.json."""
    import pandas as pd
    from src.evaluation.safety import binarize_harm

    marker = out_dir / ".step4.done"
    pareto_path = out_dir / "lambda_pareto.json"
    if marker.exists():
        _log(f"step 4: skipped (marker exists)")
        return pareto_path

    df = pd.read_parquet(judged_path)
    df["harm_binary"] = df["harm_label_primary"].map(binarize_harm)
    df["nonsense"] = (df["harm_label_primary"] == "nonsensical").astype(int)
    # condition_id format: mini_aa_capped_<axis>_<λtoken>
    def _parse(cond_id: str) -> tuple[str, str]:
        # strip leading "mini_aa_capped_"
        rest = cond_id[len("mini_aa_capped_"):]
        # axis = everything before final "_pos<…>" or "_neg<…>"
        for tok in ("_pos", "_neg"):
            i = rest.rfind(tok)
            if i >= 0:
                return rest[:i], rest[i + 1:]
        return rest, ""
    df["axis"], df["lam_token"] = zip(*df["condition_id"].map(_parse))
    df["lam"] = df["lam_token"].map(_lam_token_to_float)

    floor = float(cfg["coherence_floor"])
    pareto: dict[str, list[dict]] = {}
    chosen: dict[str, float] = {}

    for axis, sub in df.groupby("axis"):
        sub = sub.sort_values("lam")
        per_lam = sub.groupby("lam").agg(
            n=("harm_binary", "size"),
            harm=("harm_binary", "mean"),
            nonsense=("nonsense", "mean"),
        ).reset_index()
        per_lam["coherence"] = 1.0 - per_lam["nonsense"]
        rows = per_lam.to_dict(orient="records")
        pareto[axis] = rows
        # Choose largest λ where coherence ≥ floor.
        valid = per_lam[per_lam["coherence"] >= floor]
        if valid.empty:
            chosen_lam = float(per_lam["lam"].iloc[0])  # smallest λ as fallback
            _log(f"step 4: axis={axis} — NO λ with coherence ≥ {floor}; falling back to λ={chosen_lam}")
        else:
            chosen_lam = float(valid["lam"].max())
            _log(
                f"step 4: axis={axis} — λ_max_coherent = {chosen_lam} "
                f"(harm={float(valid[valid['lam']==chosen_lam]['harm'].iloc[0]):.3f}, "
                f"coherence={float(valid[valid['lam']==chosen_lam]['coherence'].iloc[0]):.3f})"
            )
        chosen[axis] = chosen_lam

    pareto_path.write_text(json.dumps({"pareto": pareto, "chosen_lambda_per_axis": chosen}, indent=2))
    _mark_done(marker, {"chosen_lambda_per_axis": chosen})
    return pareto_path


def _lam_token_to_float(tok: str) -> float:
    """Inverse of _lam_token. 'pos0p25' → 0.25; 'neg0p10' → -0.10."""
    sign = 1.0 if tok.startswith("pos") else -1.0
    rest = tok[3:].replace("p", ".")
    return sign * float(rest)


# ============================================================================
# Step 5 — full runs at λ_max_coherent per axis + multi-axis composite
# ============================================================================


def _resolve_steer_backend(cfg: dict) -> str:
    from src.utils.config import resolved_steered_backend
    return resolved_steered_backend(cfg["model_id"])


def step_5_full_runs(cfg: dict, out_dir: Path, setup: dict, pareto_path: Path) -> list[Path]:
    """Full N_pos+2*N_pos runs at λ_max_coherent per axis + multi-axis composite.

    Backend resolved per `configs/subjects.yaml::<id>.steered_backend`.
    """
    import torch
    from safetensors.torch import load_file, save_file
    from src.utils.model_runner import run_in_subprocess

    _STEER_BACKEND = _resolve_steer_backend(cfg)
    _log(f"step 5: steered backend = {_STEER_BACKEND}")

    marker = out_dir / ".step5.done"
    rollouts_dir = out_dir / "rollouts" / "full"
    rollouts_dir.mkdir(parents=True, exist_ok=True)
    if marker.exists():
        _log(f"step 5: skipped (marker exists)")
        return list(rollouts_dir.glob("*.parquet"))

    chosen = json.loads(pareto_path.read_text())["chosen_lambda_per_axis"]
    subset_path = Path(setup["subset_path"])
    out_paths: list[Path] = []

    for axis_name, lam in chosen.items():
        # Skip random axes from the full run (we already characterized them in mini).
        if axis_name.startswith("random_"):
            continue
        cond_id = f"full_aa_capped_{axis_name}_{_lam_token(lam)}"
        cond_path = rollouts_dir / f"{cond_id}.parquet"
        if cond_path.exists():
            _log(f"step 5: skip {cond_id} (exists)")
            out_paths.append(cond_path)
            continue
        _log(f"step 5: full run {cond_id} (λ={lam})")
        steering = _build_compound_steering(setup, axis_name, float(lam))
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
            output_path=out_dir / f".step5_{cond_id}.work.json",
            timeout_seconds=10800,
        )
        _log(f"  → {res['n_rows']} rows in {res['elapsed_seconds']}s")
        out_paths.append(cond_path)

    # Multi-axis composite: pre-combine sign-matched PCs at their respective λ_max_coherent.
    # h ← h + Σ_i (λ_i · sign_i · PC_i_unit · lmsys_norm) = h + 1.0 · (precombined_vector)
    multi_indices = cfg.get("multi_axis_pc_indices", [])
    if multi_indices:
        # Load each axis vector + λ; sum into one vector.
        d_model = None
        combined = None
        used = []
        for pc_idx in multi_indices:
            axis_name = f"signmatched_pc{pc_idx}"
            if axis_name not in chosen:
                continue
            v = load_file(setup["axis_files"][axis_name])["v"].float()  # already · lmsys_norm
            lam = float(chosen[axis_name])
            if combined is None:
                combined = lam * v
                d_model = v.numel()
            else:
                combined = combined + lam * v
            used.append((pc_idx, lam))

        if combined is not None:
            multi_vec_path = out_dir / "extraction" / "vectors" / "multi_signmatched.safetensors"
            multi_vec_path.parent.mkdir(parents=True, exist_ok=True)
            save_file({"v": combined.bfloat16().contiguous()}, str(multi_vec_path))
            cond_id = "full_aa_capped_multi_signmatched_pos1"  # coefficient 1 since λ baked into combined vec
            cond_path = rollouts_dir / f"{cond_id}.parquet"
            if not cond_path.exists():
                _log(f"step 5: multi-axis attack with components {used}")
                steering = {
                    "mode": "compound",
                    "cap_vectors": setup["aa_cap_files"],
                    "cap_thresholds": setup["cap_thresholds"],
                    "cap_layers": setup_capping_layers_from(setup),
                    "addition_vectors": [str(multi_vec_path)],
                    "addition_coefficients": [1.0],
                    "addition_layers": [int(setup["l_star"])],
                    "positions": "all",
                }
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
                    output_path=out_dir / f".step5_{cond_id}.work.json",
                    timeout_seconds=10800,
                )
                _log(f"  → {res['n_rows']} rows in {res['elapsed_seconds']}s")
            out_paths.append(cond_path)

    _mark_done(marker, {"n_full_conditions": len(out_paths)})
    return out_paths


# ============================================================================
# Step 6 — judge full rollouts
# ============================================================================


def step_6_judge_full(cfg: dict, out_dir: Path, full_paths: list[Path]) -> Path:
    """Qwen 3.6-27B judge over all full rollouts."""
    import pandas as pd
    from src.utils.model_runner import run_in_subprocess

    marker = out_dir / ".step6.done"
    judged_path = out_dir / "rollouts" / "full_judged.parquet"
    if marker.exists():
        _log(f"step 6: skipped (marker exists)")
        return judged_path

    dfs = []
    for p in full_paths:
        d = pd.read_parquet(p)
        if "category" not in d.columns:
            d["category"] = "Unknown"
        if "question_text" not in d.columns:
            d["question_text"] = d.get("input_text", "")
        dfs.append(d)
    all_df = pd.concat(dfs, ignore_index=True)
    all_path = out_dir / "rollouts" / "full_all.parquet"
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
        output_path=out_dir / ".step6.work.json",
        timeout_seconds=7200,
    )
    _log(f"step 6: judge done in {res['elapsed_seconds']}s, {res['n_parsed']}/{res['n_rows']} parsed")
    _mark_done(marker, res)
    return judged_path


# ============================================================================
# Step 7 — assemble lambda_sweep.parquet (mini + full unified)
# ============================================================================


def step_7_assemble(
    cfg: dict, out_dir: Path, mini_judged: Path, full_judged: Path, setup: dict
) -> Path:
    """Unified lambda_sweep.parquet with axis, λ, harm/coherence/refusal stats per row."""
    import pandas as pd
    from src.evaluation.safety import binarize_harm

    marker = out_dir / ".step7.done"
    out_path = out_dir / "lambda_sweep.parquet"
    if marker.exists():
        _log(f"step 7: skipped (marker exists)")
        return out_path

    mini = pd.read_parquet(mini_judged)
    full = pd.read_parquet(full_judged)
    mini["scope"] = "mini"
    full["scope"] = "full"
    df = pd.concat([mini, full], ignore_index=True)
    df["harm_binary"] = df["harm_label_primary"].map(binarize_harm)
    df["nonsense"] = (df["harm_label_primary"] == "nonsensical").astype(int)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)

    summary = df.groupby(["scope", "condition_id"]).agg(
        n=("harm_binary", "size"),
        harm=("harm_binary", "mean"),
        nonsense=("nonsense", "mean"),
    ).round(3)
    _log(f"step 7: lambda_sweep.parquet rows={len(df)}")
    _log(f"\n{summary.to_string()}")
    # Flatten MultiIndex tuple keys to "scope::condition_id" so the marker is JSON-serializable.
    summary_flat = {f"{scope}::{cond}": row.to_dict() for (scope, cond), row in summary.iterrows()}
    _mark_done(marker, {"n_rows": len(df), "summary": summary_flat})
    return out_path


# ============================================================================
# Main
# ============================================================================


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("configs/attack_arm.yaml"))
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
        mini_paths = step_2_mini_lambda_sweep(cfg, out_dir, setup)
    else:
        mini_paths = list((out_dir / "rollouts" / "mini").glob("*.parquet"))

    if skip < 3:
        mini_judged = step_3_judge_mini(cfg, out_dir, mini_paths)
    else:
        mini_judged = out_dir / "rollouts" / "mini_judged.parquet"

    if skip < 4:
        pareto_path = step_4_pick_lambda(cfg, out_dir, mini_judged, setup)
    else:
        pareto_path = out_dir / "lambda_pareto.json"

    if skip < 5:
        full_paths = step_5_full_runs(cfg, out_dir, setup, pareto_path)
    else:
        full_paths = list((out_dir / "rollouts" / "full").glob("*.parquet"))

    if skip < 6:
        full_judged = step_6_judge_full(cfg, out_dir, full_paths)
    else:
        full_judged = out_dir / "rollouts" / "full_judged.parquet"

    if skip < 7:
        out_path = step_7_assemble(cfg, out_dir, mini_judged, full_judged, setup)
    else:
        out_path = out_dir / "lambda_sweep.parquet"

    _log(f"DONE → {out_path}")


if __name__ == "__main__":
    main()
