"""Phase D — Multi-axis defence calibration on Gemma 4 31B thinking-OFF.

Per `plans/may_3_directive.md` 2026-05-03 retrim, thread A. Pipeline
(parent never imports torch/vllm; all heavy work via subprocess):

  1. setup     (CPU)        → load Phase B step1 setup, compute per-layer
                              PC2/PC3 projection percentiles from cached
                              role rollouts, build per-layer PC{2,3} cap
                              vector files, sample 200-prompt validation
                              subset (disjoint from Phase B's 508 test set).
  2. val_pc2   (HF)         → 5 cells, AA + PC2-cap × {p1,p10,p25,p50,p75}
                              under signmatched_pc3 attack at λ=0.25.
                              200 prompts/cell.
  3. judge_v2  (vLLM judge) → judge val_pc2 rollouts.
  4. pick_pc2  (CPU)        → argmin(harm) s.t. coherence ≥ 0.90.
  5. val_pc3   (HF)         → 5 cells, AA + PC2(τ_best) + PC3-cap × percentiles
                              under same PC3 attack. 200 prompts/cell.
  6. judge_v3  (vLLM judge) → judge val_pc3 rollouts.
  7. pick_pc3  (CPU)        → argmin(harm) s.t. coherence ≥ 0.90.
  8. test      (HF)         → 6 new test cells: {AA+PC2, AA+PC2+PC3} ×
                              {none, adv_null, pc3_attack} on the 508-prompt
                              test subset. (AA-only × * comes from Phase B
                              rollouts; reuse_phase_b_aa_rollouts=true.)
  9. judge_t   (vLLM judge) → judge test rollouts.
 10. assemble  (CPU)        → multi_axis_calibration.json + test_split.parquet
                              + headline.json.

Usage:
  uv run python -m src.experiments.phase_d \
      --config configs/phase_d_gemma_4_31b_thinking_off.yaml
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import yaml


def _log(msg: str) -> None:
    print(f"[phase_d {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _mark_done(marker: Path, payload: dict) -> None:
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps(payload, indent=2, default=str))


# ============================================================================
# Step 1 — setup: load Phase B artifacts, compute PC2/PC3 τ, sample validation
# ============================================================================


def step_1_setup(cfg: dict, out_dir: Path) -> dict:
    import numpy as np
    import pandas as pd
    import torch
    from safetensors.torch import load_file, save_file

    from src.extraction.types import ActivationCache

    marker = out_dir / ".step1.done"
    if marker.exists():
        _log("step 1: skipped (marker exists)")
        return json.loads(marker.read_text())

    phase_b_dir = Path(cfg["phase_b_dir"])
    phase_b_setup = json.loads((phase_b_dir / ".step1.done").read_text())
    capping_layers: list[int] = list(phase_b_setup["capping_layers"])
    l_star = int(phase_b_setup["l_star"])
    lmsys_norm = float(phase_b_setup["lmsys_norm"])
    _log(
        f"step 1: phase_b setup loaded. L*={l_star}  cap_layers={capping_layers}  lmsys_norm={lmsys_norm:.3f}"
    )

    # 1.1 Load PC2/PC3 sign-matched units (already at L*=14 unit, scaled by lmsys_norm).
    pc2_path = phase_b_setup["axis_files"]["signmatched_pc2"]
    pc3_path = phase_b_setup["axis_files"]["signmatched_pc3"]
    pc2_scaled = load_file(pc2_path)["vector"].float()  # (d_model,)
    pc3_scaled = load_file(pc3_path)["vector"].float()
    pc2_unit = (pc2_scaled / max(pc2_scaled.norm().item(), 1e-9)).numpy()
    pc3_unit = (pc3_scaled / max(pc3_scaled.norm().item(), 1e-9)).numpy()
    _log(
        f"step 1: PC2 unit norm={np.linalg.norm(pc2_unit):.3e}  "
        f"PC3 unit norm={np.linalg.norm(pc3_unit):.3e}"
    )

    # 1.2 τ calibration for PC2/PC3 — project role rollouts at L*=14 onto the
    #     **harm-negative** PC unit (= -signmatched_pc). PC2/PC3 are extracted
    #     at L*=14 only, so the cap is applied there too (NOT at the AA
    #     capping layer range 27-34). Reusing the L*=14 unit at later layers
    #     in the initial Phase D attempt produced 100% nonsense at all 5
    #     percentiles because the PC2 unit has large but ill-aligned
    #     projection magnitudes at layers 27-34, which led to ~177-magnitude
    #     subtractions × 8 layers ≈ catastrophic clamping. Single-layer cap at
    #     L*=14 keeps the cap action geometrically meaningful: it operates on
    #     the same activations the PCA was fit on, and at the same layer
    #     where the PC3 attack injects.
    pc_cap_layer = l_star
    pc_tau_per_layer: dict[str, dict[int, dict[str, float]]] = {
        "signmatched_pc2": {},
        "signmatched_pc3": {},
    }
    cache_l_star = ActivationCache.load(
        ActivationCache.cache_path(cfg["model_id"], "plan_b_role_rollouts", pc_cap_layer, "data/cache")
    )
    acts_l_star = cache_l_star.tensor.float().numpy()  # (n, d_model)
    for axis_name, unit in (("signmatched_pc2", pc2_unit), ("signmatched_pc3", pc3_unit)):
        projs_input = -(acts_l_star @ unit)  # role projection onto -signmatched_pc
        pc_tau_per_layer[axis_name][pc_cap_layer] = {
            "p1": float(np.quantile(projs_input, 0.01)),
            "p10": float(np.quantile(projs_input, 0.10)),
            "p25": float(np.quantile(projs_input, 0.25)),
            "p50": float(np.quantile(projs_input, 0.50)),
            "p75": float(np.quantile(projs_input, 0.75)),
            "n_samples": int(projs_input.size),
            "mean": float(projs_input.mean()),
        }
    pc_tau_path = out_dir / "extraction" / "tau_calibration_pc.json"
    pc_tau_path.parent.mkdir(parents=True, exist_ok=True)
    pc_tau_path.write_text(
        json.dumps(
            {"pc_cap_layer": pc_cap_layer, "per_axis": pc_tau_per_layer},
            indent=2,
        )
    )
    _log(
        f"step 1: PC tau calibrated at L*={pc_cap_layer} only. "
        f"PC2 p25={pc_tau_per_layer['signmatched_pc2'][pc_cap_layer]['p25']:+.3f}  "
        f"PC3 p25={pc_tau_per_layer['signmatched_pc3'][pc_cap_layer]['p25']:+.3f}"
    )

    # 1.3 Build single-layer cap vector files for PC2 and PC3 at L*=14 only.
    cap_vec_dir = out_dir / "extraction" / "cap_vectors"
    cap_vec_dir.mkdir(parents=True, exist_ok=True)

    def _save_cap_vec(name: str, layer: int, unit: np.ndarray) -> str:
        # NEGATE: cap vector input must be in the "good" direction (harm-negative
        # for PC2/PC3) so the steerer's auto-negation flips it to harm-positive
        # at the upstream cap. Same convention as Phase B's AA, which passes
        # mean(default)-mean(role) (Assistant-positive = "good direction").
        v = torch.from_numpy(-unit * lmsys_norm).bfloat16().contiguous()
        p = cap_vec_dir / f"{name}_L{layer}.safetensors"
        # Phase B saved cap vectors with key "vector"; match here.
        save_file({"vector": v}, str(p))
        return str(p)

    pc_cap_files: dict[str, dict[int, str]] = {"signmatched_pc2": {}, "signmatched_pc3": {}}
    for axis_name, unit in (("signmatched_pc2", pc2_unit), ("signmatched_pc3", pc3_unit)):
        pc_cap_files[axis_name][pc_cap_layer] = _save_cap_vec(axis_name, pc_cap_layer, unit)
    _log(f"step 1: built single-layer (L*={pc_cap_layer}) cap vector files for {list(pc_cap_files.keys())}")

    # 1.4 Build validation subset: 200 stratified prompts disjoint from
    #     Phase B's 508-prompt test subset.
    pool = pd.read_parquet(cfg["prompt_pool_path"])
    excl_path = Path(cfg["exclude_prompts_from"])
    excl = pd.read_parquet(excl_path)
    excl_ids = set(excl["prompt_id"].astype(str).tolist())
    candidates = pool[~pool["prompt_id"].astype(str).isin(excl_ids)].copy()
    if "category" not in candidates.columns:
        candidates["category"] = "Unknown"
    n_val = int(cfg["n_val_prompts"])
    n_cats = max(1, candidates["category"].nunique())
    per_cat = max(1, n_val // n_cats)
    val = (
        candidates.groupby("category", group_keys=False)
        .apply(lambda g: g.sample(n=min(per_cat, len(g)), random_state=cfg["seed"]))
        .reset_index(drop=True)
    )
    if len(val) < n_val:
        rest = candidates.drop(val.index, errors="ignore").sample(
            n=n_val - len(val), random_state=cfg["seed"]
        )
        val = pd.concat([val, rest]).reset_index(drop=True)
    val = val.head(n_val).copy()
    val["dataset"] = "dan"
    if "input_text" not in val.columns and "full_prompt" in val.columns:
        val["input_text"] = val["full_prompt"]
    val_path = out_dir / "rollouts" / "_val_subset.parquet"
    val_path.parent.mkdir(parents=True, exist_ok=True)
    val.to_parquet(val_path, index=False)
    _log(
        f"step 1: validation subset n={len(val)} (excluded {len(excl_ids)} test prompts from pool)"
    )

    # 1.5 Test subset = Phase B's existing 508-prompt subset.
    test_path = out_dir / "rollouts" / "_test_subset.parquet"
    if not test_path.exists():
        excl.to_parquet(test_path, index=False)
    _log(f"step 1: test subset n={len(excl)} (reuses phase_b/.../rollouts/_full_subset.parquet)")

    payload = {
        "phase_b_dir": str(phase_b_dir),
        "capping_layers": capping_layers,           # AA capping range (8 layers)
        "pc_cap_layer": pc_cap_layer,               # PC2/PC3 cap layer = L*=14 (single)
        "l_star": l_star,
        "lmsys_norm": lmsys_norm,
        "aa_cap_files": phase_b_setup["aa_cap_files"],
        "aa_cap_thresholds": phase_b_setup["cap_thresholds"],
        "axis_files_phase_b": phase_b_setup["axis_files"],
        "pc_cap_files": pc_cap_files,
        "pc_tau_per_layer": pc_tau_per_layer,
        "val_path": str(val_path),
        "test_path": str(test_path),
    }
    _mark_done(marker, payload)
    return payload


# ============================================================================
# Multi-axis steering builders
# ============================================================================


def _normalize_setup(setup: dict) -> dict:
    """JSON-loaded setup has int keys turned into strings; normalize back."""
    fixed = dict(setup)
    fixed["aa_cap_files"] = list(setup["aa_cap_files"])
    # pc_cap_files: {axis: {layer: path}}  (single-layer for PC2/PC3 = L*=14)
    pc_cap = {}
    for ax, by_layer in setup["pc_cap_files"].items():
        pc_cap[ax] = {int(k): v for k, v in by_layer.items()}
    fixed["pc_cap_files"] = pc_cap
    pc_tau = {}
    for ax, by_layer in setup["pc_tau_per_layer"].items():
        pc_tau[ax] = {int(k): v for k, v in by_layer.items()}
    fixed["pc_tau_per_layer"] = pc_tau
    fixed["capping_layers"] = [int(L) for L in setup["capping_layers"]]
    fixed["pc_cap_layer"] = int(setup["pc_cap_layer"])
    return fixed


def _resolve_pc_tau(setup: dict, axis_name: str, percentile: int) -> tuple[float, int]:
    """Cap threshold + layer for one PC axis at a given percentile.

    PC2/PC3 are capped at a single layer (L*=14, the extraction layer where
    PCs are geometrically meaningful). Convention matches Phase B's AA exactly:
    cap_threshold = -p<percentile> where p<X> is the X-th percentile of
    role-rollout projection on the cap-vector input direction (harm-negative =
    -signmatched_pc unit, the "good direction" analog to Phase B's
    mean(default)-mean(role) AA input). Lower percentile = more aggressive cap
    (cap fires for a larger fraction of activations because τ moves further
    toward the role-distribution mean).
    """
    L = int(setup["pc_cap_layer"])
    table = setup["pc_tau_per_layer"][axis_name]
    return -float(table[L][f"p{percentile}"]), L


def _build_multi_axis_cap_steering(
    setup: dict,
    defence_axes: list[str],  # subset of {"aa", "signmatched_pc2", "signmatched_pc3"}
    pc_taus: dict[str, int],  # {axis: percentile}
    *,
    attack_axis: str | None = None,
    attack_lambda: float = 0.0,
) -> dict:
    """Build a steering dict for multi-axis cap (+ optional attack addition).

    cap_vectors / cap_thresholds / cap_layers are flat lists where each tuple
    (vector_i, threshold_i, layer_i) is one cap operation. The upstream
    ActivationSteering(capping) iterates them in registration order. Phase B
    used this exact pattern for AA — we extend it to additional PC axes.
    """
    cap_vectors: list[str] = []
    cap_thresholds: list[float] = []
    cap_layers: list[int] = []
    for axis in defence_axes:
        if axis == "aa":
            # AA cap: 8-layer range matching Phase B exactly.
            for path, tau, L in zip(
                setup["aa_cap_files"],
                setup["aa_cap_thresholds"],
                setup["capping_layers"],
                strict=True,
            ):
                cap_vectors.append(path)
                cap_thresholds.append(float(tau))
                cap_layers.append(int(L))
        else:
            # PC2/PC3 cap: single layer (L*=14, the PCA extraction layer).
            if axis not in pc_taus:
                raise KeyError(f"missing percentile pick for {axis}")
            tau, L = _resolve_pc_tau(setup, axis, pc_taus[axis])
            cap_vectors.append(setup["pc_cap_files"][axis][L])
            cap_thresholds.append(float(tau))
            cap_layers.append(int(L))

    if attack_axis is None:
        return {
            "mode": "capping",
            "vectors": cap_vectors,
            "coefficients": [1.0] * len(cap_vectors),
            "cap_thresholds": cap_thresholds,
            "layers": cap_layers,
            "positions": "all",
        }

    return {
        "mode": "compound",
        "cap_vectors": cap_vectors,
        "cap_thresholds": cap_thresholds,
        "cap_layers": cap_layers,
        "addition_vectors": [setup["axis_files_phase_b"][attack_axis]],
        "addition_coefficients": [float(attack_lambda)],
        "addition_layers": [int(setup["l_star"])],
        "positions": "all",
    }


# ============================================================================
# Step 2 — validation sweep for PC2 τ
# ============================================================================


def _run_steered_cell(
    cfg: dict,
    setup: dict,
    *,
    cond_id: str,
    out_path: Path,
    work_path: Path,
    prompts_path: str,
    steering: dict,
    timeout_seconds: int = 7200,
) -> dict:
    from src.utils.config import resolved_steered_backend
    from src.utils.model_runner import run_in_subprocess

    backend = resolved_steered_backend(cfg["model_id"])
    res = run_in_subprocess(
        "src.evaluation.run_subject_rollouts",
        {
            "model_id": cfg["model_id"],
            "backend": backend,
            "prompts_path": prompts_path,
            "output_path": str(out_path),
            "condition_id": cond_id,
            "seed": cfg["seed"],
            "max_new_tokens": cfg["max_new_tokens"],
            "temperature": 0.0,
            "batch_size": cfg["hf_steered_batch_size"],
            "max_input_len": cfg.get("hf_steered_max_input_len", 4096),
            "steering": steering,
        },
        output_path=work_path,
        timeout_seconds=timeout_seconds,
    )
    return res


def step_2_validate_pc2(cfg: dict, out_dir: Path, setup: dict) -> list[Path]:
    marker = out_dir / ".step2.done"
    rollouts_dir = out_dir / "rollouts" / "val_pc2"
    rollouts_dir.mkdir(parents=True, exist_ok=True)
    if marker.exists():
        _log("step 2: skipped (marker exists)")
        return list(rollouts_dir.glob("*.parquet"))

    atk = cfg["calibration_attack"]
    out_paths: list[Path] = []
    for pct in cfg["tau_percentile_candidates"]:
        cond_id = f"val_aa_pc2p{pct}_atk_{atk['axis']}"
        cond_path = rollouts_dir / f"{cond_id}.parquet"
        if cond_path.exists():
            _log(f"step 2: skip {cond_id} (exists)")
            out_paths.append(cond_path)
            continue
        steering = _build_multi_axis_cap_steering(
            setup,
            defence_axes=["aa", "signmatched_pc2"],
            pc_taus={"signmatched_pc2": int(pct)},
            attack_axis=atk["axis"],
            attack_lambda=float(atk["lambda"]),
        )
        _log(f"step 2: {cond_id}")
        res = _run_steered_cell(
            cfg,
            setup,
            cond_id=cond_id,
            out_path=cond_path,
            work_path=out_dir / f".step2_{cond_id}.work.json",
            prompts_path=setup["val_path"],
            steering=steering,
            timeout_seconds=7200,
        )
        _log(f"  → {res['n_rows']} rows in {res['elapsed_seconds']}s")
        out_paths.append(cond_path)

    _mark_done(marker, {"n_cells": len(out_paths)})
    return out_paths


# ============================================================================
# Step 3 — judge val_pc2 rollouts
# ============================================================================


def _judge_python_executable() -> str | None:
    import sys

    from src.utils.config import REPO_ROOT

    venv_py = REPO_ROOT / ".venv" / "bin" / "python"
    if ".venv-sglang" in sys.executable and venv_py.exists():
        return str(venv_py)
    return None


def _judge_rollouts(
    cfg: dict,
    out_dir: Path,
    paths: list[Path],
    *,
    name: str,
) -> Path:
    import pandas as pd

    from src.utils.model_runner import run_in_subprocess

    marker = out_dir / f".judge_{name}.done"
    judged_path = out_dir / "rollouts" / f"{name}_judged.parquet"
    if marker.exists():
        _log(f"judge[{name}]: skipped (marker exists)")
        return judged_path

    dfs = []
    for p in paths:
        d = pd.read_parquet(p)
        if "category" not in d.columns:
            d["category"] = "Unknown"
        else:
            d["category"] = d["category"].fillna("Unknown")
        if "question_text" not in d.columns:
            d["question_text"] = d.get("input_text", "")
        dfs.append(d)
    all_df = pd.concat(dfs, ignore_index=True)
    all_path = out_dir / "rollouts" / f"{name}_all.parquet"
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
        output_path=out_dir / f".judge_{name}.work.json",
        timeout_seconds=10800,
        python_executable=_judge_python_executable(),
    )
    _log(f"judge[{name}]: {res['n_parsed']}/{res['n_rows']} parsed in {res['elapsed_seconds']}s")
    _mark_done(marker, res)
    return judged_path


def step_3_judge_val_pc2(cfg: dict, out_dir: Path, paths: list[Path]) -> Path:
    return _judge_rollouts(cfg, out_dir, paths, name="val_pc2")


# ============================================================================
# Step 4 — pick τ_PC2 (argmin harm s.t. coherence ≥ floor)
# ============================================================================


def _pick_tau(
    judged_path: Path,
    cfg: dict,
    *,
    cond_prefix: str,
    axis: str,
) -> tuple[int, dict]:
    import pandas as pd

    from src.evaluation.safety import binarize_harm

    df = pd.read_parquet(judged_path)
    df["harm_binary"] = df["harm_label_primary"].map(binarize_harm)
    df["nonsense"] = (df["harm_label_primary"] == "nonsensical").astype(int)
    df = df[df["condition_id"].str.startswith(cond_prefix)].copy()

    floor = float(cfg["coherence_floor"])
    rows: list[dict] = []
    for cond, sub in df.groupby("condition_id"):
        # cond format: "val_aa_pc2p25_atk_..." → percentile = "25"
        suffix = cond[len(cond_prefix) :]
        # suffix is e.g. "p25_atk_..."  → int 25
        if not suffix.startswith("p"):
            continue
        pct_str = suffix[1:].split("_")[0]
        try:
            pct = int(pct_str)
        except ValueError:
            continue
        rows.append(
            {
                "percentile": pct,
                "n": int(len(sub)),
                "harm": float(sub["harm_binary"].mean()),
                "coherence": float(1.0 - sub["nonsense"].mean()),
            }
        )
    if not rows:
        raise RuntimeError(f"no rows matched cond_prefix={cond_prefix!r} in {judged_path}")
    table = pd.DataFrame(rows).sort_values("percentile").reset_index(drop=True)
    valid = table[table["coherence"] >= floor]
    chosen = (
        valid.iloc[valid["harm"].argmin()] if len(valid) else table.iloc[table["harm"].argmin()]
    )
    return int(chosen["percentile"]), {
        "axis": axis,
        "table": table.to_dict(orient="records"),
        "chosen_percentile": int(chosen["percentile"]),
        "chosen_harm": float(chosen["harm"]),
        "chosen_coherence": float(chosen["coherence"]),
        "coherence_floor_met": bool(len(valid) > 0),
    }


def step_4_pick_tau_pc2(cfg: dict, out_dir: Path, judged_path: Path) -> dict:
    marker = out_dir / ".step4.done"
    if marker.exists():
        return json.loads(marker.read_text())
    pct, summary = _pick_tau(
        judged_path,
        cfg,
        cond_prefix="val_aa_pc2",
        axis="signmatched_pc2",
    )
    _log(
        f"step 4: τ_PC2 = p{pct}  (harm={summary['chosen_harm']:.3f}  "
        f"coherence={summary['chosen_coherence']:.3f}  "
        f"floor_met={summary['coherence_floor_met']})"
    )
    _mark_done(marker, summary)
    return summary


# ============================================================================
# Step 5 — validation sweep for PC3 τ (with PC2 fixed at chosen percentile)
# ============================================================================


def step_5_validate_pc3(cfg: dict, out_dir: Path, setup: dict, pc2_pick: dict) -> list[Path]:
    marker = out_dir / ".step5.done"
    rollouts_dir = out_dir / "rollouts" / "val_pc3"
    rollouts_dir.mkdir(parents=True, exist_ok=True)
    if marker.exists():
        _log("step 5: skipped (marker exists)")
        return list(rollouts_dir.glob("*.parquet"))

    atk = cfg["calibration_attack"]
    pc2_pct = int(pc2_pick["chosen_percentile"])
    out_paths: list[Path] = []
    for pct in cfg["tau_percentile_candidates"]:
        cond_id = f"val_aa_pc2p{pc2_pct}_pc3p{pct}_atk_{atk['axis']}"
        cond_path = rollouts_dir / f"{cond_id}.parquet"
        if cond_path.exists():
            _log(f"step 5: skip {cond_id} (exists)")
            out_paths.append(cond_path)
            continue
        steering = _build_multi_axis_cap_steering(
            setup,
            defence_axes=["aa", "signmatched_pc2", "signmatched_pc3"],
            pc_taus={"signmatched_pc2": pc2_pct, "signmatched_pc3": int(pct)},
            attack_axis=atk["axis"],
            attack_lambda=float(atk["lambda"]),
        )
        _log(f"step 5: {cond_id}")
        res = _run_steered_cell(
            cfg,
            setup,
            cond_id=cond_id,
            out_path=cond_path,
            work_path=out_dir / f".step5_{cond_id}.work.json",
            prompts_path=setup["val_path"],
            steering=steering,
            timeout_seconds=7200,
        )
        _log(f"  → {res['n_rows']} rows in {res['elapsed_seconds']}s")
        out_paths.append(cond_path)

    _mark_done(marker, {"n_cells": len(out_paths), "pc2_percentile": pc2_pct})
    return out_paths


def step_6_judge_val_pc3(cfg: dict, out_dir: Path, paths: list[Path]) -> Path:
    return _judge_rollouts(cfg, out_dir, paths, name="val_pc3")


def step_7_pick_tau_pc3(cfg: dict, out_dir: Path, judged_path: Path, pc2_pick: dict) -> dict:
    marker = out_dir / ".step7.done"
    if marker.exists():
        return json.loads(marker.read_text())
    pc2_pct = int(pc2_pick["chosen_percentile"])
    cond_prefix = f"val_aa_pc2p{pc2_pct}_pc3"
    pct, summary = _pick_tau(
        judged_path,
        cfg,
        cond_prefix=cond_prefix,
        axis="signmatched_pc3",
    )
    _log(
        f"step 7: τ_PC3 = p{pct}  (harm={summary['chosen_harm']:.3f}  "
        f"coherence={summary['chosen_coherence']:.3f}  "
        f"floor_met={summary['coherence_floor_met']})"
    )
    summary["pc2_percentile_locked"] = pc2_pct
    _mark_done(marker, summary)
    return summary


# ============================================================================
# Step 8 — test split: 6 new defence × attack cells (AA-only rows reused from Phase B)
# ============================================================================


def step_8_test(
    cfg: dict,
    out_dir: Path,
    setup: dict,
    pc2_pick: dict,
    pc3_pick: dict,
) -> list[Path]:
    marker = out_dir / ".step8.done"
    rollouts_dir = out_dir / "rollouts" / "test"
    rollouts_dir.mkdir(parents=True, exist_ok=True)
    if marker.exists():
        _log("step 8: skipped (marker exists)")
        return list(rollouts_dir.glob("*.parquet"))

    pc2_pct = int(pc2_pick["chosen_percentile"])
    pc3_pct = int(pc3_pick["chosen_percentile"])
    test_path = setup["test_path"]
    out_paths: list[Path] = []

    test_defences = cfg["test_defences"]
    test_attacks = cfg["test_attacks"]
    n_total = len(test_defences) * len(test_attacks)
    cell_idx = 0
    for defence in test_defences:
        for attack in test_attacks:
            cell_idx += 1
            d_name = defence["name"]
            a_name = attack["name"]
            cond_id = f"test_def_{d_name}_atk_{a_name}"
            # Skip AA-only (reuse Phase B rollouts at assemble step)
            if cfg.get("reuse_phase_b_aa_rollouts", True) and d_name == "aa_only":
                _log(f"step 8 [{cell_idx}/{n_total}] {cond_id} → reuse Phase B")
                continue

            cond_path = rollouts_dir / f"{cond_id}.parquet"
            if cond_path.exists():
                _log(f"step 8 [{cell_idx}/{n_total}] skip {cond_id} (exists)")
                out_paths.append(cond_path)
                continue

            pc_taus: dict[str, int] = {}
            if "signmatched_pc2" in defence["axes"]:
                pc_taus["signmatched_pc2"] = pc2_pct
            if "signmatched_pc3" in defence["axes"]:
                pc_taus["signmatched_pc3"] = pc3_pct

            steering = _build_multi_axis_cap_steering(
                setup,
                defence_axes=list(defence["axes"]),
                pc_taus=pc_taus,
                attack_axis=attack["axis"],
                attack_lambda=float(attack["lambda"]),
            )
            _log(f"step 8 [{cell_idx}/{n_total}] {cond_id}")
            res = _run_steered_cell(
                cfg,
                setup,
                cond_id=cond_id,
                out_path=cond_path,
                work_path=out_dir / f".step8_{cond_id}.work.json",
                prompts_path=test_path,
                steering=steering,
                timeout_seconds=10800,
            )
            _log(f"  → {res['n_rows']} rows in {res['elapsed_seconds']}s")
            out_paths.append(cond_path)

    _mark_done(marker, {"n_new_cells": len(out_paths)})
    return out_paths


def step_9_judge_test(cfg: dict, out_dir: Path, paths: list[Path]) -> Path:
    return _judge_rollouts(cfg, out_dir, paths, name="test")


# ============================================================================
# Step 10 — assemble multi_axis_calibration.json + test_split.parquet + headline
# ============================================================================


def step_10_assemble(
    cfg: dict,
    out_dir: Path,
    setup: dict,
    pc2_pick: dict,
    pc3_pick: dict,
    test_judged: Path,
) -> Path:
    import pandas as pd

    from src.evaluation.safety import binarize_harm

    marker = out_dir / ".step10.done"
    out_path = out_dir / "test_split.parquet"
    if marker.exists():
        _log("step 10: skipped (marker exists)")
        return out_path

    df_new = pd.read_parquet(test_judged)
    df_new["harm_binary"] = df_new["harm_label_primary"].map(binarize_harm)
    df_new["nonsense"] = (df_new["harm_label_primary"] == "nonsensical").astype(int)

    # Pull AA-only rollouts from Phase B (reused) and align them to the
    # test-split condition naming so the matrix has all 9 cells.
    phase_b_dir = Path(setup["phase_b_dir"])
    phase_b_full_judged = phase_b_dir / "rollouts" / "full_judged.parquet"
    if not phase_b_full_judged.exists():
        raise RuntimeError(f"missing Phase B full_judged: {phase_b_full_judged}")
    df_pb = pd.read_parquet(phase_b_full_judged)
    df_pb["harm_binary"] = df_pb["harm_label_primary"].map(binarize_harm)
    df_pb["nonsense"] = (df_pb["harm_label_primary"] == "nonsensical").astype(int)

    aa_attack_map = {
        "zero": "full_aa_capped_only",
        "adv_null_pos0p25": "full_aa_capped_adv_null_pos0p25",
        "signmatched_pc3_pos0p25": "full_aa_capped_signmatched_pc3_pos0p25",
    }
    aa_rows = []
    for atk_name, pb_cond in aa_attack_map.items():
        sub = df_pb[df_pb["condition_id"] == pb_cond].copy()
        if sub.empty:
            _log(f"step 10: WARNING — Phase B condition '{pb_cond}' missing")
            continue
        sub["condition_id"] = f"test_def_aa_only_atk_{atk_name}"
        aa_rows.append(sub)
    df_all = pd.concat([df_new] + aa_rows, ignore_index=True) if aa_rows else df_new
    df_all.to_parquet(out_path, index=False)

    # Parse condition_id → (defence, attack)
    def _parse(cid: str) -> tuple[str, str]:
        if not cid.startswith("test_def_"):
            return ("?", "?")
        rest = cid[len("test_def_") :]
        i = rest.find("_atk_")
        if i < 0:
            return (rest, "?")
        return rest[:i], rest[i + len("_atk_") :]

    df_all["defence"], df_all["attack"] = zip(*df_all["condition_id"].map(_parse), strict=True)

    summary = (
        df_all.groupby(["defence", "attack"])
        .agg(
            n=("harm_binary", "size"),
            harm=("harm_binary", "mean"),
            nonsense=("nonsense", "mean"),
        )
        .round(4)
    )
    summary["coherence"] = (1.0 - summary["nonsense"]).round(4)
    summary_csv = out_dir / "test_split_summary.csv"
    summary.to_csv(summary_csv)
    _log(f"step 10: test_split summary →\n{summary.to_string()}")

    # Multi-axis calibration record
    multi_calib = {
        "subject": cfg["subject_id"],
        "calibration_attack": cfg["calibration_attack"],
        "coherence_floor": cfg["coherence_floor"],
        "pc2": pc2_pick,
        "pc3": pc3_pick,
        "aa_tau_percentile_locked": int(cfg["aa_tau_percentile"]),
    }
    (out_dir / "multi_axis_calibration.json").write_text(json.dumps(multi_calib, indent=2))

    # Headline
    def _cell(d: str, a: str) -> dict[str, Any]:
        sub = df_all[(df_all["defence"] == d) & (df_all["attack"] == a)]
        return {
            "n": int(len(sub)),
            "harm_rate": float(sub["harm_binary"].mean()) if len(sub) else None,
            "coherence_rate": float(1.0 - sub["nonsense"].mean()) if len(sub) else None,
        }

    headline = {
        "subject": cfg["subject_id"],
        "test_split_n": int(len(df_all[df_all["defence"] == "aa_only"]) // 3 if len(df_all) else 0),
        "tau_pc2_percentile": int(pc2_pick["chosen_percentile"]),
        "tau_pc3_percentile": int(pc3_pick["chosen_percentile"]),
        "matrix": {
            d: {a: _cell(d, a) for a in ("zero", "adv_null_pos0p25", "signmatched_pc3_pos0p25")}
            for d in ("aa_only", "aa_pc2", "aa_pc2_pc3")
        },
    }
    (out_dir / "headline.json").write_text(json.dumps(headline, indent=2))
    _log(f"step 10: headline →\n{json.dumps(headline, indent=2)}")

    _mark_done(marker, {"n_rows": len(df_all), "summary_csv": str(summary_csv)})
    return out_path


# ============================================================================
# Main
# ============================================================================


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config", type=Path, default=Path("configs/phase_d_gemma_4_31b_thinking_off.yaml")
    )
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
        setup_raw = step_1_setup(cfg, out_dir)
    else:
        setup_raw = json.loads((out_dir / ".step1.done").read_text())
    setup = _normalize_setup(setup_raw)

    if skip < 2:
        val_pc2_paths = step_2_validate_pc2(cfg, out_dir, setup)
    else:
        val_pc2_paths = list((out_dir / "rollouts" / "val_pc2").glob("*.parquet"))

    if skip < 3:
        val_pc2_judged = step_3_judge_val_pc2(cfg, out_dir, val_pc2_paths)
    else:
        val_pc2_judged = out_dir / "rollouts" / "val_pc2_judged.parquet"

    if skip < 4:
        pc2_pick = step_4_pick_tau_pc2(cfg, out_dir, val_pc2_judged)
    else:
        pc2_pick = json.loads((out_dir / ".step4.done").read_text())

    if skip < 5:
        val_pc3_paths = step_5_validate_pc3(cfg, out_dir, setup, pc2_pick)
    else:
        val_pc3_paths = list((out_dir / "rollouts" / "val_pc3").glob("*.parquet"))

    if skip < 6:
        val_pc3_judged = step_6_judge_val_pc3(cfg, out_dir, val_pc3_paths)
    else:
        val_pc3_judged = out_dir / "rollouts" / "val_pc3_judged.parquet"

    if skip < 7:
        pc3_pick = step_7_pick_tau_pc3(cfg, out_dir, val_pc3_judged, pc2_pick)
    else:
        pc3_pick = json.loads((out_dir / ".step7.done").read_text())

    if skip < 8:
        test_paths = step_8_test(cfg, out_dir, setup, pc2_pick, pc3_pick)
    else:
        test_paths = list((out_dir / "rollouts" / "test").glob("*.parquet"))

    if skip < 9:
        test_judged = step_9_judge_test(cfg, out_dir, test_paths)
    else:
        test_judged = out_dir / "rollouts" / "test_judged.parquet"

    if skip < 10:
        out_path = step_10_assemble(cfg, out_dir, setup, pc2_pick, pc3_pick, test_judged)
    else:
        out_path = out_dir / "test_split.parquet"

    _log(f"PHASE D COMPLETE [{cfg['subject_id']}] → {out_path}")


if __name__ == "__main__":
    main()
