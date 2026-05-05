"""Phase B — Attack arm orchestrator (Stage 4 + Ext E/F), per-subject.

Reads Phase A artifacts under `results/phase_a/<subject>/extraction/`:
  - aa.safetensors (full per-layer AA + aa_at_lstar)
  - pcs.safetensors (top-10 PCs at L*, plus PCA mean)
  - L_star.txt
  - tau_calibration.json (per-layer p25 of role-positive AA projections)
  - lmsys_norms_L<L*>.json (mean residual norm at L*)
And the Phase A baseline rollouts:
  - results/phase_a/<subject>/details.parquet (with harm_binary)
  - results/phase_a/<subject>/rollouts/_dan_500.parquet (the 500 stratified DAN prompts)
  - data/cache/activations/<subject>/plan_b_per_prompt_L<L*>/L<L*>.safetensors

Pipeline (parent never imports torch/vllm; all heavy work via subprocess):
  1. setup     (CPU)        → build subset, compute v_harm, build attack vectors,
                              build per-layer AA cap vectors, write harm_direction.json
  2. mini      (HF/SGLang)  → mini-runs at all (axis × λ) cells on stratified mini subset
  3. judge_m   (vLLM judge) → judge mini rollouts
  4. pick_lam  (CPU)        → per-axis λ_max_coherent from mini judge results
  5. full      (HF/SGLang)  → full runs at λ_max_coherent for each axis + multi-axis composite
                              + AA-cap-only baseline (paper reproduction)
  6. judge_f   (vLLM judge) → judge full rollouts
  7. assemble  (CPU)        → lambda_sweep.parquet + headline.json (harm % per condition)

Steered backend resolved per `configs/subjects.yaml::<id>.steered_backend`
(SGLang for Qwen 3 32B; HF for Gemma 4 31B modes).

Usage:
  uv run python -m src.experiments.phase_b --subject qwen_3_32b
  uv run python -m src.experiments.phase_b --config configs/phase_b_qwen_3_32b.yaml
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import yaml


def _log(msg: str) -> None:
    print(f"[phase_b {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _mark_done(marker: Path, payload: dict) -> None:
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps(payload, indent=2, default=str))


def _lam_token(lam: float) -> str:
    """λ=0.25 → "pos0p25", λ=-0.10 → "neg0p1"."""
    sign = "pos" if lam >= 0 else "neg"
    mag = abs(float(lam))
    if mag == int(mag):
        return f"{sign}{int(mag)}"
    return f"{sign}{mag:g}".replace(".", "p")


def _lam_token_to_float(tok: str) -> float:
    sign = 1.0 if tok.startswith("pos") else -1.0
    rest = tok[3:].replace("p", ".")
    return sign * float(rest)


# ============================================================================
# Step 1 — setup: load Phase A artifacts, compute v_harm, build vectors
# ============================================================================


def step_1_setup(cfg: dict, out_dir: Path) -> dict:
    """CPU: build mini + full subsets; compute v_harm; build attack + cap vectors."""
    import numpy as np
    import pandas as pd
    import torch
    from safetensors.torch import load_file, save_file

    from src.extraction.types import ActivationCache

    marker = out_dir / ".step1.done"
    if marker.exists():
        _log("step 1: skipped (marker exists)")
        return json.loads(marker.read_text())

    subject = cfg["subject_id"]
    phase_a_dir = Path(cfg["phase_a_dir"])
    extr = phase_a_dir / "extraction"

    # 1.1 Load AA + PCs + L* + tau
    aa_full = load_file(str(extr / "aa.safetensors"))["aa"]   # (n_layers, d_model)
    pcs_d = load_file(str(extr / "pcs.safetensors"))
    pcs_at_lstar = pcs_d["pcs_at_lstar"].float()              # (k, d_model)
    pca_mean_at_lstar = pcs_d["pca_mean_at_lstar"].float()    # (d_model,)
    l_star = int((extr / "L_star.txt").read_text().strip())
    tau_meta = json.loads((extr / "tau_calibration.json").read_text())
    capping_layers = sorted(int(k) for k in tau_meta["per_layer"].keys())
    lmsys_meta = json.loads((extr / f"lmsys_norms_L{l_star}.json").read_text())
    lmsys_norm = float(lmsys_meta["mean_norm"])
    _log(f"step 1: subject={subject}  L*={l_star}  cap_layers={capping_layers}  lmsys_norm={lmsys_norm:.3f}")

    # 1.2 Load Phase A baseline parquet + 500 DAN prompts subset
    details = pd.read_parquet(phase_a_dir / "details.parquet")
    base = details[details["condition_id"] == "baseline"].copy()
    if base.empty:
        raise RuntimeError(f"no baseline rows in {phase_a_dir}/details.parquet")
    n_pos = int(base["harm_binary"].sum())
    n_total = len(base)
    _log(f"step 1: Phase A baseline n={n_total}  harm_pos={n_pos} ({n_pos/n_total*100:.1f}%)")

    # Use Phase A's _dan_500.parquet directly as the full subset.
    subset_path_in = phase_a_dir / "rollouts" / "_dan_500.parquet"
    if not subset_path_in.exists():
        raise RuntimeError(f"missing Phase A prompt cache: {subset_path_in}")
    subset = pd.read_parquet(subset_path_in)
    # Attach harm_binary from Phase A judge so we can stratify mini
    subset = subset.merge(
        base[["prompt_id", "harm_binary"]],
        on="prompt_id",
        how="left",
    )
    subset["harm_binary"] = subset["harm_binary"].fillna(0).astype(int)
    if "category" in subset.columns:
        subset["category"] = subset["category"].fillna("Unknown")
    else:
        subset["category"] = "Unknown"
    subset_path = out_dir / "rollouts" / "_full_subset.parquet"
    subset_path.parent.mkdir(parents=True, exist_ok=True)
    subset.to_parquet(subset_path, index=False)

    # 1.3 Mini subset: 50 stratified — half from harm-positive (or all if <25), rest from harm-negative
    mini_n_pos = min(int(cfg.get("mini_n_per_stratum", 25)), int((subset["harm_binary"] == 1).sum()))
    mini_n_neg = min(int(cfg.get("mini_n_per_stratum", 25)), int((subset["harm_binary"] == 0).sum()))
    rng = np.random.default_rng(cfg["seed"])
    mini_pos = subset[subset["harm_binary"] == 1].sample(n=mini_n_pos, random_state=int(rng.integers(0, 2**31)))
    mini_neg = subset[subset["harm_binary"] == 0].sample(n=mini_n_neg, random_state=int(rng.integers(0, 2**31)))
    mini = pd.concat([mini_pos, mini_neg], ignore_index=True)
    mini_path = out_dir / "rollouts" / "_mini_subset.parquet"
    mini.to_parquet(mini_path, index=False)
    _log(f"step 1: mini n={len(mini)} (pos={mini_n_pos} + neg={mini_n_neg})")

    # 1.4 Compute v_harm from cached baseline activations + harm labels
    cache_path = Path(cfg.get(
        "activation_cache_path",
        f"data/cache/activations/{cfg['model_id']}/plan_b_per_prompt_L{l_star}/L{l_star}",
    ))
    cache = ActivationCache.load(cache_path)
    acts = cache.tensor.float().numpy()  # (N, d)
    pid_to_row = {p: i for i, p in enumerate(cache.prompt_ids)}

    aligned_acts, aligned_harm = [], []
    for _, r in base.iterrows():
        key = f"{r['prompt_id']}::baseline"
        if key in pid_to_row:
            aligned_acts.append(acts[pid_to_row[key]])
            aligned_harm.append(int(r["harm_binary"]))
    if not aligned_acts:
        raise RuntimeError("zero baseline rows aligned to activation cache")
    acts_arr = np.stack(aligned_acts, axis=0)
    harm_arr = np.asarray(aligned_harm, dtype=np.int32)
    if int(harm_arr.sum()) == 0 or int((1 - harm_arr).sum()) == 0:
        raise RuntimeError(f"need both classes for v_harm; got n_harm={int(harm_arr.sum())}")

    mean_harm = acts_arr[harm_arr == 1].mean(axis=0)
    mean_safe = acts_arr[harm_arr == 0].mean(axis=0)
    v_harm_raw = mean_harm - mean_safe
    v_harm_norm_pre = float(np.linalg.norm(v_harm_raw))
    v_harm_unit = v_harm_raw / max(v_harm_norm_pre, 1e-9)

    aa_at_lstar = aa_full[l_star].float().numpy()
    aa_unit = aa_at_lstar / max(np.linalg.norm(aa_at_lstar), 1e-9)
    cos_v_harm_aa = float(v_harm_unit @ aa_unit)

    pcs_unit = np.stack([
        (pcs_at_lstar[i].numpy() / max(np.linalg.norm(pcs_at_lstar[i].numpy()), 1e-9))
        for i in range(pcs_at_lstar.shape[0])
    ])
    cos_v_harm_pcs = (pcs_unit @ v_harm_unit).tolist()
    _log(
        f"step 1: v_harm computed (n_harm={int(harm_arr.sum())} n_safe={int((1-harm_arr).sum())}). "
        f"cos(v_harm,AA)={cos_v_harm_aa:+.3f}  cos(v_harm,PC2..3)={cos_v_harm_pcs[1]:+.3f},{cos_v_harm_pcs[2]:+.3f}"
    )

    harm_dir_json = {
        "subject": subject,
        "layer": l_star,
        "n_total": int(len(harm_arr)),
        "n_harm": int(harm_arr.sum()),
        "n_safe": int((1 - harm_arr).sum()),
        "v_harm_norm_pre": v_harm_norm_pre,
        "cos_sim_v_harm_aa": cos_v_harm_aa,
        "cos_sim_v_harm_pcs": cos_v_harm_pcs,
        "lmsys_norm_at_lstar": lmsys_norm,
    }
    (out_dir / "harm_direction.json").write_text(json.dumps(harm_dir_json, indent=2))

    # 1.5 Build attack vectors at L* — already · lmsys_norm so HF addition coefficient = λ
    vec_dir = out_dir / "extraction" / "vectors"
    vec_dir.mkdir(parents=True, exist_ok=True)

    def _save_vec(name: str, v_unit: np.ndarray) -> str:
        # Key "vector" matches sglang_hook_factories._load_vector default; HF
        # backend's _load_vec falls through to first value so this is portable.
        v = torch.from_numpy(v_unit * lmsys_norm).bfloat16().contiguous()
        p = vec_dir / f"{name}.safetensors"
        save_file({"vector": v}, str(p))
        return str(p)

    axis_files: dict[str, str] = {}
    axis_signs: dict[str, int] = {}

    # Per-PC sign-matched: sign_i = sign(cos(v_harm, PC_i)). Steers toward harm-aligned end.
    for pc_idx in cfg["pc_indices_to_attack"]:
        c = cos_v_harm_pcs[pc_idx - 1]
        sign_i = -1 if c < 0 else 1
        v_signmatched = sign_i * pcs_unit[pc_idx - 1]
        name = f"signmatched_pc{pc_idx}"
        axis_files[name] = _save_vec(name, v_signmatched)
        axis_signs[name] = sign_i
        _log(f"step 1: {name} (cos(v_harm,PC{pc_idx})={c:+.3f}, sign={sign_i:+d})")

    # v_harm direct (oracle upper bound)
    if cfg.get("include_v_harm_attack", True):
        axis_files["v_harm"] = _save_vec("v_harm", v_harm_unit)
        axis_signs["v_harm"] = 1

    # Random baselines (Ext E equivalent — same scale as PC attacks)
    rng_r = np.random.default_rng(cfg["seed"])
    for r in range(int(cfg["n_random_baselines"])):
        rv = rng_r.normal(size=v_harm_unit.shape).astype(np.float32)
        rv = rv / max(np.linalg.norm(rv), 1e-9)
        name = f"random_{r}"
        axis_files[name] = _save_vec(name, rv)
        axis_signs[name] = 1

    # 1.6 Build adversarial null-space direction (T4.6) — LASSO-derived AA-orthogonal.
    # Read top-K LASSO selected PCs from Phase A metrics; build u_adv = Σ c_i · PC_i,
    # then project AA out: u_adv -= <u_adv, aa_unit> · aa_unit.
    if cfg.get("include_adversarial_null_space", True):
        metrics = json.loads((phase_a_dir / "metrics.json").read_text())
        bsl = metrics.get("headline", {}).get("blind_spot_auc_lift", {}) or {}
        selected_pcs = bsl.get("selected_pcs", []) or []
        if selected_pcs:
            # Phase A's blind-spot LASSO doesn't expose coefficients; weight all selected
            # equally with sign-match to v_harm. (Stage 4 T4.6 says "use LASSO coefs"; we
            # don't have the raw coefs persisted, so equal-weighted sign-matched is the
            # closest faithful approximation.)
            u_raw = np.zeros_like(v_harm_unit)
            comps = []
            for tag in selected_pcs:
                if not tag.startswith("pc"):
                    continue
                pc_idx = int(tag[2:])
                c = cos_v_harm_pcs[pc_idx - 1]
                s = -1.0 if c < 0 else 1.0
                u_raw = u_raw + s * pcs_unit[pc_idx - 1]
                comps.append((pc_idx, s))
            # Project AA out
            u_proj = u_raw - (u_raw @ aa_unit) * aa_unit
            u_norm = float(np.linalg.norm(u_proj))
            if u_norm > 1e-6:
                u_unit = u_proj / u_norm
                axis_files["adv_null"] = _save_vec("adv_null", u_unit)
                axis_signs["adv_null"] = 1
                _log(f"step 1: adv_null built from {comps}; <u,AA>={float(u_unit @ aa_unit):+.3e} (≈0 by construction)")
        else:
            _log("step 1: skipping adv_null — Phase A LASSO selected 0 PCs")

    # 1.7 Build per-layer AA cap vectors (Phase A skipped plan_b step 4)
    aa_per_layer_files = []
    cap_thresholds = []
    for layer in capping_layers:
        v_layer = aa_full[layer].float()
        v_unit = v_layer / max(v_layer.norm().item(), 1e-9)
        v_scaled = (v_unit * lmsys_norm).bfloat16().contiguous()
        p = vec_dir / f"aa_L{layer}.safetensors"
        save_file({"vector": v_scaled}, str(p))
        aa_per_layer_files.append(str(p))
        # tau in role-positive space = -p75 of +AA (paper's sign-flip math; see
        # plan_b.py:638-645). The upstream cap re-normalizes the vector to unit,
        # so tau is in unit-projection space, matching tau_calibration's scale.
        cap_thresholds.append(-float(tau_meta["per_layer"][str(layer)]["p75"]))
    _log(f"step 1: built {len(aa_per_layer_files)} per-layer AA cap vectors; "
         f"taus={[round(t,2) for t in cap_thresholds]}")

    payload = {
        "subject": subject,
        "subset_path": str(subset_path),
        "mini_path": str(mini_path),
        "n_total": int(len(subset)),
        "n_pos": int(n_pos),
        "axis_files": axis_files,
        "axis_signs": axis_signs,
        "aa_cap_files": aa_per_layer_files,
        "cap_thresholds": cap_thresholds,
        "capping_layers": capping_layers,
        "lmsys_norm": lmsys_norm,
        "l_star": l_star,
        "cos_v_harm_aa": cos_v_harm_aa,
        "cos_v_harm_pcs": cos_v_harm_pcs,
    }
    _mark_done(marker, payload)
    return payload


# ============================================================================
# Step 2 — mini λ sweep
# ============================================================================


def _build_compound_steering(setup: dict, axis_name: str, lam: float) -> dict:
    return {
        "mode": "compound",
        "cap_vectors": setup["aa_cap_files"],
        "cap_thresholds": setup["cap_thresholds"],
        "cap_layers": list(setup["capping_layers"]),
        "addition_vectors": [setup["axis_files"][axis_name]],
        "addition_coefficients": [float(lam)],
        "addition_layers": [int(setup["l_star"])],
        "positions": "all",
    }


def _build_aa_cap_only_steering(setup: dict) -> dict:
    return {
        "mode": "capping",
        "vectors": setup["aa_cap_files"],
        "coefficients": [1.0] * len(setup["aa_cap_files"]),
        "cap_thresholds": setup["cap_thresholds"],
        "layers": list(setup["capping_layers"]),
        "positions": "all",
    }


def step_2_mini_lambda_sweep(cfg: dict, out_dir: Path, setup: dict) -> list[Path]:
    from src.utils.config import resolved_steered_backend
    from src.utils.model_runner import run_in_subprocess

    backend = resolved_steered_backend(cfg["model_id"])
    _log(f"step 2: steered backend = {backend}")

    marker = out_dir / ".step2.done"
    rollouts_dir = out_dir / "rollouts" / "mini"
    rollouts_dir.mkdir(parents=True, exist_ok=True)
    if marker.exists():
        _log("step 2: skipped (marker exists)")
        return list(rollouts_dir.glob("*.parquet"))

    mini_path = Path(setup["mini_path"])

    # Order axes: PCs first, then v_harm, then adv_null, then randoms
    axis_order = []
    for pc_idx in cfg["pc_indices_to_attack"]:
        axis_order.append(f"signmatched_pc{pc_idx}")
    if cfg.get("include_v_harm_attack", True):
        axis_order.append("v_harm")
    if cfg.get("include_adversarial_null_space", True) and "adv_null" in setup["axis_files"]:
        axis_order.append("adv_null")
    for r in range(int(cfg["n_random_baselines"])):
        axis_order.append(f"random_{r}")

    out_paths: list[Path] = []
    total = len(axis_order) * len(cfg["lambda_schedule"])
    cell = 0
    t0 = time.time()
    for axis_name in axis_order:
        for lam in cfg["lambda_schedule"]:
            cell += 1
            cond_id = f"mini_aa_capped_{axis_name}_{_lam_token(lam)}"
            cond_path = rollouts_dir / f"{cond_id}.parquet"
            if cond_path.exists():
                _log(f"step 2 [{cell}/{total}] skip {cond_id} (exists)")
                out_paths.append(cond_path)
                continue
            _log(f"step 2 [{cell}/{total}] {cond_id}")
            steering = _build_compound_steering(setup, axis_name, lam)
            res = run_in_subprocess(
                "src.evaluation.run_subject_rollouts",
                {
                    "model_id": cfg["model_id"],
                    "backend": backend,
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

    _log(f"step 2: {total} mini cells in {time.time() - t0:.0f}s")
    _mark_done(marker, {"n_cells": total, "elapsed": time.time() - t0})
    return out_paths


# ============================================================================
# Step 3 — judge mini rollouts
# ============================================================================


def _judge_python_executable() -> str | None:
    """Judge needs vLLM, which lives in .venv (NOT .venv-sglang). When the
    parent orchestrator runs from .venv-sglang (SGLang subjects), point the
    judge subprocess at .venv/bin/python; otherwise inherit sys.executable.
    """
    import sys
    from src.utils.config import REPO_ROOT
    venv_py = REPO_ROOT / ".venv" / "bin" / "python"
    if ".venv-sglang" in sys.executable and venv_py.exists():
        return str(venv_py)
    return None


def step_3_judge_mini(cfg: dict, out_dir: Path, mini_paths: list[Path]) -> Path:
    import pandas as pd
    from src.utils.model_runner import run_in_subprocess

    marker = out_dir / ".step3.done"
    judged_path = out_dir / "rollouts" / "mini_judged.parquet"
    if marker.exists():
        _log("step 3: skipped (marker exists)")
        return judged_path

    dfs = []
    for p in mini_paths:
        d = pd.read_parquet(p)
        if "category" not in d.columns:
            d["category"] = "Unknown"
        else:
            d["category"] = d["category"].fillna("Unknown")
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
        python_executable=_judge_python_executable(),
    )
    _log(f"step 3: judged {res['n_parsed']}/{res['n_rows']} rows in {res['elapsed_seconds']}s")
    _mark_done(marker, res)
    return judged_path


# ============================================================================
# Step 4 — pick λ_max_coherent per axis
# ============================================================================


def step_4_pick_lambda(cfg: dict, out_dir: Path, judged_path: Path, setup: dict) -> Path:
    import pandas as pd
    from src.evaluation.safety import binarize_harm

    marker = out_dir / ".step4.done"
    pareto_path = out_dir / "lambda_pareto.json"
    if marker.exists():
        _log("step 4: skipped (marker exists)")
        return pareto_path

    df = pd.read_parquet(judged_path)
    df["harm_binary"] = df["harm_label_primary"].map(binarize_harm)
    df["nonsense"] = (df["harm_label_primary"] == "nonsensical").astype(int)

    def _parse(cond_id: str) -> tuple[str, str]:
        rest = cond_id[len("mini_aa_capped_"):]
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
        per_lam = sub.groupby("lam").agg(
            n=("harm_binary", "size"),
            harm=("harm_binary", "mean"),
            nonsense=("nonsense", "mean"),
        ).reset_index().sort_values("lam")
        per_lam["coherence"] = 1.0 - per_lam["nonsense"]
        pareto[axis] = per_lam.to_dict(orient="records")
        valid = per_lam[per_lam["coherence"] >= floor]
        if valid.empty:
            chosen_lam = float(per_lam["lam"].iloc[0])
            _log(f"step 4: axis={axis} — NO λ ≥ floor {floor}; fallback λ={chosen_lam}")
        else:
            chosen_lam = float(valid["lam"].max())
            row = valid[valid["lam"] == chosen_lam].iloc[0]
            _log(f"step 4: axis={axis} — λ_max_coherent={chosen_lam} "
                 f"(harm={row['harm']:.3f}  coherence={row['coherence']:.3f})")
        chosen[axis] = chosen_lam

    pareto_path.write_text(json.dumps({"pareto": pareto, "chosen_lambda_per_axis": chosen}, indent=2))
    _mark_done(marker, {"chosen_lambda_per_axis": chosen})
    return pareto_path


# ============================================================================
# Step 5 — full runs at λ_max_coherent + AA-cap baseline + multi-axis composite
# ============================================================================


def step_5_full_runs(cfg: dict, out_dir: Path, setup: dict, pareto_path: Path) -> list[Path]:
    import torch
    from safetensors.torch import load_file, save_file
    from src.utils.config import resolved_steered_backend
    from src.utils.model_runner import run_in_subprocess

    backend = resolved_steered_backend(cfg["model_id"])
    _log(f"step 5: steered backend = {backend}")

    marker = out_dir / ".step5.done"
    rollouts_dir = out_dir / "rollouts" / "full"
    rollouts_dir.mkdir(parents=True, exist_ok=True)
    if marker.exists():
        _log("step 5: skipped (marker exists)")
        return list(rollouts_dir.glob("*.parquet"))

    chosen = json.loads(pareto_path.read_text())["chosen_lambda_per_axis"]
    subset_path = Path(setup["subset_path"])
    out_paths: list[Path] = []

    # 5a. AA-cap-only baseline (paper reproduction; expected harm reduction ≥ 30 pp)
    cap_only_path = rollouts_dir / "full_aa_capped_only.parquet"
    if cap_only_path.exists():
        _log("step 5: skip full_aa_capped_only (exists)")
        out_paths.append(cap_only_path)
    else:
        _log("step 5: full AA-cap-only baseline")
        res = run_in_subprocess(
            "src.evaluation.run_subject_rollouts",
            {
                "model_id": cfg["model_id"],
                "backend": backend,
                "prompts_path": str(subset_path),
                "output_path": str(cap_only_path),
                "condition_id": "full_aa_capped_only",
                "seed": cfg["seed"],
                "max_new_tokens": cfg["max_new_tokens"],
                "temperature": 0.0,
                "batch_size": cfg["hf_steered_batch_size"],
                "max_input_len": cfg.get("hf_steered_max_input_len", 4096),
                "steering": _build_aa_cap_only_steering(setup),
            },
            output_path=out_dir / ".step5_aa_capped_only.work.json",
            timeout_seconds=10800,
        )
        _log(f"  → {res['n_rows']} rows in {res['elapsed_seconds']}s")
        out_paths.append(cap_only_path)

    # 5b. Per-axis full runs at λ_max_coherent (skip randoms — characterized in mini)
    for axis_name, lam in chosen.items():
        if axis_name.startswith("random_"):
            continue
        cond_id = f"full_aa_capped_{axis_name}_{_lam_token(lam)}"
        cond_path = rollouts_dir / f"{cond_id}.parquet"
        if cond_path.exists():
            _log(f"step 5: skip {cond_id} (exists)")
            out_paths.append(cond_path)
            continue
        _log(f"step 5: full run {cond_id} (λ={lam})")
        res = run_in_subprocess(
            "src.evaluation.run_subject_rollouts",
            {
                "model_id": cfg["model_id"],
                "backend": backend,
                "prompts_path": str(subset_path),
                "output_path": str(cond_path),
                "condition_id": cond_id,
                "seed": cfg["seed"],
                "max_new_tokens": cfg["max_new_tokens"],
                "temperature": 0.0,
                "batch_size": cfg["hf_steered_batch_size"],
                "max_input_len": cfg.get("hf_steered_max_input_len", 4096),
                "steering": _build_compound_steering(setup, axis_name, float(lam)),
            },
            output_path=out_dir / f".step5_{cond_id}.work.json",
            timeout_seconds=10800,
        )
        _log(f"  → {res['n_rows']} rows in {res['elapsed_seconds']}s")
        out_paths.append(cond_path)

    # 5c. Multi-axis composite (sign-matched PC2+PC3 simultaneously, λ baked into combined vector)
    multi_indices = cfg.get("multi_axis_pc_indices", [])
    if multi_indices:
        combined = None
        used: list[tuple[int, float]] = []
        for pc_idx in multi_indices:
            axis_name = f"signmatched_pc{pc_idx}"
            if axis_name not in chosen:
                continue
            v = load_file(setup["axis_files"][axis_name])["vector"].float()
            lam = float(chosen[axis_name])
            combined = lam * v if combined is None else combined + lam * v
            used.append((pc_idx, lam))
        if combined is not None and used:
            multi_vec_path = out_dir / "extraction" / "vectors" / "multi_signmatched.safetensors"
            multi_vec_path.parent.mkdir(parents=True, exist_ok=True)
            save_file({"vector": combined.bfloat16().contiguous()}, str(multi_vec_path))
            cond_id = "full_aa_capped_multi_signmatched_pos1"
            cond_path = rollouts_dir / f"{cond_id}.parquet"
            if not cond_path.exists():
                _log(f"step 5: multi-axis with components {used}")
                steering = {
                    "mode": "compound",
                    "cap_vectors": setup["aa_cap_files"],
                    "cap_thresholds": setup["cap_thresholds"],
                    "cap_layers": list(setup["capping_layers"]),
                    "addition_vectors": [str(multi_vec_path)],
                    "addition_coefficients": [1.0],
                    "addition_layers": [int(setup["l_star"])],
                    "positions": "all",
                }
                res = run_in_subprocess(
                    "src.evaluation.run_subject_rollouts",
                    {
                        "model_id": cfg["model_id"],
                        "backend": backend,
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
    import pandas as pd
    from src.utils.model_runner import run_in_subprocess

    marker = out_dir / ".step6.done"
    judged_path = out_dir / "rollouts" / "full_judged.parquet"
    if marker.exists():
        _log("step 6: skipped (marker exists)")
        return judged_path

    dfs = []
    for p in full_paths:
        d = pd.read_parquet(p)
        if "category" not in d.columns:
            d["category"] = "Unknown"
        else:
            d["category"] = d["category"].fillna("Unknown")
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
        python_executable=_judge_python_executable(),
    )
    _log(f"step 6: judged {res['n_parsed']}/{res['n_rows']} rows in {res['elapsed_seconds']}s")
    _mark_done(marker, res)
    return judged_path


# ============================================================================
# Step 7 — assemble lambda_sweep.parquet + headline.json
# ============================================================================


def step_7_assemble(
    cfg: dict, out_dir: Path, mini_judged: Path, full_judged: Path, setup: dict
) -> Path:
    import pandas as pd
    from src.evaluation.safety import binarize_harm

    marker = out_dir / ".step7.done"
    out_path = out_dir / "lambda_sweep.parquet"
    if marker.exists():
        _log("step 7: skipped (marker exists)")
        return out_path

    mini = pd.read_parquet(mini_judged)
    full = pd.read_parquet(full_judged)
    mini["scope"] = "mini"
    full["scope"] = "full"
    df = pd.concat([mini, full], ignore_index=True)
    df["harm_binary"] = df["harm_label_primary"].map(binarize_harm)
    df["nonsense"] = (df["harm_label_primary"] == "nonsensical").astype(int)
    df.to_parquet(out_path, index=False)

    summary = df.groupby(["scope", "condition_id"]).agg(
        n=("harm_binary", "size"),
        harm=("harm_binary", "mean"),
        nonsense=("nonsense", "mean"),
    ).round(3)
    _log(f"step 7: lambda_sweep.parquet rows={len(df)}")
    _log(f"\n{summary.to_string()}")

    # Build headline.json: AA-cap reduction + per-attack recovery
    full_only = df[df["scope"] == "full"]
    cap_only = full_only[full_only["condition_id"] == "full_aa_capped_only"]
    cap_only_harm = float(cap_only["harm_binary"].mean()) if len(cap_only) > 0 else None

    # Phase A baseline (already on disk, unsteered → no cap)
    phase_a_metrics = json.loads((Path(cfg["phase_a_dir"]) / "metrics.json").read_text())
    baseline_harm = float(phase_a_metrics["headline"]["baseline_harm_rate"])

    headline = {
        "subject": cfg["subject_id"],
        "baseline_harm_rate": baseline_harm,
        "aa_cap_only_harm_rate": cap_only_harm,
        "aa_cap_reduction_pp": (
            (baseline_harm - cap_only_harm) * 100.0 if cap_only_harm is not None else None
        ),
        "per_attack_full": {},
        "cos_v_harm_aa": setup["cos_v_harm_aa"],
        "L_star": setup["l_star"],
    }
    for cond, sub in full_only.groupby("condition_id"):
        if cond == "full_aa_capped_only":
            continue
        h = float(sub["harm_binary"].mean())
        n = float(sub["nonsense"].mean())
        recovery = (
            (h - cap_only_harm) * 100.0 if cap_only_harm is not None else None
        )
        headline["per_attack_full"][cond] = {
            "n": int(len(sub)),
            "harm_rate": h,
            "nonsense_rate": n,
            "coherence_rate": 1.0 - n,
            "recovery_pp_vs_aa_cap": recovery,
        }
    (out_dir / "headline.json").write_text(json.dumps(headline, indent=2))

    summary_flat = {f"{scope}::{cond}": row.to_dict() for (scope, cond), row in summary.iterrows()}
    _mark_done(marker, {"n_rows": len(df), "summary": summary_flat, "headline": headline})
    return out_path


# ============================================================================
# Main
# ============================================================================


_SUBJECT_CONFIG_DEFAULT = {
    "qwen_3_32b": "configs/phase_b_qwen_3_32b.yaml",
    "gemma_4_31b_thinking_off": "configs/phase_b_gemma_4_31b_thinking_off.yaml",
    "gemma_4_31b_thinking_on": "configs/phase_b_gemma_4_31b_thinking_on.yaml",
    "gemma_2_27b": "configs/phase_b_gemma_2_27b.yaml",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", type=str, default=None,
                    help="Resolves to configs/phase_b_<subject>.yaml if --config not given")
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--skip-step", type=int, default=None)
    args = ap.parse_args()

    if args.config is None:
        if args.subject is None:
            raise SystemExit("must provide --subject or --config")
        cfg_path = Path(_SUBJECT_CONFIG_DEFAULT.get(
            args.subject, f"configs/phase_b_{args.subject}.yaml"
        ))
    else:
        cfg_path = args.config

    cfg = yaml.safe_load(cfg_path.read_text())

    from src.utils.config import assert_venv_for_subject
    assert_venv_for_subject(cfg["model_id"])

    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    _log(f"config: {cfg_path}; output_dir: {out_dir}")

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

    _log(f"PHASE B COMPLETE [{cfg['subject_id']}] → {out_path}")


if __name__ == "__main__":
    main()
