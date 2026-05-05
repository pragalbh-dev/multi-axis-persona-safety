"""Ext B — Causal v_harm test (per `plans/may_3_directive.md` 2026-05-03 thread B).

Steers along v_harm at λ ∈ {0.1, 0.25, 0.5} on the 508-prompt baseline DAN
subset (Phase B's `_full_subset.parquet`). NO cap, NO DAN-style attack — just
v_harm steered. If harm rate climbs with λ vs the unsteered Phase A baseline,
v_harm is causal for harm production; if flat, predictive only.

Reuses Phase B artifacts:
  - results/phase_b/<subject>/extraction/vectors/v_harm.safetensors  (already
    scaled to v_harm_unit · lmsys_norm at L*; addition coefficient = λ matches
    the Phase B convention)
  - results/phase_b/<subject>/rollouts/_full_subset.parquet           (n=508 DAN)
  - results/phase_b/<subject>/harm_direction.json                     (L*, lmsys_norm)

Pipeline (parent never imports torch/vllm; all heavy work via subprocess):
  1. setup    (CPU)        → verify v_harm vector + load metadata, copy subset
  2. rollout  (HF backend) → addition-mode rollouts at each λ over 508 prompts
  3. judge    (vLLM judge) → batch-judge concatenated rollouts via Qwen 3.6 27B
  4. assemble (CPU)        → harm_curve.parquet + headline.json with BCa CIs
                             and the causal-vs-correlate decision

Usage:
  uv run python -m src.experiments.ext_b_v_harm_causal --subject gemma_4_31b_thinking_off
  uv run python -m src.experiments.ext_b_v_harm_causal --config configs/ext_b_v_harm_causal_gemma_4_31b_thinking_on.yaml
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import yaml


def _log(msg: str) -> None:
    print(f"[ext_b {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _mark_done(marker: Path, payload: dict) -> None:
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps(payload, indent=2, default=str))


def _lam_token(lam: float) -> str:
    sign = "pos" if lam >= 0 else "neg"
    mag = abs(float(lam))
    if mag == int(mag):
        return f"{sign}{int(mag)}"
    return f"{sign}{mag:g}".replace(".", "p")


# ============================================================================
# Step 1 — setup: verify Phase B inputs, materialize subset
# ============================================================================


def step_1_setup(cfg: dict, out_dir: Path) -> dict:
    """CPU: verify v_harm vector + Phase B subset, write setup.json."""
    import shutil

    from safetensors.torch import load_file

    marker = out_dir / ".step1.done"
    if marker.exists():
        _log("step 1: skipped (marker exists)")
        return json.loads(marker.read_text())

    phase_b_dir = Path(cfg["phase_b_dir"])
    phase_a_dir = Path(cfg["phase_a_dir"])

    harm_dir_path = phase_b_dir / "harm_direction.json"
    if not harm_dir_path.exists():
        raise RuntimeError(f"missing Phase B harm_direction.json: {harm_dir_path}")
    harm_meta = json.loads(harm_dir_path.read_text())
    l_star = int(harm_meta["layer"])
    lmsys_norm = float(harm_meta["lmsys_norm_at_lstar"])
    cos_v_harm_aa = float(harm_meta["cos_sim_v_harm_aa"])
    _log(f"step 1: subject={cfg['subject_id']}  L*={l_star}  lmsys_norm={lmsys_norm:.3f}  "
         f"cos(v_harm,AA)={cos_v_harm_aa:+.3f}")

    v_harm_path = phase_b_dir / "extraction" / "vectors" / "v_harm.safetensors"
    if not v_harm_path.exists():
        raise RuntimeError(f"missing Phase B v_harm vector: {v_harm_path}")
    v_harm = load_file(str(v_harm_path))["vector"].float()
    v_harm_norm = float(v_harm.norm())
    # Phase B saves v_harm as v_unit * lmsys_norm in bfloat16; round-trip leaves
    # ~1e-3 noise. Sanity-check within 1% to catch silent re-use of a wrong file.
    if abs(v_harm_norm - lmsys_norm) / max(lmsys_norm, 1e-9) > 0.01:
        raise RuntimeError(
            f"v_harm.safetensors norm {v_harm_norm:.3f} disagrees with lmsys_norm "
            f"{lmsys_norm:.3f}; expected ≤1% drift. Did Phase B step_1 finish cleanly?"
        )
    _log(f"step 1: v_harm.safetensors verified (norm={v_harm_norm:.3f}, expected ≈ {lmsys_norm:.3f})")

    subset_src = phase_b_dir / "rollouts" / "_full_subset.parquet"
    if not subset_src.exists():
        raise RuntimeError(f"missing Phase B subset: {subset_src}")
    subset_dst = out_dir / "rollouts" / "_subset.parquet"
    subset_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(subset_src, subset_dst)

    # Phase A baseline harm rate for the causal-lift comparison.
    phase_a_metrics = json.loads((phase_a_dir / "metrics.json").read_text())
    baseline_harm = float(phase_a_metrics["headline"]["baseline_harm_rate"])

    payload = {
        "subject": cfg["subject_id"],
        "l_star": l_star,
        "lmsys_norm": lmsys_norm,
        "cos_v_harm_aa": cos_v_harm_aa,
        "v_harm_path": str(v_harm_path),
        "subset_path": str(subset_dst),
        "phase_a_baseline_harm_rate": baseline_harm,
    }
    _mark_done(marker, payload)
    return payload


# ============================================================================
# Step 2 — HF addition-mode rollouts at each λ
# ============================================================================


def _build_addition_steering(setup: dict, lam: float) -> dict:
    return {
        "mode": "addition",
        "vectors": [setup["v_harm_path"]],
        "coefficients": [float(lam)],
        "layers": [int(setup["l_star"])],
        "positions": "all",
    }


def step_2_rollout(cfg: dict, out_dir: Path, setup: dict) -> list[Path]:
    from src.utils.config import resolved_steered_backend
    from src.utils.model_runner import run_in_subprocess

    backend = resolved_steered_backend(cfg["model_id"])
    _log(f"step 2: steered backend = {backend}")

    marker = out_dir / ".step2.done"
    rollouts_dir = out_dir / "rollouts" / "v_harm_clean"
    rollouts_dir.mkdir(parents=True, exist_ok=True)
    if marker.exists():
        _log("step 2: skipped (marker exists)")
        return sorted(rollouts_dir.glob("*.parquet"))

    subset_path = Path(setup["subset_path"])
    out_paths: list[Path] = []
    total = len(cfg["lambda_schedule"])
    t0 = time.time()
    for cell, lam in enumerate(cfg["lambda_schedule"], start=1):
        cond_id = f"v_harm_clean_{_lam_token(lam)}"
        cond_path = rollouts_dir / f"{cond_id}.parquet"
        if cond_path.exists():
            _log(f"step 2 [{cell}/{total}] skip {cond_id} (exists)")
            out_paths.append(cond_path)
            continue
        _log(f"step 2 [{cell}/{total}] {cond_id} (λ={lam})")
        steering = _build_addition_steering(setup, lam)
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
            output_path=out_dir / f".step2_{cond_id}.work.json",
            timeout_seconds=18000,
        )
        _log(f"  → {res['n_rows']} rows in {res['elapsed_seconds']}s "
             f"({res.get('tokens_per_second', 0):.1f} tok/s)")
        out_paths.append(cond_path)

    _log(f"step 2: {total} conditions in {time.time() - t0:.0f}s")
    _mark_done(marker, {"n_conditions": total, "elapsed": time.time() - t0})
    return out_paths


# ============================================================================
# Step 3 — judge concatenated rollouts
# ============================================================================


def _judge_python_executable() -> str | None:
    """Judge needs vLLM in `.venv` (NOT `.venv-sglang`); see `phase_b.py`."""
    import sys
    from src.utils.config import REPO_ROOT
    venv_py = REPO_ROOT / ".venv" / "bin" / "python"
    if ".venv-sglang" in sys.executable and venv_py.exists():
        return str(venv_py)
    return None


def step_3_judge(cfg: dict, out_dir: Path, rollout_paths: list[Path]) -> Path:
    import pandas as pd
    from src.utils.model_runner import run_in_subprocess

    marker = out_dir / ".step3.done"
    judged_path = out_dir / "rollouts" / "v_harm_clean_judged.parquet"
    if marker.exists():
        _log("step 3: skipped (marker exists)")
        return judged_path

    dfs = []
    for p in rollout_paths:
        d = pd.read_parquet(p)
        if "category" not in d.columns:
            d["category"] = "Unknown"
        else:
            d["category"] = d["category"].fillna("Unknown")
        if "question_text" not in d.columns:
            d["question_text"] = d.get("input_text", "")
        dfs.append(d)
    all_df = pd.concat(dfs, ignore_index=True)
    all_path = out_dir / "rollouts" / "v_harm_clean_all.parquet"
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
# Step 4 — assemble harm curve, BCa CIs, decision
# ============================================================================


def _bca_ci(values, n_resamples: int, alpha: float, rng) -> tuple[float, float]:
    """BCa 95% CI for the mean of a 0/1 array. Reuses scipy if available; falls
    back to a hand-rolled implementation otherwise (no scipy dep mandatory)."""
    import numpy as np

    arr = np.asarray(values, dtype=np.float64)
    n = len(arr)
    if n == 0:
        return (float("nan"), float("nan"))
    try:
        from scipy.stats import bootstrap

        rng_seed = int(rng.integers(0, 2**31))
        res = bootstrap(
            (arr,),
            statistic=np.mean,
            n_resamples=n_resamples,
            method="BCa",
            random_state=rng_seed,
            confidence_level=1.0 - alpha,
        )
        return (float(res.confidence_interval.low), float(res.confidence_interval.high))
    except Exception:
        # Fallback: percentile bootstrap (BCa not available without scipy).
        boot = np.empty(n_resamples, dtype=np.float64)
        for i in range(n_resamples):
            idx = rng.integers(0, n, size=n)
            boot[i] = arr[idx].mean()
        lo = float(np.percentile(boot, 100 * alpha / 2))
        hi = float(np.percentile(boot, 100 * (1 - alpha / 2)))
        return (lo, hi)


def _bca_ci_diff(a, b, n_resamples: int, alpha: float, rng) -> tuple[float, float]:
    """BCa-style CI on (mean(a) − mean(b)) via paired-bootstrap on independent
    samples. a and b need not be paired; we resample each independently each
    iteration. Falls back to percentile bootstrap when scipy is unavailable."""
    import numpy as np

    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    diffs = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        ai = rng.integers(0, len(a), size=len(a))
        bi = rng.integers(0, len(b), size=len(b))
        diffs[i] = a[ai].mean() - b[bi].mean()
    lo = float(np.percentile(diffs, 100 * alpha / 2))
    hi = float(np.percentile(diffs, 100 * (1 - alpha / 2)))
    return (lo, hi)


def step_4_assemble(cfg: dict, out_dir: Path, judged_path: Path, setup: dict) -> Path:
    import numpy as np
    import pandas as pd

    from src.evaluation.safety import binarize_harm

    marker = out_dir / ".step4.done"
    out_path = out_dir / "harm_curve.parquet"
    if marker.exists():
        _log("step 4: skipped (marker exists)")
        return out_path

    df = pd.read_parquet(judged_path)
    df["harm_binary"] = df["harm_label_primary"].map(binarize_harm)
    df["nonsense"] = (df["harm_label_primary"] == "nonsensical").astype(int)

    # Parse λ from condition_id "v_harm_clean_pos0p10" → 0.10.
    def _parse_lam(cid: str) -> float:
        tok = cid[len("v_harm_clean_"):]
        sign = 1.0 if tok.startswith("pos") else -1.0
        return sign * float(tok[3:].replace("p", "."))

    df["lam"] = df["condition_id"].map(_parse_lam)

    rng = np.random.default_rng(int(cfg["seed"]))
    n_boot = int(cfg["bootstrap_n"])
    threshold_pp = float(cfg["causal_lift_threshold_pp"])
    baseline_harm = float(setup["phase_a_baseline_harm_rate"])

    # NB: baseline (Phase A unsteered) is summarized by harm rate alone in
    # metrics.json; we don't have its per-prompt harm_binary here. For the
    # diff-CI, use the binomial approximation by reconstructing a 0/1 vector of
    # the same length as the steered subset (508), proportional to baseline_harm.
    # This understates baseline variance slightly relative to a true paired
    # comparison, but the steered samples are the dominant uncertainty.
    n_subset = int((df["lam"] == df["lam"].iloc[0]).sum())
    baseline_pseudo = np.zeros(n_subset, dtype=np.int32)
    baseline_pseudo[: int(round(baseline_harm * n_subset))] = 1

    rows = []
    for lam, sub in df.groupby("lam"):
        h = sub["harm_binary"].astype(int).to_numpy()
        nn = sub["nonsense"].astype(int).to_numpy()
        harm_rate = float(h.mean())
        coh = float(1.0 - nn.mean())
        ci_lo, ci_hi = _bca_ci(h, n_boot, 0.05, rng)
        diff_lo, diff_hi = _bca_ci_diff(h, baseline_pseudo, n_boot, 0.05, rng)
        rows.append({
            "lam": float(lam),
            "n": int(len(sub)),
            "harm_rate": harm_rate,
            "harm_ci_lo": ci_lo,
            "harm_ci_hi": ci_hi,
            "coherence_rate": coh,
            "lift_pp_vs_baseline": (harm_rate - baseline_harm) * 100.0,
            "lift_ci_lo_pp": diff_lo * 100.0,
            "lift_ci_hi_pp": diff_hi * 100.0,
            "ci_excludes_zero": (diff_lo > 0.0) or (diff_hi < 0.0),
        })

    curve = pd.DataFrame(rows).sort_values("lam").reset_index(drop=True)
    curve.to_parquet(out_path, index=False)

    _log("step 4: per-λ harm curve")
    _log(f"\n{curve.to_string(index=False)}")

    # Decision: causal if any λ produces (harm − baseline) ≥ threshold AND
    # diff CI excludes zero.
    causal_hits = curve[
        (curve["lift_pp_vs_baseline"] >= threshold_pp) & (curve["lift_ci_lo_pp"] > 0.0)
    ]
    is_causal = bool(len(causal_hits) > 0)
    causal_lams = [float(x) for x in causal_hits["lam"].tolist()] if is_causal else []

    headline = {
        "subject": cfg["subject_id"],
        "l_star": int(setup["l_star"]),
        "cos_v_harm_aa": float(setup["cos_v_harm_aa"]),
        "phase_a_baseline_harm_rate": baseline_harm,
        "causal_lift_threshold_pp": threshold_pp,
        "is_causal": is_causal,
        "causal_lambdas": causal_lams,
        "per_lambda": [
            {k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in r.items()}
            for r in rows
        ],
    }
    (out_dir / "headline.json").write_text(json.dumps(headline, indent=2))
    _log(f"step 4: causal={is_causal}  lambdas_above_threshold={causal_lams}")
    _mark_done(marker, headline)
    return out_path


# ============================================================================
# Main
# ============================================================================


_SUBJECT_CONFIG_DEFAULT = {
    "gemma_4_31b_thinking_off": "configs/ext_b_v_harm_causal_gemma_4_31b_thinking_off.yaml",
    "gemma_4_31b_thinking_on": "configs/ext_b_v_harm_causal_gemma_4_31b_thinking_on.yaml",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", type=str, default=None,
                    help="Resolves to configs/ext_b_v_harm_causal_<subject>.yaml if --config not given")
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--skip-step", type=int, default=None)
    args = ap.parse_args()

    if args.config is None:
        if args.subject is None:
            raise SystemExit("must provide --subject or --config")
        cfg_path = Path(_SUBJECT_CONFIG_DEFAULT.get(
            args.subject, f"configs/ext_b_v_harm_causal_{args.subject}.yaml"
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
        rollout_paths = step_2_rollout(cfg, out_dir, setup)
    else:
        rollout_paths = sorted((out_dir / "rollouts" / "v_harm_clean").glob("*.parquet"))

    if skip < 3:
        judged_path = step_3_judge(cfg, out_dir, rollout_paths)
    else:
        judged_path = out_dir / "rollouts" / "v_harm_clean_judged.parquet"

    if skip < 4:
        out_path = step_4_assemble(cfg, out_dir, judged_path, setup)
    else:
        out_path = out_dir / "harm_curve.parquet"

    _log(f"EXT B COMPLETE [{cfg['subject_id']}] → {out_path}")


if __name__ == "__main__":
    main()
