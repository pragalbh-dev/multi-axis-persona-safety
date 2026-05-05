"""Phase E — Capability eval (trimmed) per `plans/may_3_directive.md` 2026-05-03.

Scope:
  - 4 subjects × {unsteered, aa_cap} × {IFEval, GSM8k, EQ-Bench}
  - + multi_axis_cap × 3 benches on gemma_4_31b_thinking_off only
  - Total = 9 (subject × condition) rollout cells, each covering all 3 benches
    via a combined prompts parquet (one model load per cell, post-hoc split on
    `dataset` for scoring).

Pipeline (parent never imports torch/vllm):
  1. setup    (CPU)        → prep combined prompts parquet (IFEval ⊕ GSM8k ⊕ EQ-Bench)
                            and resolve per-subject AA-cap + multi-axis-cap steering
                            specs from Phase B / Plan B / Phase D artifacts.
  2. rollout  (vllm/hf/sglang) → 9 cells via run_subject_rollouts subprocess.
  3. score    (CPU)        → per-(cell, bench) scoring; writes per-cell metrics JSONs.
  4. assemble (CPU)        → results/phase_e/headline.json + per-subject JSON.

Usage:
  uv run python -m src.experiments.phase_e --config configs/phase_e.yaml
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import yaml


def _log(msg: str) -> None:
    print(f"[phase_e {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _mark_done(marker: Path, payload: dict) -> None:
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps(payload, indent=2, default=str))


# ============================================================================
# Step 1 — setup: prompts parquets + per-subject steering specs
# ============================================================================


def _build_combined_prompts(cfg: dict, prompts_dir: Path) -> Path:
    """Concat IFEval + GSM8k + EQ-Bench prompts into a single parquet.

    Schema (uniform):
      prompt_id (str), dataset (str), input_text (str), gold (object) — gold
      carries the bench-specific reference (instruction_id_list/kwargs for
      IFEval, gold answer string for GSM8k, reference dict for EQ-Bench).

    Bench-specific gold columns are also kept verbatim so scorers can read
    them directly (see capability_score.py).
    """
    import pandas as pd

    out_path = prompts_dir / "combined.parquet"
    if out_path.exists():
        return out_path

    rows: list[dict] = []
    for bench, spec in cfg["benches"].items():
        src = Path(spec["src_jsonl"])
        prompt_field = spec["prompt_field"]
        with src.open() as f:
            for line in f:
                d = json.loads(line)
                pid = str(d.get("id", d.get("key", len(rows))))
                row = {
                    "prompt_id": f"{bench}_{pid}",
                    "dataset": bench,
                    "input_text": d[prompt_field],
                }
                if bench == "ifeval":
                    # Serialize as JSON strings so pandas/pyarrow doesn't union
                    # the kwargs dict schema across all 541 rows (which inflates
                    # each kwargs entry with None placeholders for keys that
                    # belong to other rows' instruction types — IFEval scorer
                    # expects clean per-row kwargs).
                    row["instruction_id_list_json"] = json.dumps(d["instruction_id_list"])
                    row["kwargs_json"] = json.dumps(d["kwargs"])
                elif bench == "gsm8k":
                    row["answer"] = d["answer"]
                elif bench == "eq_bench":
                    row["reference_answer_fullscale"] = d.get(
                        "reference_answer_fullscale", d.get("reference_answer", "{}")
                    )
                rows.append(row)
    df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    return out_path


def _resolve_aa_cap_for_subject(subject_id: str, src: dict) -> dict:
    """Return {vectors: [...], cap_thresholds: [...], layers: [int per vector]}.

    For Phase B subjects: read .step1.done verbatim.
    For Plan B Gemma 2 27B: combine extraction/tau_calibration.json (layers +
    p25 thresholds — converted to negative per cap_threshold sign convention)
    with extraction/vectors/aa_L<X>.safetensors.
    """
    src_type = src["type"]
    src_dir = Path(src["dir"])

    if src_type == "phase_b":
        step1 = json.loads((src_dir / ".step1.done").read_text())
        return {
            "vectors": list(step1["aa_cap_files"]),
            "cap_thresholds": [float(t) for t in step1["cap_thresholds"]],
            "layers": [int(L) for L in step1["capping_layers"]],
        }

    if src_type == "plan_b":
        tau_calib = json.loads((src_dir / "extraction" / "tau_calibration.json").read_text())
        layers = [int(L) for L in tau_calib["capping_layers"]]
        pct = int(tau_calib["tau_percentile"])
        per_layer = tau_calib["per_layer"]
        thresholds = [-float(per_layer[str(L)][f"p{pct}"]) for L in layers]
        vec_dir = src_dir / "extraction" / "vectors"
        vectors = [str(vec_dir / f"aa_L{L}.safetensors") for L in layers]
        return {
            "vectors": vectors,
            "cap_thresholds": thresholds,
            "layers": layers,
        }

    raise ValueError(f"unknown aa_cap source type: {src_type!r}")


def _resolve_multi_axis_cap_for_subject(subject_id: str, phase_d_dir: Path, aa_cap: dict) -> dict:
    """Build multi-axis cap spec = AA + PC2 + PC3 from Phase D step1 + step10.

    Phase D's chosen percentiles are in multi_axis_calibration.json. PC2 / PC3
    cap vectors are at extraction/cap_vectors/. Both PCs are capped at the
    single PCA layer (L*=14 for g4_off) — different from AA's 8-layer range.
    """
    step1 = json.loads((phase_d_dir / ".step1.done").read_text())
    pc_cap_layer = int(step1["pc_cap_layer"])
    calib = json.loads((phase_d_dir / "multi_axis_calibration.json").read_text())
    pc2_pct = int(calib["pc2"]["chosen_percentile"])
    pc3_pct = int(calib["pc3"]["chosen_percentile"])

    def _resolve_pc(axis_name: str, pct: int) -> tuple[str, float, int]:
        path = step1["pc_cap_files"][axis_name][str(pc_cap_layer)]
        # Phase D τ convention: cap_threshold = -p<pct> of role-rollout
        # projection on the harm-negative input direction. Same sign-flip as
        # Phase B AA. See phase_d.py::_resolve_pc_tau.
        tau = -float(step1["pc_tau_per_layer"][axis_name][str(pc_cap_layer)][f"p{pct}"])
        return path, tau, pc_cap_layer

    pc2_path, pc2_tau, pc2_L = _resolve_pc("signmatched_pc2", pc2_pct)
    pc3_path, pc3_tau, pc3_L = _resolve_pc("signmatched_pc3", pc3_pct)

    return {
        "vectors": list(aa_cap["vectors"]) + [pc2_path, pc3_path],
        "cap_thresholds": list(aa_cap["cap_thresholds"]) + [pc2_tau, pc3_tau],
        "layers": list(aa_cap["layers"]) + [pc2_L, pc3_L],
        "pc2_percentile": pc2_pct,
        "pc3_percentile": pc3_pct,
    }


def _normalize_cap_vector_files(spec: dict, scratch_dir: Path, prefix: str) -> dict:
    """Re-save each cap vector under key 'vector' inside scratch_dir.

    Plan B Gemma 2 27B saved its AA cap vectors with key 'v'; Phase B / Phase D
    saved with key 'vector'. The HF backend's _load_vec handles both, but SGLang
    `_load_vector` expects 'vector' (the default in capping_factory). For a
    sweep that mixes both backends per subject, normalize once here so every
    downstream consumer reads a uniform layout.
    """
    from safetensors.torch import load_file, save_file

    scratch_dir.mkdir(parents=True, exist_ok=True)
    new_paths: list[str] = []
    for i, p in enumerate(spec["vectors"]):
        src = Path(p)
        d = load_file(str(src))
        if "vector" in d:
            tensor = d["vector"]
        elif "v" in d:
            tensor = d["v"]
        else:
            tensor = next(iter(d.values()))
        target = scratch_dir / f"{prefix}_{i:02d}_{src.stem}.safetensors"
        save_file({"vector": tensor.contiguous()}, str(target))
        new_paths.append(str(target))
    return {**spec, "vectors": new_paths}


def step_1_setup(cfg: dict, out_dir: Path) -> dict:
    marker = out_dir / ".step1.done"
    if marker.exists():
        _log("step 1: skipped (marker exists)")
        return json.loads(marker.read_text())

    prompts_dir = out_dir / "prompts"
    combined_path = _build_combined_prompts(cfg, prompts_dir)
    _log(f"step 1: combined prompts → {combined_path}")

    norm_dir = out_dir / "vectors"
    aa_specs: dict[str, dict] = {}
    for subject_id in cfg["subjects"]:
        src = cfg["aa_cap_sources"][subject_id]
        raw = _resolve_aa_cap_for_subject(subject_id, src)
        aa_specs[subject_id] = _normalize_cap_vector_files(
            raw, norm_dir / subject_id / "aa", prefix="aa"
        )
        _log(
            f"step 1: aa_cap[{subject_id}] = {len(aa_specs[subject_id]['vectors'])} vectors "
            f"at layers {aa_specs[subject_id]['layers']}"
        )

    ma_specs: dict[str, dict] = {}
    for subject_id, ms in cfg.get("multi_axis_subjects", {}).items():
        phase_d_dir = Path(ms["phase_d_dir"])
        raw_ma = _resolve_multi_axis_cap_for_subject(
            subject_id, phase_d_dir, aa_specs[subject_id]
        )
        ma_specs[subject_id] = _normalize_cap_vector_files(
            raw_ma, norm_dir / subject_id / "multi_axis", prefix="ma"
        )
        # _normalize_cap_vector_files only normalizes the `vectors` list; carry
        # over PC percentile metadata.
        for k in ("pc2_percentile", "pc3_percentile"):
            if k in raw_ma:
                ma_specs[subject_id][k] = raw_ma[k]
        _log(
            f"step 1: multi_axis_cap[{subject_id}] = {len(ma_specs[subject_id]['vectors'])} vectors "
            f"(AA layers {aa_specs[subject_id]['layers']} + PC2/PC3 at L={ma_specs[subject_id]['layers'][-1]})"
        )

    payload = {
        "combined_prompts_path": str(combined_path),
        "aa_cap_specs": aa_specs,
        "multi_axis_specs": ma_specs,
    }
    _mark_done(marker, payload)
    return payload


# ============================================================================
# Step 2 — rollouts (9 cells)
# ============================================================================


def _cells(cfg: dict, setup: dict) -> list[dict]:
    """Enumerate (subject_id, condition_id, steering, backend) tuples."""
    out: list[dict] = []
    for subject_id in cfg["subjects"]:
        # unsteered (vllm)
        out.append({
            "subject_id": subject_id,
            "condition": "unsteered",
            "backend": "vllm",
            "steering": {"mode": "none"},
        })
        # aa_cap (hf or sglang per subjects.yaml::steered_backend)
        aa = setup["aa_cap_specs"][subject_id]
        out.append({
            "subject_id": subject_id,
            "condition": "aa_cap",
            "backend": None,  # resolved from configs/subjects.yaml
            "steering": {
                "mode": "capping",
                "vectors": list(aa["vectors"]),
                "coefficients": [1.0] * len(aa["vectors"]),
                "cap_thresholds": list(aa["cap_thresholds"]),
                "layers": list(aa["layers"]),
                "positions": "all",
            },
        })
        # multi_axis_cap — only configured subjects
        if subject_id in setup.get("multi_axis_specs", {}):
            ma = setup["multi_axis_specs"][subject_id]
            out.append({
                "subject_id": subject_id,
                "condition": "multi_axis_cap",
                "backend": None,
                "steering": {
                    "mode": "capping",
                    "vectors": list(ma["vectors"]),
                    "coefficients": [1.0] * len(ma["vectors"]),
                    "cap_thresholds": list(ma["cap_thresholds"]),
                    "layers": list(ma["layers"]),
                    "positions": "all",
                },
            })
    return out


def _python_for_backend(backend: str) -> str | None:
    """Route each backend to the venv that has its deps.

    .venv         → vllm + transformers + accelerate (HF backend works)
    .venv-sglang  → sglang only (no vllm, no accelerate)

    Phase E sweeps mixed backends, so we route per-cell:
      vllm   → .venv/bin/python
      hf     → .venv/bin/python
      sglang → .venv-sglang/bin/python

    If the relevant venv is missing on this host, return None so the parent's
    sys.executable is used (let it fail loudly with the original error).
    """
    import sys

    from src.utils.config import REPO_ROOT

    target = ".venv" if backend in ("vllm", "hf") else ".venv-sglang"
    if target in sys.executable:
        return None  # already in the right venv
    candidate = REPO_ROOT / target / "bin" / "python"
    return str(candidate) if candidate.exists() else None


def step_2_rollouts(cfg: dict, out_dir: Path, setup: dict) -> list[dict]:
    from src.utils.config import resolved_steered_backend
    from src.utils.model_runner import run_in_subprocess

    marker = out_dir / ".step2.done"
    rollouts_dir = out_dir / "rollouts"
    rollouts_dir.mkdir(parents=True, exist_ok=True)

    cells = _cells(cfg, setup)
    n = len(cells)
    completed: list[dict] = []
    for i, cell in enumerate(cells, start=1):
        subject_id = cell["subject_id"]
        cond = cell["condition"]
        cell_id = f"{subject_id}__{cond}"
        out_path = rollouts_dir / f"{cell_id}.parquet"
        if out_path.exists():
            _log(f"step 2 [{i}/{n}] skip {cell_id} (exists)")
            completed.append({"cell_id": cell_id, "out_path": str(out_path), "skipped": True})
            continue

        backend = cell["backend"] or (
            "vllm" if cond == "unsteered" else resolved_steered_backend(subject_id)
        )
        args = {
            "model_id": subject_id,
            "backend": backend,
            "prompts_path": setup["combined_prompts_path"],
            "output_path": str(out_path),
            "condition_id": f"phase_e_{cond}",
            "seed": cfg["seed"],
            "max_new_tokens": int(cfg["max_new_tokens"]),
            "temperature": 0.0,
        }
        if backend == "vllm":
            args["profile"] = cfg.get("vllm_profile", "short")
        else:
            args["steering"] = cell["steering"]
            args["batch_size"] = int(cfg.get("hf_steered_batch_size", 8))
            args["max_input_len"] = int(cfg["max_input_len"])

        _log(f"step 2 [{i}/{n}] {cell_id} backend={backend}")
        res = run_in_subprocess(
            "src.evaluation.run_subject_rollouts",
            args,
            output_path=out_dir / f".step2_{cell_id}.work.json",
            timeout_seconds=28800,  # 8 hr per cell ceiling
            # Bumped 2026-05-04 from 14400 after g4_on AA-cap on HF exceeded 4hr.
            # g4_off HF steered cells took ~115min each; g4_on thinking-mode
            # adds reasoning traces that inflate output tokens roughly 2-3×, so
            # 8hr leaves comfortable margin. (see plans/decisions.md)
            python_executable=_python_for_backend(backend),
        )
        _log(f"  → {res['n_rows']} rows in {res['elapsed_seconds']}s "
             f"({res['tokens_per_second']} tok/s)")
        completed.append({
            "cell_id": cell_id,
            "subject_id": subject_id,
            "condition": cond,
            "backend": backend,
            "out_path": str(out_path),
            "n_rows": res["n_rows"],
            "tokens_per_second": res["tokens_per_second"],
            "elapsed_seconds": res["elapsed_seconds"],
        })

    _mark_done(marker, {"cells": completed})
    return completed


# ============================================================================
# Step 3 — score per (cell, bench)
# ============================================================================


def step_3_score(cfg: dict, out_dir: Path, setup: dict, cells: list[dict]) -> dict:
    import pandas as pd

    from src.evaluation.capability_score import score_bench

    marker = out_dir / ".step3.done"
    if marker.exists():
        _log("step 3: skipped (marker exists)")
        return json.loads(marker.read_text())

    metrics_dir = out_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    benches = list(cfg["benches"].keys())

    all_metrics: dict[str, dict[str, dict[str, dict]]] = {}
    for cell in cells:
        cell_id = cell["cell_id"]
        df = pd.read_parquet(cell["out_path"])
        per_bench: dict[str, dict] = {}
        for bench in benches:
            sub = df[df["dataset"] == bench].copy()
            if sub.empty:
                _log(f"step 3: {cell_id} / {bench} → no rows")
                continue
            m = score_bench(bench, sub)
            per_bench[bench] = m
            _log(f"step 3: {cell_id} / {bench}  n={m['n']}  score={m['score']:.4f}")
        cell_path = metrics_dir / f"{cell_id}.json"
        cell_path.write_text(json.dumps({"cell_id": cell_id, "per_bench": per_bench}, indent=2))
        all_metrics[cell_id] = per_bench

    _mark_done(marker, {"metrics_dir": str(metrics_dir), "by_cell": all_metrics})
    return {"metrics_dir": str(metrics_dir), "by_cell": all_metrics}


# ============================================================================
# Step 4 — assemble headline.json
# ============================================================================


def step_4_assemble(cfg: dict, out_dir: Path, setup: dict, score_payload: dict) -> Path:
    marker = out_dir / ".step4.done"
    headline_path = out_dir / "headline.json"
    if marker.exists():
        _log("step 4: skipped (marker exists)")
        return headline_path

    by_cell: dict[str, dict] = score_payload["by_cell"]

    # Re-pivot: matrix[subject][condition][bench] = {n, score, extra}
    matrix: dict[str, dict[str, dict[str, dict]]] = {}
    for cell_id, per_bench in by_cell.items():
        subject_id, cond = cell_id.split("__", 1)
        matrix.setdefault(subject_id, {})[cond] = per_bench

    # Cross-condition deltas per subject × bench (AA-cap delta vs unsteered).
    deltas: dict[str, dict[str, dict[str, float]]] = {}
    for subject_id, per_cond in matrix.items():
        deltas[subject_id] = {}
        unsteered = per_cond.get("unsteered", {})
        for cond, per_bench in per_cond.items():
            if cond == "unsteered":
                continue
            deltas[subject_id][cond] = {}
            for bench, m in per_bench.items():
                base = unsteered.get(bench, {}).get("score")
                if base is None:
                    continue
                deltas[subject_id][cond][bench] = round(float(m["score"]) - float(base), 4)

    # Subject-level aggregate score = mean across benches (per condition).
    aggregates: dict[str, dict[str, float]] = {}
    for subject_id, per_cond in matrix.items():
        aggregates[subject_id] = {}
        for cond, per_bench in per_cond.items():
            scores = [m["score"] for m in per_bench.values() if "score" in m]
            if scores:
                aggregates[subject_id][cond] = round(float(sum(scores) / len(scores)), 4)

    headline = {
        "experiment_id": cfg["experiment_id"],
        "subjects": cfg["subjects"],
        "benches": list(cfg["benches"].keys()),
        "matrix": matrix,
        "deltas_vs_unsteered": deltas,
        "aggregate_per_condition": aggregates,
        "multi_axis_subjects": list(cfg.get("multi_axis_subjects", {}).keys()),
    }
    headline_path.write_text(json.dumps(headline, indent=2))
    _log("step 4: headline written.")
    _log("  Aggregates per (subject, condition):")
    for subject_id, per_cond in aggregates.items():
        for cond, agg in per_cond.items():
            _log(f"    {subject_id:30s}  {cond:18s}  agg={agg:.4f}")
    _mark_done(marker, {"headline_path": str(headline_path)})
    return headline_path


# ============================================================================
# Main
# ============================================================================


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("configs/phase_e.yaml"))
    ap.add_argument("--skip-step", type=int, default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())

    # Per-subject venv guard happens at rollout time inside the subprocess child;
    # the parent stays venv-agnostic so a single Phase E run can sweep both
    # SGLang (Gemma 2 / Qwen 3) and HF (Gemma 4 modes) subjects.

    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    _log(f"config: {args.config}; output_dir: {out_dir}")

    skip = args.skip_step or 0
    setup = step_1_setup(cfg, out_dir) if skip < 1 else json.loads(
        (out_dir / ".step1.done").read_text()
    )

    cells = step_2_rollouts(cfg, out_dir, setup) if skip < 2 else json.loads(
        (out_dir / ".step2.done").read_text()
    )["cells"]

    score_payload = step_3_score(cfg, out_dir, setup, cells) if skip < 3 else json.loads(
        (out_dir / ".step3.done").read_text()
    )

    headline = step_4_assemble(cfg, out_dir, setup, score_payload) if skip < 4 else (
        out_dir / "headline.json"
    )
    _log(f"PHASE E COMPLETE → {headline}")


if __name__ == "__main__":
    main()
