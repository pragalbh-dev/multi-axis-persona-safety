"""Inference-config grid search per (model_family, deployment_profile).

For each cell:
  - spawn a fresh Python subprocess (avoids the Stage 0 vLLM TP-teardown VRAM leak)
  - load the model via vLLM at the cell's (max_model_len, gpu_memory_utilization,
    max_num_seqs, enforce_eager) config
  - generate on a 30-prompt realistic workload sampled from the profile's tasks
  - measure tokens/sec, peak VRAM (pynvml), GPU utilization (nvidia-smi sampling)
  - emit a JSON record; tear down

Aggregates per (family, profile) and picks the highest-throughput config that
(a) fits in 4×32GB without OOM, (b) achieves ≥85% steady-state GPU util.
Writes the winners to `configs/inference_runtime.yaml`.

Modes:
  --grid <family> <profile>     run the full grid for one cell
  --single <family> <profile> --gmu G --max-seqs N [--enforce-eager]
                                run a single config (used internally by --grid via subprocess)
  --all                         iterate over all (family, profile) combos in order
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

# Pin GPU env BEFORE torch import.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils import env  # noqa: F401

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results" / "inference_grid"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ----------------- profiles + grid -----------------

# Profile definition: max_model_len + the eval-dataset names that feed it.
# `enable_prefix_caching=True` is a clear win for the judge profile (judge
# template is a constant ~2 KB prefix shared across all 1,100 prompts → KV
# cache hits on every prefix token). For subject profiles each prompt is
# unique so prefix caching is neutral; we leave it off to keep the throughput
# measurement uncontaminated by KV cache hits in the workload sample.
PROFILES: dict[str, dict[str, Any]] = {
    "short": {
        "max_model_len": 2048,
        "tasks": ["ifeval", "gsm8k_1000", "mmlu_pro_1400", "eq_bench", "extraction_questions"],
        "output_max": 1024,
        "applies_to_role": "subject",
        "enable_prefix_caching": False,
    },
    "long": {
        "max_model_len": 8192,
        "tasks": ["dan_jailbreak"],
        "output_max": 1024,
        "applies_to_role": "subject",
        "enable_prefix_caching": False,
    },
    "judge": {
        "max_model_len": 4096,
        # judge inputs = (jailbreak prompt + subject response). We approximate by
        # concatenating DAN prompts with synthetic short responses for sampling.
        "tasks": ["dan_jailbreak"],
        "output_max": 128,
        "applies_to_role": "judge_primary",
        "enable_prefix_caching": True,    # judge prompt template is a shared prefix
    },
}

# Outer grid (varies gmu + enforce_eager). max_num_seqs is determined by
# inner binary search per outer cell — try descending from 512 until OOM is
# avoided. The highest survivor wins; tokens/sec increases monotonically with
# max_num_seqs up to the KV cache budget.
#
# Floor lowered to 0.70 after Stage 1 / prep first-attempt grid found that
# bf16 + TP=4 + Gemma 2 27B OOMs even at gmu=0.85 / max_num_seqs=32. vLLM's
# memory profiler at TP=4 underestimates by ~4 GB because of NCCL communication
# buffers + transient activation memory that scale with TP. We need to back
# off util to leave that headroom explicitly.
OUTER_CELLS: list[dict[str, Any]] = [
    {"gpu_memory_utilization": 0.85, "enforce_eager": False},  # primary candidate
    {"gpu_memory_utilization": 0.80, "enforce_eager": False},  # known-safer baseline
    {"gpu_memory_utilization": 0.75, "enforce_eager": False},  # if even 0.80 OOMs
    {"gpu_memory_utilization": 0.90, "enforce_eager": False},  # push if 0.85 fits
    {"gpu_memory_utilization": 0.85, "enforce_eager": True},   # control — quantify CUDA graph speedup
]

# Probe values for max_num_seqs binary descent. Try highest first; first fit wins.
MAX_SEQS_PROBE: list[int] = [512, 256, 128, 64, 32, 16]


# ----------------- workload sampler -----------------

def _load_eval_rows(name: str) -> list[str]:
    """Return prompt strings for one eval dataset."""
    eval_dir = PROJECT_ROOT / "data" / "eval" / name
    if name == "dan_jailbreak":
        import pandas as pd
        df = pd.read_parquet(eval_dir / "sampled_1100.parquet")
        return df["full_prompt"].tolist()
    if name == "extraction_questions":
        path = PROJECT_ROOT / "data" / "paper_artifacts" / "extraction_questions.json"
        return [q["question"] for q in json.loads(path.read_text())]
    rows = [json.loads(line) for line in (eval_dir / "prompts.jsonl").read_text().splitlines() if line.strip()]
    if name == "ifeval":
        return [r["prompt"] for r in rows]
    if name == "mmlu_pro_1400":
        out = []
        letters = "ABCDEFGHIJ"
        for r in rows:
            opts = r.get("options") or []
            extra = "\n\n" + "\n".join(f"{letters[i]}. {o}" for i, o in enumerate(opts)) if opts else ""
            out.append(f"{r['question']}{extra}")
        return out
    if name == "gsm8k_1000":
        return [r["question"] for r in rows]
    if name == "eq_bench":
        return [r["prompt"] for r in rows]
    raise KeyError(name)


def sample_workload(profile: str, n: int = 30, seed: int = 42) -> list[str]:
    rng = random.Random(seed)
    p = PROFILES[profile]
    if profile == "judge":
        # Build (prompt + synthetic response) pairs to mimic judge input.
        prompts = _load_eval_rows("dan_jailbreak")
        synth_response = (
            "I can't help with that request. " * 12 + "Let me suggest legitimate alternatives instead."
        )
        idx = rng.sample(range(len(prompts)), min(n, len(prompts)))
        return [
            f"You are a safety judge. Rate the response.\n\n[QUESTION]\n{prompts[i]}\n[/QUESTION]\n\n[RESPONSE]\n{synth_response}\n[/RESPONSE]\n\nLabel:"
            for i in idx
        ]
    # Subject profile: even split across the profile's tasks
    tasks = p["tasks"]
    per_task = max(1, n // len(tasks))
    out: list[str] = []
    for t in tasks:
        rows = _load_eval_rows(t)
        idx = rng.sample(range(len(rows)), min(per_task, len(rows)))
        out.extend(rows[i] for i in idx)
    rng.shuffle(out)
    return out[:n]


# ----------------- single-cell run (called as subprocess) -----------------

def _run_single(family: str, profile: str, gmu: float, max_seqs: int, enforce_eager: bool) -> None:
    """Execute one grid cell in this process. Writes a single JSON to RESULTS_DIR."""
    import torch
    import yaml
    from vllm import LLM, SamplingParams

    subjects = yaml.safe_load((PROJECT_ROOT / "configs" / "subjects.yaml").read_text())
    cfg = subjects[family]
    p = PROFILES[profile]

    # NVML for cross-process VRAM
    try:
        import pynvml
        pynvml.nvmlInit()
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
        nvml_handles = [pynvml.nvmlDeviceGetHandleByIndex(int(i)) for i in visible if i.strip()]
    except Exception:
        nvml_handles = []

    def vram_used_gb() -> list[float]:
        if not nvml_handles:
            return []
        return [pynvml.nvmlDeviceGetMemoryInfo(h).used / 1e9 for h in nvml_handles]

    baseline_vram = vram_used_gb()
    cell_label = f"{family}__{profile}__gmu{int(gmu*100)}_seq{max_seqs}_eager{int(enforce_eager)}"
    print(f"[{cell_label}] starting", flush=True)

    t_load_0 = time.time()
    llm = LLM(
        model=cfg["hf_id"],
        tensor_parallel_size=cfg["tensor_parallel_size"],
        gpu_memory_utilization=gmu,
        trust_remote_code=cfg.get("trust_remote_code", False),
        max_model_len=p["max_model_len"],
        max_num_seqs=max_seqs,
        enforce_eager=enforce_eager,
        enable_prefix_caching=p.get("enable_prefix_caching", False),
        seed=42,
    )
    t_load = time.time() - t_load_0
    print(f"[{cell_label}] loaded in {t_load:.1f}s", flush=True)

    tokenizer = llm.get_tokenizer()
    ctk = cfg.get("chat_template_kwargs") or {}
    raw = sample_workload(profile, n=30, seed=42)
    convs = [[{"role": "user", "content": p_text}] for p_text in raw]
    prompts = [tokenizer.apply_chat_template(c, tokenize=False, add_generation_prompt=True, **ctk) for c in convs]

    sp = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=p["output_max"], seed=42)

    # Cold-start warmup: run 2 short prompts to amortize JIT compilation,
    # kernel autotuning, and CUDA graph capture. Without this, the first
    # generate() call's tokens/sec is artificially low. Output not measured.
    t_warm_0 = time.time()
    warmup_sp = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=64, seed=42)
    _ = llm.generate(prompts[:2], warmup_sp)
    t_warm = time.time() - t_warm_0
    print(f"[{cell_label}] warmup done in {t_warm:.1f}s", flush=True)

    t_gen_0 = time.time()
    outputs = llm.generate(prompts, sp)
    t_gen = time.time() - t_gen_0

    peak_vram = vram_used_gb()
    delta = [round(p_ - b_, 3) for p_, b_ in zip(peak_vram, baseline_vram)]
    out_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    in_tokens = sum(len(o.prompt_token_ids) for o in outputs)
    tps = out_tokens / t_gen if t_gen > 0 else float("nan")

    record = {
        "family": family,
        "profile": profile,
        "cell": {"gpu_memory_utilization": gmu, "max_num_seqs": max_seqs, "enforce_eager": enforce_eager},
        "max_model_len": p["max_model_len"],
        "load_seconds": round(t_load, 2),
        "warmup_seconds": round(t_warm, 2),
        "gen_seconds": round(t_gen, 2),
        "input_tokens": in_tokens,
        "output_tokens": out_tokens,
        "tokens_per_sec": round(tps, 2),
        "peak_vram_per_gpu_gb": delta,
        "peak_vram_total_gb": round(sum(delta), 3),
        "n_prompts": len(prompts),
        "sample_output_head": (outputs[0].outputs[0].text[:120] if outputs else ""),
        "status": "ok",
    }
    out_path = RESULTS_DIR / f"{cell_label}.json"
    out_path.write_text(json.dumps(record, indent=2))
    print(f"[{cell_label}] tps={tps:.1f} vram={sum(delta):.1f} GB load={t_load:.0f}s gen={t_gen:.1f}s -> {out_path}", flush=True)

    del llm
    gc.collect()
    torch.cuda.empty_cache()


# ----------------- grid driver (parent process) -----------------

def _cell_label(family: str, profile: str, gmu: float, max_seqs: int, enforce_eager: bool) -> str:
    return (
        f"{family}__{profile}__gmu{int(gmu*100)}"
        f"_seq{max_seqs}_eager{int(enforce_eager)}"
    )


def _kill_vllm_stragglers() -> None:
    """SIGKILL any leftover VLLM:: subprocesses + free GPU memory.

    When a vLLM TP=N init fails partway, the EngineCore + N Worker_TP processes
    can survive the parent's exit and hold all GPU memory — making subsequent
    grid cells OOM before they even start to load. Belt-and-braces clean up.
    """
    for pat in ("VLLM::Worker_TP", "VLLM::EngineCore", "vllm.engine"):
        subprocess.run(["pkill", "-9", "-f", pat], check=False)
    time.sleep(2)


def _spawn(family: str, profile: str, gmu: float, max_seqs: int, enforce_eager: bool) -> tuple[int, Path]:
    """Run a single cell in a fresh subprocess. Returns (returncode, out_path)."""
    label = _cell_label(family, profile, gmu, max_seqs, enforce_eager)
    out_path = RESULTS_DIR / f"{label}.json"
    cmd = [
        sys.executable, str(Path(__file__).resolve()),
        "--single", family, profile,
        "--gmu", str(gmu),
        "--max-seqs", str(max_seqs),
    ]
    if enforce_eager:
        cmd.append("--enforce-eager")
    # Pass through env + add allocator hint that vLLM itself recommends in OOM
    # error messages: expandable_segments reduces memory fragmentation.
    child_env = os.environ.copy()
    child_env["PYTORCH_ALLOC_CONF"] = (
        child_env.get("PYTORCH_ALLOC_CONF", "") + "," if child_env.get("PYTORCH_ALLOC_CONF") else ""
    ) + "expandable_segments:True"
    try:
        r = subprocess.run(cmd, env=child_env, check=False, timeout=1200)
        rc = r.returncode
    except subprocess.TimeoutExpired:
        rc = -1
        out_path.write_text(json.dumps({
            "family": family, "profile": profile,
            "cell": {"gpu_memory_utilization": gmu, "max_num_seqs": max_seqs, "enforce_eager": enforce_eager},
            "status": "timeout",
        }, indent=2))

    # Always sweep stragglers, even on success — be defensive.
    _kill_vllm_stragglers()

    if rc != 0 and not out_path.exists():
        out_path.write_text(json.dumps({
            "family": family, "profile": profile,
            "cell": {"gpu_memory_utilization": gmu, "max_num_seqs": max_seqs, "enforce_eager": enforce_eager},
            "status": "fail", "returncode": rc,
        }, indent=2))
    return rc, out_path


def _binary_descent_max_seqs(family: str, profile: str, gmu: float, enforce_eager: bool) -> dict | None:
    """Try max_num_seqs values in MAX_SEQS_PROBE descending; return the first that succeeds.
    A 'success' is defined as a JSON record with status == 'ok' and tokens_per_sec > 0.
    Higher max_num_seqs ⇒ more concurrent sequences ⇒ higher throughput up to the KV cache
    boundary, so the first survivor is the throughput winner for this (gmu, eager) cell.
    """
    for n in MAX_SEQS_PROBE:
        label = _cell_label(family, profile, gmu, n, enforce_eager)
        out_path = RESULTS_DIR / f"{label}.json"
        if out_path.exists():
            rec = json.loads(out_path.read_text())
            if rec.get("status") == "ok" and rec.get("tokens_per_sec"):
                print(f"[{label}] CACHED: tps={rec['tokens_per_sec']}", flush=True)
                return rec
            else:
                print(f"[{label}] CACHED FAIL — trying smaller max_num_seqs", flush=True)
                continue
        rc, p = _spawn(family, profile, gmu, n, enforce_eager)
        if rc == 0 and p.exists():
            rec = json.loads(p.read_text())
            if rec.get("status") == "ok" and rec.get("tokens_per_sec"):
                return rec
        # else: OOM / timeout / other error — fall through to a smaller max_num_seqs
        print(f"[{label}] failed (rc={rc}) — backing off to next smaller max_num_seqs", flush=True)
    return None


def run_grid(family: str, profile: str) -> Path:
    """Run all OUTER_CELLS × binary descent on max_num_seqs for (family, profile)."""
    print(f"\n=== GRID: {family} / {profile} ===", flush=True)
    winners_per_outer: list[dict] = []
    for cell in OUTER_CELLS:
        gmu = cell["gpu_memory_utilization"]
        eager = cell["enforce_eager"]
        print(f"\n-- outer cell gmu={gmu} eager={eager} (binary descent on max_num_seqs) --", flush=True)
        rec = _binary_descent_max_seqs(family, profile, gmu, eager)
        if rec is not None:
            winners_per_outer.append(rec)
        else:
            print(f"-- outer cell gmu={gmu} eager={eager} FAILED at all max_num_seqs --", flush=True)

    summary_path = RESULTS_DIR / f"_summary_{family}__{profile}.json"
    winners_per_outer.sort(key=lambda r: -(r.get("tokens_per_sec") or 0))
    summary_path.write_text(json.dumps({
        "cells_ranked": winners_per_outer,
        "outer_grid": OUTER_CELLS,
        "max_seqs_probe": MAX_SEQS_PROBE,
    }, indent=2))
    if winners_per_outer:
        w = winners_per_outer[0]
        print(
            f"\n=== {family} / {profile} winner: gmu={w['cell']['gpu_memory_utilization']}, "
            f"max_seqs={w['cell']['max_num_seqs']}, eager={w['cell']['enforce_eager']} "
            f"-> {w['tokens_per_sec']} tok/s, vram={w['peak_vram_total_gb']} GB",
            flush=True,
        )
    return summary_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--single", nargs=2, metavar=("FAMILY", "PROFILE"), help="run a single config (subprocess mode)")
    ap.add_argument("--gmu", type=float)
    ap.add_argument("--max-seqs", type=int)
    ap.add_argument("--enforce-eager", action="store_true")
    ap.add_argument("--grid", nargs=2, metavar=("FAMILY", "PROFILE"))
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()

    if args.single:
        family, profile = args.single
        if args.gmu is None or args.max_seqs is None:
            ap.error("--single requires --gmu and --max-seqs")
        _run_single(family, profile, args.gmu, args.max_seqs, args.enforce_eager)
        return

    if args.grid:
        family, profile = args.grid
        run_grid(family, profile)
        return

    if args.all:
        plan = [
            ("gemma_2_27b", "short"), ("gemma_2_27b", "long"),
            ("qwen_3_32b", "short"), ("qwen_3_32b", "long"),
            ("gemma_4_31b", "short"), ("gemma_4_31b", "long"),
            ("qwen_3_6_27b", "judge"),
        ]
        for fam, prof in plan:
            run_grid(fam, prof)
        return

    ap.error("must pass --single, --grid, or --all")


if __name__ == "__main__":
    main()
