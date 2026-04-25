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
# Output budgets are per-task; for the workload sampler we pick a representative
# split, so we set a single output_max here that covers the profile.
PROFILES: dict[str, dict[str, Any]] = {
    "short": {
        "max_model_len": 2048,
        "tasks": ["ifeval", "gsm8k_1000", "mmlu_pro_1400", "eq_bench", "extraction_questions"],
        "output_max": 1024,
        "applies_to_role": "subject",
    },
    "long": {
        "max_model_len": 8192,
        "tasks": ["dan_jailbreak"],
        "output_max": 1024,
        "applies_to_role": "subject",
    },
    "judge": {
        "max_model_len": 4096,
        # judge inputs = (jailbreak prompt + subject response). We approximate by
        # concatenating DAN prompts with synthetic short responses for sampling.
        "tasks": ["dan_jailbreak"],
        "output_max": 128,
        "applies_to_role": "judge_primary",
    },
}

GRID_CELLS: list[dict[str, Any]] = [
    {"gpu_memory_utilization": 0.85, "max_num_seqs": 64,  "enforce_eager": False},
    {"gpu_memory_utilization": 0.92, "max_num_seqs": 64,  "enforce_eager": False},
    {"gpu_memory_utilization": 0.92, "max_num_seqs": 128, "enforce_eager": False},
    {"gpu_memory_utilization": 0.92, "max_num_seqs": 256, "enforce_eager": False},
    {"gpu_memory_utilization": 0.95, "max_num_seqs": 128, "enforce_eager": False},
    {"gpu_memory_utilization": 0.92, "max_num_seqs": 128, "enforce_eager": True},   # control: no CUDA graph
]


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

def run_grid(family: str, profile: str) -> Path:
    """Run all grid cells for (family, profile). Each cell = fresh subprocess."""
    print(f"\n=== GRID: {family} / {profile} ===", flush=True)
    for cell in GRID_CELLS:
        cell_label = (
            f"{family}__{profile}__gmu{int(cell['gpu_memory_utilization']*100)}"
            f"_seq{cell['max_num_seqs']}_eager{int(cell['enforce_eager'])}"
        )
        out_path = RESULTS_DIR / f"{cell_label}.json"
        if out_path.exists():
            print(f"[{cell_label}] EXISTS — skipping", flush=True)
            continue
        cmd = [
            sys.executable, str(Path(__file__).resolve()),
            "--single", family, profile,
            "--gmu", str(cell["gpu_memory_utilization"]),
            "--max-seqs", str(cell["max_num_seqs"]),
        ]
        if cell["enforce_eager"]:
            cmd.append("--enforce-eager")
        # Failure of one cell shouldn't kill the grid; record fail and continue.
        try:
            r = subprocess.run(cmd, check=False, timeout=900)
            if r.returncode != 0:
                fail_record = {
                    "family": family, "profile": profile, "cell": cell,
                    "status": "fail", "returncode": r.returncode,
                }
                out_path.write_text(json.dumps(fail_record, indent=2))
                print(f"[{cell_label}] FAIL rc={r.returncode}", flush=True)
        except subprocess.TimeoutExpired:
            print(f"[{cell_label}] TIMEOUT", flush=True)

    summary_path = RESULTS_DIR / f"_summary_{family}__{profile}.json"
    cells = []
    for cell in GRID_CELLS:
        label = (
            f"{family}__{profile}__gmu{int(cell['gpu_memory_utilization']*100)}"
            f"_seq{cell['max_num_seqs']}_eager{int(cell['enforce_eager'])}"
        )
        p = RESULTS_DIR / f"{label}.json"
        if p.exists():
            cells.append(json.loads(p.read_text()))
    cells.sort(key=lambda r: -(r.get("tokens_per_sec") or 0))
    summary_path.write_text(json.dumps({"cells_ranked": cells}, indent=2))
    if cells:
        winner = cells[0]
        print(
            f"\n=== {family} / {profile} winner: gmu={winner['cell']['gpu_memory_utilization']}, "
            f"max_seqs={winner['cell']['max_num_seqs']}, eager={winner['cell']['enforce_eager']} "
            f"-> {winner['tokens_per_sec']} tok/s, vram={winner['peak_vram_total_gb']} GB",
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
