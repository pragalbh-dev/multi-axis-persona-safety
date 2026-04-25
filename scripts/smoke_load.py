"""Smoke-test model loading via vLLM on 2x RTX 5090 (GPUs 2,3, TP=2).

Stage 0 T0.4 / T0.5 / T0.9. For each subject or judge entry in
configs/subjects.yaml, this script:
  1. Loads the quantized model with vLLM
  2. Runs a batch of 5 test prompts via LLM.generate()
  3. Records VRAM peak, tokens/sec, first-token latency per run
  4. Unloads cleanly (delete LLM, torch.cuda.empty_cache, gc)

Output: results/stage_0_smoke/<family>.json per run, plus a consolidated
summary printed at the end. Records feed the CONVENTIONS.md
"Batch size & TP per model" section and decisions.md per subject.

Usage:
    uv run python scripts/smoke_load.py --family qwen_3_6_27b
    uv run python scripts/smoke_load.py --all
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils import env  # noqa: F401  pins CUDA_VISIBLE_DEVICES=0,1,2,3

import torch  # noqa: E402
import yaml  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "results" / "stage_0_smoke"
SUBJECTS_YAML = PROJECT_ROOT / "configs" / "subjects.yaml"

TEST_PROMPTS = [
    "Explain the difference between accuracy and precision in one paragraph.",
    "Write a haiku about activation vectors.",
    "What is the Cauchy-Schwarz inequality, and why is it useful?",
    "List three common causes of false positives in binary classification.",
    "In two sentences, describe what a transformer residual stream carries.",
]

MAX_OUT_TOKENS = 128


def _load_subjects() -> dict[str, dict[str, Any]]:
    return yaml.safe_load(SUBJECTS_YAML.read_text())


def _build_prompts(tokenizer, prompts: list[str], chat_template_kwargs: dict) -> list[str]:
    """Apply chat template to each prompt. Returns list of formatted strings."""
    out = []
    for p in prompts:
        conv = [{"role": "user", "content": p}]
        s = tokenizer.apply_chat_template(
            conv, tokenize=False, add_generation_prompt=True, **chat_template_kwargs,
        )
        out.append(s)
    return out


def smoke_one(family: str, cfg: dict, thinking_variant: str | None = None) -> dict:
    """Load, run 5 prompts, unload. Returns a stats dict."""
    from vllm import LLM, SamplingParams

    # Attention backend override
    if cfg.get("attention_backend"):
        os.environ["VLLM_ATTENTION_BACKEND"] = cfg["attention_backend"]
    else:
        os.environ.pop("VLLM_ATTENTION_BACKEND", None)

    # Effective chat template kwargs (thinking ON/OFF variant overrides)
    ctk = dict(cfg.get("chat_template_kwargs") or {})
    if thinking_variant == "on":
        ctk["enable_thinking"] = True
    elif thinking_variant == "off":
        ctk["enable_thinking"] = False

    run_label = family + (f"__thinking_{thinking_variant}" if thinking_variant else "")
    print(f"\n=== {run_label} ===")
    print(f"  hf_id: {cfg['hf_id']}")
    print(f"  TP: {cfg['tensor_parallel_size']}  attn: {os.environ.get('VLLM_ATTENTION_BACKEND', 'default')}")
    print(f"  trust_remote_code: {cfg.get('trust_remote_code', False)}  chat_template_kwargs: {ctk}")

    # Use pynvml for driver-level VRAM (cross-process — torch.max_memory_allocated
    # reports 0 when vLLM uses TP>1 subprocesses).
    try:
        import pynvml
        pynvml.nvmlInit()
        # CUDA_VISIBLE_DEVICES (e.g. "0,1,2,3") maps to local torch indices 0..N-1
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
        nvml_handles = [pynvml.nvmlDeviceGetHandleByIndex(int(idx)) for idx in visible if idx.strip()]
    except Exception:
        nvml_handles = []

    def _read_vram_gb() -> list[float]:
        if not nvml_handles:
            return []
        return [pynvml.nvmlDeviceGetMemoryInfo(h).used / 1e9 for h in nvml_handles]

    baseline_vram = _read_vram_gb()

    t0 = time.time()
    llm = LLM(
        model=cfg["hf_id"],
        tensor_parallel_size=cfg["tensor_parallel_size"],
        gpu_memory_utilization=cfg.get("gpu_memory_utilization", 0.85),
        trust_remote_code=cfg.get("trust_remote_code", False),
        max_model_len=cfg.get("max_model_len", 2048),
        enforce_eager=cfg.get("enforce_eager", False),
        seed=42,
    )
    t_load = time.time() - t0
    print(f"  loaded in {t_load:.1f}s")

    tokenizer = llm.get_tokenizer()
    prompts = _build_prompts(tokenizer, TEST_PROMPTS, ctk)
    sp = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=MAX_OUT_TOKENS, seed=42)

    t1 = time.time()
    outputs = llm.generate(prompts, sp)
    t_gen = time.time() - t1

    # Peak VRAM measured immediately after gen, via NVML driver readings
    peak_vram = _read_vram_gb()
    peak_per_gpu = [round(p - b, 3) for p, b in zip(peak_vram, baseline_vram)] if baseline_vram else []

    out_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    toks_per_sec = out_tokens / t_gen if t_gen > 0 else float("nan")

    sample = outputs[0].outputs[0].text[:200].replace("\n", " ")
    print(f"  gen {out_tokens} tok / {t_gen:.2f}s = {toks_per_sec:.1f} tok/s")
    print(f"  peak VRAM per GPU: {[round(p,1) for p in peak_per_gpu]} GB  (total {sum(peak_per_gpu):.1f} GB)")
    print(f"  sample output[0]: {sample!r}")

    stats = {
        "family": family,
        "thinking_variant": thinking_variant,
        "hf_id": cfg["hf_id"],
        "tensor_parallel_size": cfg["tensor_parallel_size"],
        "attention_backend": os.environ.get("VLLM_ATTENTION_BACKEND", "default"),
        "trust_remote_code": cfg.get("trust_remote_code", False),
        "chat_template_kwargs": ctk,
        "load_seconds": round(t_load, 2),
        "gen_seconds": round(t_gen, 2),
        "output_tokens": out_tokens,
        "tokens_per_sec": round(toks_per_sec, 2),
        "peak_vram_per_gpu_gb": [round(p, 3) for p in peak_per_gpu],
        "peak_vram_total_gb": round(sum(peak_per_gpu), 3),
        "sample_output": sample,
    }

    # Unload
    del llm
    gc.collect()
    torch.cuda.empty_cache()

    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", help="Run a single family from configs/subjects.yaml")
    ap.add_argument("--all", action="store_true", help="Run every family in subjects.yaml")
    args = ap.parse_args()

    subjects = _load_subjects()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.family:
        families = [args.family]
    elif args.all:
        families = list(subjects.keys())
    else:
        ap.error("--family or --all required")

    summary = []
    for fam in families:
        cfg = subjects[fam]
        variants: list[str | None]
        if cfg.get("thinking_modes"):
            variants = cfg["thinking_modes"]
        else:
            variants = [None]
        for v in variants:
            v_str = str(v) if v is not None else None  # defensive against YAML on/off -> bool
            stats = smoke_one(fam, cfg, thinking_variant=v_str)
            suffix = f"__thinking_{v_str}" if v_str else ""
            out_file = OUT_DIR / f"{stats['family']}{suffix}.json"
            out_file.write_text(json.dumps(stats, indent=2))
            print(f"  wrote {out_file}")
            summary.append(stats)

    print("\n=== summary ===")
    for s in summary:
        print(f"  {s['family']}{'('+s['thinking_variant']+')' if s['thinking_variant'] else ''}: "
              f"load={s['load_seconds']}s gen={s['gen_seconds']}s "
              f"tok/s={s['tokens_per_sec']} vram_total={s['peak_vram_total_gb']} GB")


if __name__ == "__main__":
    main()
