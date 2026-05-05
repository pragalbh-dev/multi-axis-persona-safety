"""Throughput rough gist: tokens/sec for HF vs SGLang on a fixed workload.

Single configuration per call, per spike doc Section "Acceptance criteria 5":
  100 prompts × 256 out tokens, bf16, T=0, batch=32 (HF) or default (SGLang),
  with optional single-axis cap τ=p25 (synthetic vector and synthetic τ for
  the spike — no real AA / data needed).

Usage:
  source .venv/bin/activate          # HF backend
  python scripts/bench_sglang_vs_hf.py --backend hf --model-path google/gemma-2-27b-it --condition unsteered

  source .venv-sglang/bin/activate   # SGLang backend (requires server already running)
  python scripts/bench_sglang_vs_hf.py --backend sglang --port 30000 --condition unsteered

Writes one row per call to results/sglang_spike/throughput.json.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch


_REPO = Path(__file__).resolve().parent.parent
_RESULTS = _REPO / "results" / "sglang_spike"
_OUT = _RESULTS / "throughput.json"

PROMPTS = None  # populated from data/eval/dan_jailbreak/sampled_1100.parquet
N_PROMPTS = 100
MAX_NEW_TOKENS = 256
BATCH_SIZE = 16          # 32 OOMs HF on 96 GB GPU with long DAN prompts
MAX_INPUT_LEN = 1024     # truncate long DAN prompts so HF KV cache fits


def _load_prompts(n: int) -> list[str]:
    import pandas as pd
    p = _REPO / "data" / "eval" / "dan_jailbreak" / "sampled_1100.parquet"
    if not p.exists():
        # Fallback for spike: synthetic prompts.
        return [f"Tell me about topic {i}." for i in range(n)]
    df = pd.read_parquet(p)
    # Prefer string columns in priority order.
    for col in ("full_prompt", "input_text", "question_text", "prompt"):
        if col in df.columns:
            return df[col].head(n).astype(str).tolist()
    # Fallback: first object-dtype column
    for col in df.columns:
        if df[col].dtype == "object":
            return df[col].head(n).astype(str).tolist()
    raise RuntimeError(f"No string prompt column in {p}")


def _record(row: dict) -> None:
    _RESULTS.mkdir(parents=True, exist_ok=True)
    if _OUT.exists():
        data = json.loads(_OUT.read_text())
    else:
        data = {"rows": []}
    data["rows"].append(row)
    _OUT.write_text(json.dumps(data, indent=2))
    print(f"[bench] appended row to {_OUT}: {row}")


def bench_hf(model_path: str, condition: str, layer: int = 22) -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from src.steering import SteeringConfig, from_config

    prompts = _load_prompts(N_PROMPTS)

    print(f"[bench:hf] loading {model_path} bf16…")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
    )
    model.eval()

    if condition == "capped":
        from safetensors.torch import load_file
        v = load_file(str(_REPO / "data" / "cache" / "sglang_spike_vectors" / "axis_0.safetensors"))["vector"]
        cfg = SteeringConfig(
            vectors=[v],
            intervention_type="capping",
            layer_indices=[(layer, layer)],
            cap_thresholds=[0.0],
        )
        ctx = from_config(model, cfg)
    elif condition == "unsteered":
        ctx = None
    else:
        raise ValueError(f"unknown condition: {condition}")

    # Render chat template once
    rendered = []
    for p in prompts:
        msgs = [{"role": "user", "content": p}]
        rendered.append(tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))

    total_new_tokens = 0
    t0 = time.time()

    def _run_batches():
        nonlocal total_new_tokens
        for i in range(0, len(rendered), BATCH_SIZE):
            chunk = rendered[i:i + BATCH_SIZE]
            inp = tokenizer(chunk, return_tensors="pt", padding=True, truncation=True, max_length=MAX_INPUT_LEN,
                            add_special_tokens=False).to(model.device)
            with torch.inference_mode():
                out = model.generate(
                    **inp,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                    temperature=1.0, top_p=1.0, top_k=0,
                )
            new_per_row = (out.shape[1] - inp["input_ids"].shape[1])
            total_new_tokens += new_per_row * out.shape[0]

    if ctx is None:
        _run_batches()
    else:
        with ctx:
            _run_batches()

    wall = time.time() - t0
    tps = total_new_tokens / wall if wall > 0 else 0.0
    _record({
        "backend": "hf",
        "model_path": model_path,
        "condition": condition,
        "n_prompts": len(prompts),
        "max_new_tokens": MAX_NEW_TOKENS,
        "batch_size": BATCH_SIZE,
        "total_new_tokens": total_new_tokens,
        "wall_seconds": round(wall, 2),
        "tokens_per_sec": round(tps, 2),
    })


def bench_sglang(model_path: str, condition: str, port: int) -> None:
    """SGLang server must already be running with the right --forward-hooks JSON."""
    import asyncio
    import aiohttp

    prompts = _load_prompts(N_PROMPTS)
    base = f"http://127.0.0.1:{port}"

    async def _run_one(session, prompt):
        payload = {
            "model": "default",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": MAX_NEW_TOKENS,
            "temperature": 0.0,
            "top_p": 1.0,
        }
        async with session.post(f"{base}/v1/chat/completions", json=payload, timeout=600) as r:
            d = await r.json()
            usage = d.get("usage", {})
            return usage.get("completion_tokens", 0)

    async def _run_all():
        connector = aiohttp.TCPConnector(limit=BATCH_SIZE)
        async with aiohttp.ClientSession(connector=connector) as session:
            tasks = [_run_one(session, p) for p in prompts]
            return await asyncio.gather(*tasks)

    t0 = time.time()
    new_tokens_list = asyncio.run(_run_all())
    wall = time.time() - t0
    total = int(sum(new_tokens_list))
    tps = total / wall if wall > 0 else 0.0
    _record({
        "backend": "sglang",
        "model_path": model_path,
        "condition": condition,
        "n_prompts": len(prompts),
        "max_new_tokens": MAX_NEW_TOKENS,
        "concurrency": BATCH_SIZE,
        "total_new_tokens": total,
        "wall_seconds": round(wall, 2),
        "tokens_per_sec": round(tps, 2),
    })


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--backend", required=True, choices=["hf", "sglang"])
    p.add_argument("--model-path", default="google/gemma-2-27b-it")
    p.add_argument("--condition", required=True, choices=["unsteered", "capped"])
    p.add_argument("--port", type=int, default=30000)
    p.add_argument("--layer", type=int, default=22)
    args = p.parse_args()

    if args.backend == "hf":
        bench_hf(args.model_path, args.condition, layer=args.layer)
    else:
        bench_sglang(args.model_path, args.condition, port=args.port)


if __name__ == "__main__":
    main()
