"""Subprocess work-module: subject-phase generation.

Two backends:
  --backend vllm  (default for unsteered runs)  → vLLM continuous batching
                                                  via configs/inference_runtime.yaml profile.
  --backend hf    (mandatory for steered/capped) → HF + accelerate device_map="auto",
                                                  ActivationSteering context manager
                                                  wrapping model.generate.

vLLM cannot fire register_forward_hook on its inference path; HF is the only
trusted route for steered/capped generation in Plan B. SGLang --forward-hooks
is the post-Plan B alternative; see plans/sglang_post_plan_b_spike.md.

Args JSON schema:
{
  "model_id":     str,            # key in configs/subjects.yaml + inference_runtime.yaml
  "backend":      "vllm"|"hf",    # explicit; caller picks based on cfg.steering.mode
  "profile":      "short"|"long", # only used by vllm backend; ignored by hf
  "prompts_path": str,            # parquet of (prompt_id, dataset, input_text)
  "output_path":  str,            # parquet of (prompt_id, dataset, condition_id, ..., response_text)
  "condition_id": str,            # tag this generation pass with a condition (e.g. "baseline", "aa_cap_pc2_pos2")
  "seed":         int,
  "max_new_tokens": int,
  "temperature":  float,
  # Steering (hf backend only):
  "steering": {
    "mode":          "none"|"capping"|"addition",
    "vectors":       list[str],   # paths to .safetensors direction tensors
    "coefficients":  list[float], # one per vector
    "cap_thresholds": list[float],# one per vector (capping mode only)
    "layers":        list[int]|list[list[int]],  # ints for addition; list of [start,end] for capping
    "positions":     "all"|"last"
  } | None
}

Output JSON:
{
  "status": "ok",
  "elapsed_seconds": float,
  "peak_vram_per_gpu": [4 floats],
  "artifacts": [output_path],
  "n_rows": int,
  "tokens_per_second": float
}
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from src.utils import env as _env  # noqa: F401


def _run_vllm(args: dict) -> tuple[int, float]:
    """Unsteered vLLM generation. Returns (n_rows_written, tokens_per_sec)."""
    import pandas as pd
    import yaml
    from vllm import LLM, SamplingParams

    rt = yaml.safe_load(Path("configs/inference_runtime.yaml").read_text())
    model_cfg = rt[args["model_id"]]
    profile = model_cfg["profiles"][args["profile"]]
    chat_template_kwargs = model_cfg.get("chat_template_kwargs") or {}

    llm = LLM(
        model=model_cfg["hf_id"],
        tensor_parallel_size=model_cfg["tensor_parallel_size"],
        gpu_memory_utilization=profile["gpu_memory_utilization"],
        max_model_len=profile["max_model_len"],
        max_num_seqs=profile["max_num_seqs"],
        enable_prefix_caching=profile.get("enable_prefix_caching", False),
        trust_remote_code=model_cfg.get("trust_remote_code", False),
        enforce_eager=False,
        seed=args["seed"],
    )
    tok = llm.get_tokenizer()

    prompts_df = pd.read_parquet(args["prompts_path"])
    chat_prompts = []
    for _, r in prompts_df.iterrows():
        msgs = [{"role": "user", "content": r["input_text"]}]
        s = tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, **chat_template_kwargs
        )
        assert isinstance(s, str)
        chat_prompts.append(s)

    sp = SamplingParams(
        temperature=args.get("temperature", 0.0),
        top_p=1.0,
        max_tokens=args["max_new_tokens"],
        seed=args["seed"],
    )
    t0 = time.time()
    outputs = llm.generate(chat_prompts, sp)
    gen_seconds = time.time() - t0

    responses = [o.outputs[0].text for o in outputs]
    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)

    out = prompts_df.copy()
    out["condition_id"] = args["condition_id"]
    out["response_text"] = responses
    out["response_tokens"] = [len(o.outputs[0].token_ids) for o in outputs]
    out["model_id"] = args["model_id"]
    out["seed"] = args["seed"]

    out_path = Path(args["output_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)

    tokens_per_sec = total_tokens / gen_seconds if gen_seconds > 0 else 0.0
    return len(out), round(tokens_per_sec, 2)


def _load_vec(path: str):
    """Load a single tensor from .safetensors (key 'v' preferred) or .pt."""
    import torch
    from safetensors.torch import load_file

    if path.endswith(".safetensors"):
        d = load_file(path)
        if "v" in d:
            return d["v"]
        return next(iter(d.values()))
    return torch.load(path, weights_only=True)


def _build_steering_contexts(model, steering: dict):
    """Return (cm_or_None) — a context manager applying the requested intervention.

    Modes:
      - "none"      → null cm (yields the model unchanged)
      - "addition"  → ActivationSteering(addition) over (vectors, coefficients, layers)
      - "capping"   → ActivationSteering(capping) over (vectors, cap_thresholds, layers)
      - "compound"  → cap_and_steer composing capping (cap_*) + addition (addition_*).
    """
    from contextlib import nullcontext

    import torch

    from src.steering.steerer import cap_and_steer
    from src.steering.types import SteeringConfig
    from src.steering.steerer import from_config

    mode = steering.get("mode", "none")
    if mode == "none":
        return nullcontext()

    if mode == "addition":
        vecs = [_load_vec(p) for p in steering["vectors"]]
        cfg = SteeringConfig(
            vectors=vecs,
            intervention_type="addition",
            layer_indices=[int(x) for x in steering["layers"]],
            coefficients=[float(c) for c in steering["coefficients"]],
            positions=steering.get("positions", "all"),
        )
        return from_config(model, cfg)

    if mode == "capping":
        vecs = [_load_vec(p) for p in steering["vectors"]]
        layers = steering["layers"]
        # Convert flat layer list to per-vector range-tuples (one layer == range)
        if isinstance(layers[0], int):
            ranges = [(int(L), int(L)) for L in layers]
        else:
            ranges = [(int(a), int(b)) for a, b in layers]
        cfg = SteeringConfig(
            vectors=vecs,
            intervention_type="capping",
            layer_indices=ranges,
            cap_thresholds=[float(t) for t in steering["cap_thresholds"]],
            positions=steering.get("positions", "all"),
        )
        return from_config(model, cfg)

    if mode == "compound":
        # Build cap_cfg
        cap_vecs = [_load_vec(p) for p in steering["cap_vectors"]]
        cap_layers = steering["cap_layers"]
        cap_ranges = [(int(L), int(L)) for L in cap_layers]
        cap_cfg = SteeringConfig(
            vectors=cap_vecs,
            intervention_type="capping",
            layer_indices=cap_ranges,
            cap_thresholds=[float(t) for t in steering["cap_thresholds"]],
            positions=steering.get("positions", "all"),
        )
        add_vecs = [_load_vec(p) for p in steering["addition_vectors"]]
        steer_cfg = SteeringConfig(
            vectors=add_vecs,
            intervention_type="addition",
            layer_indices=[int(x) for x in steering["addition_layers"]],
            coefficients=[float(c) for c in steering["addition_coefficients"]],
            positions=steering.get("positions", "all"),
        )
        return cap_and_steer(model, cap_cfg, steer_cfg)

    raise ValueError(f"Unknown steering mode: {mode}")


def _run_hf(args: dict) -> tuple[int, float]:
    """Steered/capped HF generation under ActivationSteering context.

    Uses external/assistant-axis::ActivationSteering via src/steering/steerer.py.
    Loaded with device_map="auto" + sdpa attention.
    """
    import pandas as pd
    import torch
    import yaml
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id_key = args["model_id"]
    subjects = yaml.safe_load(Path("configs/subjects.yaml").read_text())
    hf_id = subjects[model_id_key]["hf_id"]
    chat_template_kwargs = subjects[model_id_key].get("chat_template_kwargs") or {}
    trust_remote_code = subjects[model_id_key].get("trust_remote_code", False)

    tok = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=trust_remote_code)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    # CRITICAL: when chat-templated DAN prompts exceed max_input_len, truncate
    # from the LEFT — keeps the harmful question + generation marker (the actual
    # probe of the experiment), drops the persona preamble. truncation_side="right"
    # would cut the question and silently break the experiment.
    tok.truncation_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        hf_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        # NOTE: passing attn_implementation="sdpa" here is SILENTLY IGNORED on
        # Gemma 2 27B because its checkpoint config sets _attn_implementation_internal
        # to "eager" (legacy guard for older softcap-incompatible sdpa kernels).
        # We post-load force sdpa below — empirically verified to flip dispatch
        # (0 eager_attention_forward calls in a probed forward, 2026-04-25).
        # PyTorch 2.10 sdpa handles attn_logit_softcapping correctly.
        attn_implementation="sdpa",
        trust_remote_code=trust_remote_code,
    )
    # Force-flip per-layer dispatch. Verified empirically that this propagates
    # to every Gemma2Attention.config (single shared config object).
    model.config._attn_implementation = "sdpa"
    for sub in model.modules():
        if hasattr(sub, "config") and hasattr(sub.config, "_attn_implementation"):
            sub.config._attn_implementation = "sdpa"
    model.eval()

    steering = args.get("steering") or {"mode": "none"}
    prompts_df = pd.read_parquet(args["prompts_path"])
    batch_size = args.get("batch_size", 8)
    max_input_len = args.get("max_input_len", 2048)

    responses: list[str] = []
    response_token_counts: list[int] = []
    total_new_tokens = 0

    t0 = time.time()
    rows = prompts_df.to_dict("records")
    cm = _build_steering_contexts(model, steering)

    with cm:
        for batch_start in range(0, len(rows), batch_size):
            batch = rows[batch_start : batch_start + batch_size]
            chat_prompts = [
                tok.apply_chat_template(
                    [{"role": "user", "content": r["input_text"]}],
                    tokenize=False,
                    add_generation_prompt=True,
                    **chat_template_kwargs,
                )
                for r in batch
            ]
            inputs = tok(
                chat_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_input_len,
            ).to(next(model.parameters()).device)
            with torch.no_grad():
                gen = model.generate(
                    **inputs,
                    max_new_tokens=args["max_new_tokens"],
                    do_sample=args.get("temperature", 0.0) > 0,
                    temperature=args.get("temperature", 0.0) or 1.0,
                    pad_token_id=tok.eos_token_id,
                )
            prompt_len = inputs["input_ids"].shape[1]
            new_tokens = gen[:, prompt_len:]
            for i in range(new_tokens.size(0)):
                ids = new_tokens[i].tolist()
                while ids and ids[-1] == tok.pad_token_id:
                    ids.pop()
                text = tok.decode(ids, skip_special_tokens=True)
                responses.append(text)
                response_token_counts.append(len(ids))
                total_new_tokens += len(ids)

    gen_seconds = time.time() - t0

    out = prompts_df.copy()
    out["condition_id"] = args["condition_id"]
    out["response_text"] = responses
    out["response_tokens"] = response_token_counts
    out["model_id"] = args["model_id"]
    out["seed"] = args["seed"]

    out_path = Path(args["output_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)

    del model
    torch.cuda.empty_cache()

    tps = total_new_tokens / gen_seconds if gen_seconds > 0 else 0.0
    return len(out), round(tps, 2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--args-json", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    cli = parser.parse_args()

    args = json.loads(cli.args_json.read_text())
    t0 = time.time()

    backend = args.get("backend", "vllm")
    if backend == "vllm":
        n_rows, tps = _run_vllm(args)
    elif backend == "hf":
        n_rows, tps = _run_hf(args)
    else:
        raise ValueError(f"Unknown backend: {backend}")

    elapsed = time.time() - t0

    from src.utils.model_runner import peak_vram_per_gpu_gib, write_result

    write_result(
        cli.output,
        {
            "status": "ok",
            "elapsed_seconds": round(elapsed, 2),
            "peak_vram_per_gpu": peak_vram_per_gpu_gib(),
            "artifacts": [args["output_path"]],
            "n_rows": n_rows,
            "tokens_per_second": tps,
            "backend": backend,
            "condition_id": args["condition_id"],
        },
    )


if __name__ == "__main__":
    main()
