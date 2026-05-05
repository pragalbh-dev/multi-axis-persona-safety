"""Subprocess work-module: subject-phase generation.

Three backends:
  --backend vllm    (default for unsteered runs)  → vLLM continuous batching
                                                    via configs/inference_runtime.yaml profile.
  --backend hf      (mandatory for steered Gemma 4) → HF + accelerate device_map="auto",
                                                    ActivationSteering context manager
                                                    wrapping model.generate.
  --backend sglang  (default for steered Gemma 2 27B + Qwen 3 32B post-2026-04-30 spike)
                                                  → SGLang server with --forward-hooks JSON
                                                    pointing at src.steering.sglang_hook_factories.

vLLM cannot fire register_forward_hook on its inference path. SGLang gives
~1.84x per-token throughput vs HF on Gemma 2 27B (spike 2026-04-30); we keep
HF as the safe fallback for any subject SGLang can't serve cleanly (currently
only Gemma 4 31B due to FlashInfer rmsnorm shape bug on 0.5.10). Per-subject
backend opt-in lives in `configs/subjects.yaml::<subject>.steered_backend`.

See `plans/sglang_post_plan_b_spike.md` for the spike + verdict.

Args JSON schema:
{
  "model_id":     str,                    # key in configs/subjects.yaml + inference_runtime.yaml
  "backend":      "vllm"|"hf"|"sglang",   # caller picks: vllm for unsteered, hf|sglang for steered
                                          # per cfg.subjects[id].steered_backend
  "profile":      "short"|"long",         # only used by vllm backend; ignored by hf/sglang
  "prompts_path": str,                    # parquet of (prompt_id, dataset, input_text)
  "output_path":  str,                    # parquet of (prompt_id, dataset, condition_id, ..., response_text)
  "condition_id": str,                    # tag this generation pass with a condition (e.g. "baseline", "aa_cap_pc2_pos2")
  "seed":         int,
  "max_new_tokens": int,
  "temperature":  float,
  "sglang_port":  int,                    # optional, default from inference_runtime.sglang_defaults
  # Steering (hf or sglang backend):
  "steering": {
    "mode":          "none"|"addition"|"capping"|"compound",
    # Single-mode keys (addition / capping):
    "vectors":       list[str],           # paths to .safetensors direction tensors
                                          # (sglang accepts inline tensors too — they get re-saved)
    "coefficients":  list[float],         # one per vector (addition)
    "cap_thresholds": list[float],        # one per vector (capping)
    "layers":        list[int]|list[list[int]],  # ints for addition; [start,end] ranges for capping
    # Compound-mode keys:
    "cap_vectors":   list[str], "cap_thresholds": list[float], "cap_layers": list[int]|list[list[int]],
    "addition_vectors": list[str], "addition_coefficients": list[float], "addition_layers": list[int],
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
    from vllm import LLM, SamplingParams

    from src.utils.config import load_inference_runtime

    rt = load_inference_runtime()
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
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from src.utils.config import load_subjects

    model_id_key = args["model_id"]
    subjects = load_subjects()
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


def _save_vector_to_safetensors(vec, out_path: Path) -> Path:
    """SGLang --forward-hooks JSON is path-based; serialize a torch tensor to
    a safetensors file with the canonical 'vector' key.

    The hook factory's `_load_vector` (sglang_hook_factories.py:54) reads
    1-D tensors under key 'vector' — we match that contract.
    """
    import torch
    from safetensors.torch import save_file

    if not isinstance(vec, torch.Tensor):
        vec = torch.as_tensor(vec)
    if vec.ndim != 1:
        vec = vec.reshape(-1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_file({"vector": vec.contiguous().to(dtype=torch.float32, device="cpu")}, str(out_path))
    return out_path


def _materialize_steering_vectors(steering: dict, scratch_dir: Path) -> dict:
    """Return a copy of `steering` with all vector references resolved to
    safetensors paths under `scratch_dir`. Inline tensors / PT files are
    re-saved; existing safetensors paths are kept as-is.

    SGLang --forward-hooks needs path-based vectors because the JSON spec is
    serialized at server launch time.
    """
    import os

    out = dict(steering)
    scratch_dir.mkdir(parents=True, exist_ok=True)

    def _resolve_list(key: str, prefix: str) -> None:
        if key not in out or not out[key]:
            return
        new_paths: list[str] = []
        for i, item in enumerate(out[key]):
            if isinstance(item, str) and item.endswith(".safetensors") and os.path.exists(item):
                new_paths.append(item)
                continue
            # Anything else: load → save under scratch_dir
            tensor = _load_vec(item) if isinstance(item, str) else item
            target = scratch_dir / f"{prefix}_{i}.safetensors"
            _save_vector_to_safetensors(tensor, target)
            new_paths.append(str(target))
        out[key] = new_paths

    _resolve_list("vectors", "v")
    _resolve_list("cap_vectors", "cap_v")
    _resolve_list("addition_vectors", "add_v")
    return out


def _build_forward_hooks_json(steering: dict, layer_module_glob_prefix: str) -> list[dict]:
    """Translate a SteeringSubConfig dict into SGLang --forward-hooks JSON list.

    Mode → factory mapping:
      addition  → addition_factory          (one hook per (vector, layer) pair)
      capping   → capping_factory           (with negate_vector=True per from_config)
      compound  → two hooks per layer       (cap entry first, steer entry second —
                                             matches PyTorch hook-firing order =
                                             same as src/steering/steerer.py:cap_and_steer)
      multi_axis_cap → multi_axis_cap_factory  (one hook per layer, all axes inside)

    Vectors must already be path-resolved via _materialize_steering_vectors.
    """
    mode = steering.get("mode", "none")
    if mode == "none":
        return []

    positions = steering.get("positions", "all")

    def _target_for(layer: int) -> str:
        return f"{layer_module_glob_prefix}.{int(layer)}"

    if mode == "addition":
        vps = steering["vectors"]
        coeffs = steering["coefficients"]
        layers = steering["layers"]
        if len(vps) != len(coeffs) or len(vps) != len(layers):
            raise ValueError("addition: vectors/coefficients/layers length mismatch")
        hooks = []
        for i, (vp, c, L) in enumerate(zip(vps, coeffs, layers)):
            hooks.append({
                "name": f"addition_{i}",
                "target_modules": [_target_for(L)],
                "hook_factory": "src.steering.sglang_hook_factories:addition_factory",
                "config": {"vector_path": vp, "coefficient": float(c), "positions": positions},
            })
        return hooks

    if mode == "capping":
        vps = steering["vectors"]
        thresholds = steering["cap_thresholds"]
        layers = steering["layers"]
        if len(vps) != len(thresholds):
            raise ValueError("capping: vectors/cap_thresholds length mismatch")
        # `layers` is a list of [start, end] ranges per vector — expand to the union.
        hooks = []
        for i, (vp, tau) in enumerate(zip(vps, thresholds)):
            entry = layers[i]
            if isinstance(entry, int):
                rng = (entry, entry)
            else:
                rng = (int(entry[0]), int(entry[1]))
            for L in range(rng[0], rng[1] + 1):
                hooks.append({
                    "name": f"capping_{i}_L{L}",
                    "target_modules": [_target_for(L)],
                    "hook_factory": "src.steering.sglang_hook_factories:capping_factory",
                    "config": {
                        "vector_path": vp, "tau": float(tau),
                        "positions": positions, "negate_vector": True,
                    },
                })
        return hooks

    if mode == "compound":
        # cap (range per vector) + addition (single layer per vector). Order matters:
        # cap hooks first → steer hooks second, matching cap_and_steer in HF.
        cap_vps = steering["cap_vectors"]
        cap_taus = steering["cap_thresholds"]
        cap_layers = steering["cap_layers"]
        add_vps = steering["addition_vectors"]
        add_coeffs = steering["addition_coefficients"]
        add_layers = steering["addition_layers"]

        hooks: list[dict] = []
        for i, (vp, tau, L) in enumerate(zip(cap_vps, cap_taus, cap_layers)):
            # compound mode currently uses single-layer caps in plan_b
            entry = L
            if isinstance(entry, list):
                rng = (int(entry[0]), int(entry[1]))
            else:
                rng = (int(entry), int(entry))
            for layer in range(rng[0], rng[1] + 1):
                hooks.append({
                    "name": f"compound_cap_{i}_L{layer}",
                    "target_modules": [_target_for(layer)],
                    "hook_factory": "src.steering.sglang_hook_factories:capping_factory",
                    "config": {
                        "vector_path": vp, "tau": float(tau),
                        "positions": positions, "negate_vector": True,
                    },
                })
        for i, (vp, c, L) in enumerate(zip(add_vps, add_coeffs, add_layers)):
            hooks.append({
                "name": f"compound_steer_{i}",
                "target_modules": [_target_for(L)],
                "hook_factory": "src.steering.sglang_hook_factories:addition_factory",
                "config": {"vector_path": vp, "coefficient": float(c), "positions": positions},
            })
        return hooks

    raise ValueError(f"Unsupported steering mode for SGLang backend: {mode}")


def _wait_for_sglang_ready(log_path: Path, marker: str, error_patterns: tuple[str, ...],
                           timeout_seconds: int) -> None:
    """Tail `log_path` until `marker` appears or any error pattern shows up.

    Avoids the polling/sleep antipattern by reading new bytes on each iteration.
    Raises RuntimeError on error pattern hit, TimeoutError if the marker never
    arrives within the budget.
    """
    import re

    deadline = time.time() + timeout_seconds
    err_re = re.compile("|".join(error_patterns))
    seen = ""
    while time.time() < deadline:
        if log_path.exists():
            seen = log_path.read_text(errors="ignore")
            if marker in seen:
                return
            m = err_re.search(seen)
            if m:
                tail = "\n".join(seen.splitlines()[-30:])
                raise RuntimeError(f"SGLang server failed: matched {m.group(0)!r}\n--- log tail ---\n{tail}")
        time.sleep(2)
    tail = "\n".join(seen.splitlines()[-30:]) if seen else "<empty>"
    raise TimeoutError(f"SGLang did not reach {marker!r} within {timeout_seconds}s\n--- log tail ---\n{tail}")


def _run_sglang(args: dict) -> tuple[int, float]:
    """Steered/capped generation via SGLang --forward-hooks.

    Server lifecycle:
      1. Materialize steering vectors → safetensors under scratch_dir.
      2. Build --forward-hooks JSON list from the steering dict.
      3. Launch sglang.launch_server with --forward-hooks <json>, wait for
         "Application startup complete" in the launch log.
      4. Render chat-templated prompts, fire concurrent /v1/chat/completions
         requests over aiohttp (concurrency = sglang_defaults.request_concurrency).
      5. Tear down server (kill PID, sleep 5).

    Returns (n_rows, tokens_per_second). Token count uses
    `usage.completion_tokens` from each response.
    """
    import asyncio
    import os
    import shutil
    import signal
    import subprocess

    import aiohttp
    import pandas as pd
    from transformers import AutoTokenizer

    from src.utils.config import load_inference_runtime, load_subjects

    model_id_key = args["model_id"]
    subjects = load_subjects()
    rt = load_inference_runtime()
    if model_id_key not in subjects:
        raise KeyError(f"unknown model_id={model_id_key!r}")

    sub_cfg = subjects[model_id_key]
    hf_id = sub_cfg["hf_id"]
    layer_glob_prefix = sub_cfg.get("layer_module_glob_prefix", "model.layers")
    chat_template_kwargs = sub_cfg.get("chat_template_kwargs") or {}
    trust_remote_code = bool(sub_cfg.get("trust_remote_code", False))

    sgl_defaults = rt.get("sglang_defaults", {})
    port = int(args.get("sglang_port", sgl_defaults.get("port", 30000)))
    tp = int(sgl_defaults.get("tp", 1))
    dtype = sgl_defaults.get("dtype", "bfloat16")
    mem_frac = float(sgl_defaults.get("mem_fraction_static", 0.85))
    attn_backend = sgl_defaults.get("attention_backend", "triton")
    disable_pcg = bool(sgl_defaults.get("disable_piecewise_cuda_graph", True))
    startup_marker = sgl_defaults.get("startup_log_marker", "Application startup complete")
    startup_timeout = int(sgl_defaults.get("startup_timeout_seconds", 600))
    concurrency = int(sgl_defaults.get("request_concurrency", 32))
    request_timeout = int(sgl_defaults.get("request_timeout_seconds", 600))
    extra_env = dict(sgl_defaults.get("env") or {})

    # Scratch dir for vectors + log.
    output_path = Path(args["output_path"])
    scratch_dir = output_path.parent / f".sglang_scratch_{output_path.stem}"
    scratch_dir.mkdir(parents=True, exist_ok=True)
    log_path = scratch_dir / "server.log"
    if log_path.exists():
        log_path.unlink()

    steering = args.get("steering") or {"mode": "none"}
    steering = _materialize_steering_vectors(steering, scratch_dir / "vectors")
    forward_hooks = _build_forward_hooks_json(steering, layer_glob_prefix)
    if not forward_hooks:
        # Sanity: SGLang for unsteered makes no sense — caller should use vllm.
        raise ValueError("SGLang backend invoked with no steering hooks; route to vllm/hf instead")

    forward_hooks_json = json.dumps(forward_hooks)
    forward_hooks_file = scratch_dir / "forward_hooks.json"
    forward_hooks_file.write_text(forward_hooks_json)

    # Build server launch command. Use sys.executable so the SGLang env's
    # python is inherited (sglang.launch_server only resolves in .venv-sglang).
    import sys as _sys
    cmd = [
        _sys.executable, "-m", "sglang.launch_server",
        "--model-path", hf_id,
        "--tp", str(tp),
        "--dtype", dtype,
        "--port", str(port),
        "--mem-fraction-static", str(mem_frac),
        "--attention-backend", attn_backend,
        "--forward-hooks", forward_hooks_json,
    ]
    if disable_pcg:
        cmd.append("--disable-piecewise-cuda-graph")
    if trust_remote_code:
        cmd.append("--trust-remote-code")

    env = os.environ.copy()
    env.update({k: str(v) for k, v in extra_env.items()})
    # Ensure nvcc is on PATH for sgl_kernel JIT (sm_120 has no precompiled wheels)
    # AND the venv's bin dir is on PATH so ninja / cuda tools resolve when SGLang
    # spawns its own grandchild compile processes.
    cuda_bin = "/usr/local/cuda/bin"
    venv_bin = str(Path(_sys.executable).parent)
    path_parts = env.get("PATH", "").split(":")
    for needed in (venv_bin, cuda_bin):
        if needed not in path_parts:
            path_parts.insert(0, needed)
    env["PATH"] = ":".join(path_parts)

    # Launch detached (setsid), redirect stdout+stderr to log_path.
    print(f"[sglang] launching server: {' '.join(cmd[:6])} ... (hooks: {len(forward_hooks)})")
    with log_path.open("w") as logf:
        proc = subprocess.Popen(
            cmd, stdout=logf, stderr=subprocess.STDOUT, env=env,
            start_new_session=True,
        )

    server_pid = proc.pid
    try:
        _wait_for_sglang_ready(
            log_path,
            marker=startup_marker,
            error_patterns=("Traceback", "RuntimeError", "AssertionError", "FAILED", "OOM"),
            timeout_seconds=startup_timeout,
        )
        print(f"[sglang] server ready on port {port} (pid {server_pid})")

        # Tokenize prompts using the model's chat template.
        tok = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=trust_remote_code)
        prompts_df = pd.read_parquet(args["prompts_path"])
        rows = prompts_df.to_dict("records")

        chat_prompts = []
        for r in rows:
            msgs = [{"role": "user", "content": r["input_text"]}]
            s = tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True, **chat_template_kwargs,
            )
            assert isinstance(s, str)
            chat_prompts.append(s)

        max_new = int(args["max_new_tokens"])
        temperature = float(args.get("temperature", 0.0))
        seed = int(args["seed"])

        async def _one_request(session, prompt_text):
            payload = {
                "model": "default",
                "messages": [{"role": "user", "content": prompt_text}],
                "max_tokens": max_new,
                "temperature": temperature,
                "top_p": 1.0,
                "seed": seed,
            }
            async with session.post(
                f"http://127.0.0.1:{port}/v1/chat/completions",
                json=payload, timeout=request_timeout,
            ) as r:
                d = await r.json()
                if "choices" not in d:
                    raise RuntimeError(f"SGLang error: {d}")
                text = d["choices"][0]["message"]["content"]
                usage = d.get("usage", {})
                ntok = int(usage.get("completion_tokens", 0))
                return text, ntok

        async def _all_requests():
            connector = aiohttp.TCPConnector(limit=concurrency)
            timeout = aiohttp.ClientTimeout(total=request_timeout)
            async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
                tasks = [_one_request(session, p) for p in [r["input_text"] for r in rows]]
                return await asyncio.gather(*tasks)

        t0 = time.time()
        results = asyncio.run(_all_requests())
        gen_seconds = time.time() - t0

        responses = [text for text, _ in results]
        token_counts = [n for _, n in results]
        total_new_tokens = sum(token_counts)
    finally:
        # Tear down server cleanly.
        try:
            os.killpg(os.getpgid(server_pid), signal.SIGTERM)
            time.sleep(2)
            os.killpg(os.getpgid(server_pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        # Also kill any straggler processes from this launch.
        subprocess.run(["pkill", "-9", "-f", f"sglang.launch_server.*--port {port}"],
                       check=False, capture_output=True)
        time.sleep(3)

    out = prompts_df.copy()
    out["condition_id"] = args["condition_id"]
    out["response_text"] = responses
    out["response_tokens"] = token_counts
    out["model_id"] = args["model_id"]
    out["seed"] = args["seed"]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output_path, index=False)

    # Clean up scratch (vectors + log) once results are persisted.
    if not args.get("keep_sglang_scratch", False):
        shutil.rmtree(scratch_dir, ignore_errors=True)

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
    elif backend == "sglang":
        n_rows, tps = _run_sglang(args)
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
