"""Spike test: HF + ActivationSteering vs SGLang + --forward-hooks equivalence.

Three-phase orchestrator (sequential, not parallel — both engines won't fit on
a single 96GB GPU at once):

  1. `--phase setup`   — generate fixed prompts, fixed random steering vectors
                         (one per axis), save to `data/cache/sglang_spike_vectors/`
                         and `results/sglang_spike/prompts.json`. Idempotent.
  2. `--phase hf --pattern <name>` — load HF model, apply pattern, generate
                         64 tokens per prompt at temperature=0, save token IDs
                         to `results/sglang_spike/hf_<pattern>.json`. Tear down.
  3. `--phase sglang --pattern <name>` — query a running SGLang server
                         (assumed launched with the matching --forward-hooks JSON),
                         generate 64 tokens per prompt at temperature=0, save
                         to `results/sglang_spike/sglang_<pattern>.json`.
  4. `--phase compare`  — read all paired files, write
                         `results/sglang_spike/equivalence.json` with per-cell
                         token-id match length and first-divergence index.

Patterns: unsteered, addition_pos, addition_neg, capping, cap_and_steer, multi_axis_cap.

Usage example (Gemma 2 27B):
  # 1. Setup once
  source .venv-sglang/bin/activate
  python -m tests.integration.sglang_hooks_smoke --phase setup --hidden-size 4608 --layer 22

  # 2. HF reference outputs (one process per pattern; HF re-loads each time but
  #    the steering wrapper is cheap)
  source .venv/bin/activate
  python -m tests.integration.sglang_hooks_smoke --phase hf --model-path google/gemma-2-27b-it --pattern unsteered
  python -m tests.integration.sglang_hooks_smoke --phase hf --model-path google/gemma-2-27b-it --pattern capping
  ... etc.

  # 3. SGLang outputs — relaunch server per pattern, then:
  source .venv-sglang/bin/activate
  python -m tests.integration.sglang_hooks_smoke --phase sglang --pattern capping --port 30000

  # 4. Compare
  python -m tests.integration.sglang_hooks_smoke --phase compare
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file as _safe_save


_REPO = Path(__file__).resolve().parents[2]
_RESULTS = _REPO / "results" / "sglang_spike"
_VECTORS = _REPO / "data" / "cache" / "sglang_spike_vectors"


PROMPTS = [
    "Explain in two sentences why the sky is blue.",
    "List three differences between a violin and a cello.",
    "Write a haiku about the ocean.",
    "What is 17 multiplied by 23? Show your work.",
    "Give me a one-sentence definition of recursion.",
]

PATTERNS = ["unsteered", "addition_pos", "addition_neg", "capping", "cap_and_steer", "multi_axis_cap"]
MAX_NEW_TOKENS = 64


def _ensure_dirs() -> None:
    _RESULTS.mkdir(parents=True, exist_ok=True)
    _VECTORS.mkdir(parents=True, exist_ok=True)


# ── Phase 1: setup ─────────────────────────────────────────────────────────


def phase_setup(hidden_size: int, layer: int, n_axes: int = 3, seed: int = 42, layer_glob_prefix: str = "model.layers") -> None:
    """Generate fixed orthonormal vectors + write hooks JSON specs to disk.

    `layer_glob_prefix` controls the SGLang fnmatch glob used in target_modules.
    Default "model.layers" works for Gemma 2 / Qwen 3. Multimodal subjects like
    Gemma 4 31B (Gemma4ForConditionalGeneration) need "model.language_model.layers".
    """
    _ensure_dirs()
    g = torch.Generator().manual_seed(seed)
    M = torch.randn(hidden_size, n_axes, generator=g, dtype=torch.float32)
    Q, _ = torch.linalg.qr(M)
    vectors = [Q[:, i].clone().contiguous() for i in range(n_axes)]

    # Save each axis as its own safetensors file for easy referencing in --forward-hooks JSON.
    for i, v in enumerate(vectors):
        path = _VECTORS / f"axis_{i}.safetensors"
        _safe_save({"vector": v}, str(path))

    # Save prompts + meta.
    meta = {
        "prompts": PROMPTS,
        "patterns": PATTERNS,
        "hidden_size": hidden_size,
        "layer": layer,
        "layer_glob_prefix": layer_glob_prefix,
        "n_axes": n_axes,
        "seed": seed,
        "max_new_tokens": MAX_NEW_TOKENS,
        "vector_paths": [str(_VECTORS / f"axis_{i}.safetensors") for i in range(n_axes)],
        "tau_default": 0.0,
        "coefficient_default": 1.0,
    }
    (_RESULTS / "prompts.json").write_text(json.dumps(meta, indent=2))
    print(f"[setup] wrote {len(vectors)} axis vectors @ d={hidden_size}, layer={layer}")
    print(f"[setup] vectors → {_VECTORS}")
    print(f"[setup] meta → {_RESULTS / 'prompts.json'}")

    # Also emit example --forward-hooks JSON for each pattern (informational).
    examples = _build_hooks_specs(meta)
    (_RESULTS / "forward_hooks_specs.json").write_text(json.dumps(examples, indent=2))
    print(f"[setup] wrote example --forward-hooks JSON → {_RESULTS / 'forward_hooks_specs.json'}")


def _build_hooks_specs(meta: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """For each pattern, the corresponding --forward-hooks JSON list."""
    layer = meta["layer"]
    prefix = meta.get("layer_glob_prefix", "model.layers")
    target_module = f"{prefix}.{layer}"
    vps = meta["vector_paths"]
    return {
        "unsteered": [],
        "addition_pos": [
            {
                "name": "addition_pos",
                "target_modules": [target_module],
                "hook_factory": "src.steering.sglang_hook_factories:addition_factory",
                "config": {"vector_path": vps[0], "coefficient": 1.0, "positions": "all"},
            }
        ],
        "addition_neg": [
            {
                "name": "addition_neg",
                "target_modules": [target_module],
                "hook_factory": "src.steering.sglang_hook_factories:addition_factory",
                "config": {"vector_path": vps[0], "coefficient": -1.0, "positions": "all"},
            }
        ],
        "capping": [
            {
                "name": "capping",
                "target_modules": [target_module],
                "hook_factory": "src.steering.sglang_hook_factories:capping_factory",
                "config": {"vector_path": vps[0], "tau": 0.0, "positions": "all", "negate_vector": True},
            }
        ],
        "cap_and_steer": [
            {
                "name": "cap",
                "target_modules": [target_module],
                "hook_factory": "src.steering.sglang_hook_factories:capping_factory",
                "config": {"vector_path": vps[0], "tau": 0.0, "positions": "all", "negate_vector": True},
            },
            {
                "name": "steer",
                "target_modules": [target_module],
                "hook_factory": "src.steering.sglang_hook_factories:addition_factory",
                "config": {"vector_path": vps[1], "coefficient": 0.5, "positions": "all"},
            },
        ],
        "multi_axis_cap": [
            {
                "name": "cap_axis_0",
                "target_modules": [target_module],
                "hook_factory": "src.steering.sglang_hook_factories:capping_factory",
                "config": {"vector_path": vps[0], "tau": 0.0, "positions": "all", "negate_vector": True},
            },
            {
                "name": "cap_axis_1",
                "target_modules": [target_module],
                "hook_factory": "src.steering.sglang_hook_factories:capping_factory",
                "config": {"vector_path": vps[1], "tau": 0.0, "positions": "all", "negate_vector": True},
            },
            {
                "name": "cap_axis_2",
                "target_modules": [target_module],
                "hook_factory": "src.steering.sglang_hook_factories:capping_factory",
                "config": {"vector_path": vps[2], "tau": 0.0, "positions": "all", "negate_vector": True},
            },
        ],
    }


# ── Phase 2: HF reference ─────────────────────────────────────────────────


def phase_hf(model_path: str, pattern: str | list[str], output_path: Path | None = None) -> None:
    """Generate token IDs from HF + ActivationSteering wrapper.

    `pattern` may be a single string or a list of pattern names; in the latter
    case, the model is loaded once and each pattern is generated sequentially.
    """
    _ensure_dirs()
    meta = json.loads((_RESULTS / "prompts.json").read_text())
    layer = meta["layer"]
    vector_paths = meta["vector_paths"]

    patterns = [pattern] if isinstance(pattern, str) else list(pattern)

    # Load vectors.
    from safetensors.torch import load_file
    vectors = [load_file(p)["vector"] for p in vector_paths]

    # Load HF.
    sys.path.insert(0, str(_REPO))
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[hf] loading {model_path} bf16 (will run patterns: {patterns})…")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
    )
    model.eval()
    print(f"[hf] loaded in {time.time() - t0:.1f}s")

    for pat in patterns:
        _hf_run_one_pattern(model, tokenizer, model_path, pat, layer, vectors, output_path if len(patterns) == 1 else None)

    # Free memory at the end.
    del model
    torch.cuda.empty_cache()


def _hf_run_one_pattern(model, tokenizer, model_path: str, pattern: str, layer: int, vectors, output_path: Path | None) -> None:
    """Single-pattern HF generation, called by phase_hf for each pattern."""
    from src.steering import SteeringConfig, cap_and_steer, from_config, multi_axis_cap

    steer_ctx = None
    if pattern == "unsteered":
        steer_ctx = None
    elif pattern == "addition_pos":
        cfg = SteeringConfig(
            vectors=[vectors[0]],
            intervention_type="addition",
            layer_indices=[layer],
            coefficients=[1.0],
        )
        steer_ctx = from_config(model, cfg)
    elif pattern == "addition_neg":
        cfg = SteeringConfig(
            vectors=[vectors[0]],
            intervention_type="addition",
            layer_indices=[layer],
            coefficients=[-1.0],
        )
        steer_ctx = from_config(model, cfg)
    elif pattern == "capping":
        cfg = SteeringConfig(
            vectors=[vectors[0]],
            intervention_type="capping",
            layer_indices=[(layer, layer)],
            cap_thresholds=[0.0],
        )
        steer_ctx = from_config(model, cfg)
    elif pattern == "cap_and_steer":
        cap_cfg = SteeringConfig(
            vectors=[vectors[0]],
            intervention_type="capping",
            layer_indices=[(layer, layer)],
            cap_thresholds=[0.0],
        )
        steer_cfg = SteeringConfig(
            vectors=[vectors[1]],
            intervention_type="addition",
            layer_indices=[layer],
            coefficients=[0.5],
        )
        steer_ctx = cap_and_steer(model, cap_cfg, steer_cfg)
    elif pattern == "multi_axis_cap":
        steer_ctx = multi_axis_cap(
            model,
            [(vectors[0], (layer, layer), 0.0),
             (vectors[1], (layer, layer), 0.0),
             (vectors[2], (layer, layer), 0.0)],
        )
    else:
        raise ValueError(f"unknown pattern: {pattern}")

    # Generate.
    rows: list[dict[str, Any]] = []
    if steer_ctx is None:
        # Plain generation path.
        for i, prompt in enumerate(PROMPTS):
            tokens = _hf_generate_one(model, tokenizer, prompt, MAX_NEW_TOKENS)
            rows.append({"prompt_idx": i, "prompt": prompt, "generated_token_ids": tokens})
            print(f"[hf:{pattern}] prompt {i}: {len(tokens)} tokens")
    else:
        with steer_ctx:
            for i, prompt in enumerate(PROMPTS):
                tokens = _hf_generate_one(model, tokenizer, prompt, MAX_NEW_TOKENS)
                rows.append({"prompt_idx": i, "prompt": prompt, "generated_token_ids": tokens})
                print(f"[hf:{pattern}] prompt {i}: {len(tokens)} tokens")

    out_path = output_path or (_RESULTS / f"hf_{pattern}.json")
    out_path.write_text(json.dumps({"pattern": pattern, "model_path": model_path, "rows": rows}, indent=2))
    print(f"[hf:{pattern}] wrote {out_path}")


def _hf_generate_one(model, tokenizer, prompt: str, max_new_tokens: int) -> list[int]:
    msgs = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inp = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(model.device)
    with torch.inference_mode():
        out = model.generate(
            **inp,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,  # ignored when do_sample=False
            top_p=1.0,
            top_k=0,
        )
    new_ids = out[0, inp["input_ids"].shape[1]:].tolist()
    return new_ids


# ── Phase 3: SGLang test ──────────────────────────────────────────────────


def phase_sglang(
    pattern: str,
    port: int = 30000,
    model_path: str = "google/gemma-2-27b-it",
    output_path: Path | None = None,
) -> None:
    """Query running SGLang server, generate 64 tokens per prompt at T=0.

    Uses the /generate endpoint with pre-tokenized input_ids (same chat
    template as HF) and reads `output_ids` directly from the response, so
    no re-tokenization is needed for token-id comparison.
    """
    import requests
    from transformers import AutoTokenizer

    _ensure_dirs()
    base = f"http://127.0.0.1:{port}"
    try:
        r = requests.get(f"{base}/health_generate", timeout=10)
        r.raise_for_status()
    except Exception as e:
        print(f"[sglang:{pattern}] server not reachable at {base}: {e}", file=sys.stderr)
        sys.exit(2)

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    rows: list[dict[str, Any]] = []
    for i, prompt in enumerate(PROMPTS):
        msgs = [{"role": "user", "content": prompt}]
        rendered = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        input_ids = tokenizer(rendered, add_special_tokens=False)["input_ids"]
        payload = {
            "input_ids": input_ids,
            "sampling_params": {
                "temperature": 0.0,
                "top_p": 1.0,
                "max_new_tokens": MAX_NEW_TOKENS,
            },
            "return_logprob": False,
        }
        r = requests.post(f"{base}/generate", json=payload, timeout=180)
        r.raise_for_status()
        data = r.json()
        # SGLang /generate returns output_ids in meta_info or top-level depending on version.
        token_ids = (
            data.get("output_ids")
            or data.get("meta_info", {}).get("output_token_ids")
            or data.get("meta_info", {}).get("output_ids")
            or []
        )
        if not token_ids:
            # Last fallback: re-tokenize the text (less reliable, but handles minor schema drift).
            text = data.get("text") or data.get("output", "")
            token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        text_out = data.get("text", tokenizer.decode(token_ids, skip_special_tokens=True))
        rows.append({
            "prompt_idx": i,
            "prompt": prompt,
            "generated_token_ids": token_ids,
            "text": text_out,
            "input_token_count": len(input_ids),
        })
        print(f"[sglang:{pattern}] prompt {i}: {len(token_ids)} tokens")

    out_path = output_path or (_RESULTS / f"sglang_{pattern}.json")
    out_path.write_text(json.dumps({"pattern": pattern, "port": port, "model_path": model_path, "rows": rows}, indent=2))
    print(f"[sglang:{pattern}] wrote {out_path}")


# ── Phase 4: compare ──────────────────────────────────────────────────────


def _match_len(a: list[int], b: list[int]) -> int:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


def phase_compare(results_dir: Path | None = None) -> None:
    """Read every hf_<p>.json + sglang_<p>.json pair and emit equivalence.json.

    Reports three metrics per (pattern, prompt):
      - cross_backend_match_len: HF[pattern] vs SGLang[pattern] (kernel + hook drift)
      - hf_steering_signature:   HF[pattern] vs HF[unsteered]   (pure HF hook effect)
      - sg_steering_signature:   SG[pattern] vs SG[unsteered]   (pure SGLang hook effect)

    The hook is "correct" if HF and SGLang signatures agree qualitatively (hook
    fires at similar token positions with similar magnitude), even if cross-
    backend match is short due to bf16 kernel drift.
    """
    _ensure_dirs()
    rdir = results_dir or _RESULTS
    rdir.mkdir(parents=True, exist_ok=True)
    # Load unsteered baselines.
    hf_un_path = rdir / "hf_unsteered.json"
    sg_un_path = rdir / "sglang_unsteered.json"
    hf_un = json.loads(hf_un_path.read_text()) if hf_un_path.exists() else None
    sg_un = json.loads(sg_un_path.read_text()) if sg_un_path.exists() else None

    out: dict[str, Any] = {"per_pattern": {}}
    for pat in PATTERNS:
        hf_path = rdir / f"hf_{pat}.json"
        sg_path = rdir / f"sglang_{pat}.json"
        hf = json.loads(hf_path.read_text()) if hf_path.exists() else None
        sg = json.loads(sg_path.read_text()) if sg_path.exists() else None

        if hf is None and sg is None:
            print(f"[compare] no data for {pat} — skipping")
            continue

        cells = []
        n_rows = max(len(hf["rows"]) if hf else 0, len(sg["rows"]) if sg else 0)
        for i in range(n_rows):
            hf_ids = (hf["rows"][i]["generated_token_ids"] if hf else None) or []
            sg_ids = (sg["rows"][i]["generated_token_ids"] if sg else None) or []
            hf_un_ids = (hf_un["rows"][i]["generated_token_ids"] if hf_un else None) or []
            sg_un_ids = (sg_un["rows"][i]["generated_token_ids"] if sg_un else None) or []
            cells.append({
                "prompt_idx": i,
                "cross_backend_match_len": _match_len(hf_ids, sg_ids) if hf_ids and sg_ids else None,
                "hf_steer_signature_len": _match_len(hf_ids, hf_un_ids) if hf_ids and hf_un_ids else None,
                "sg_steer_signature_len": _match_len(sg_ids, sg_un_ids) if sg_ids and sg_un_ids else None,
                "hf_len": len(hf_ids),
                "sglang_len": len(sg_ids),
            })

        def _med(field: str) -> float | None:
            vals = [c[field] for c in cells if c[field] is not None]
            if not vals:
                return None
            vals.sort()
            return vals[len(vals) // 2]

        out["per_pattern"][pat] = {
            "cells": cells,
            "median_cross_backend_match": _med("cross_backend_match_len"),
            "median_hf_steer_signature": _med("hf_steer_signature_len"),
            "median_sg_steer_signature": _med("sg_steer_signature_len"),
            "n_total": len(cells),
        }
    (rdir / "equivalence.json").write_text(json.dumps(out, indent=2))
    print(f"[compare] wrote {rdir / 'equivalence.json'}")
    print(f"{'pattern':<22} {'cross':<8} {'HF-sig':<8} {'SG-sig':<8}  (token count where divergence starts; lower = stronger steering effect)")
    for pat, s in out["per_pattern"].items():
        print(
            f"  {pat:<20} {str(s['median_cross_backend_match']):<8} "
            f"{str(s['median_hf_steer_signature']):<8} "
            f"{str(s['median_sg_steer_signature']):<8}"
        )


# ── CLI ────────────────────────────────────────────────────────────────────


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--phase", required=True, choices=["setup", "hf", "sglang", "compare"])
    p.add_argument("--pattern", help="pattern name or comma-separated list (required for hf/sglang). Valid names: " + ",".join(PATTERNS))
    p.add_argument("--model-path", default="google/gemma-2-27b-it")
    p.add_argument("--hidden-size", type=int, default=4608, help="hidden dim (default Gemma 2 27B = 4608)")
    p.add_argument("--layer", type=int, default=22, help="single-layer steering target")
    p.add_argument("--layer-glob-prefix", default="model.layers", help="SGLang target_modules prefix; use 'model.language_model.layers' for Gemma 4")
    p.add_argument("--port", type=int, default=30000, help="sglang server port")
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--results-dir", type=Path, default=None, help="for --phase compare: directory to read hf_/sglang_ JSONs from")
    args = p.parse_args()

    if args.phase == "setup":
        phase_setup(hidden_size=args.hidden_size, layer=args.layer, layer_glob_prefix=args.layer_glob_prefix)
    elif args.phase == "hf":
        if not args.pattern:
            p.error("--pattern required for --phase hf")
        pats = [p.strip() for p in args.pattern.split(",")]
        for pat in pats:
            if pat not in PATTERNS:
                raise SystemExit(f"unknown pattern: {pat}; valid: {PATTERNS}")
        phase_hf(args.model_path, pats if len(pats) > 1 else pats[0], args.output)
    elif args.phase == "sglang":
        if not args.pattern:
            p.error("--pattern required for --phase sglang")
        phase_sglang(args.pattern, port=args.port, output_path=args.output)
    elif args.phase == "compare":
        phase_compare(results_dir=args.results_dir)


if __name__ == "__main__":
    main()
