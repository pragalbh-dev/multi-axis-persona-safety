"""HuggingFace forward-hook extraction backend.

Wraps `external/assistant-axis::ActivationExtractor` so we get the paper's exact
hook semantics. Loads bf16 weights via plain HF transformers (`device_map="auto"`,
`attn_implementation="sdpa"`). Used by Plan B for:

  - Step 1b: per-rollout activations at every layer (PCA + AA fitting input).
  - Step 1c: lmsys-chat-1m residual-norm cache (single-layer hook at L*).
  - Step 7b: per-prompt activation extraction over (prompt, response) pairs
             to fill `aa_projection` + `pc_projections` in details.parquet.

Mean-response-token aggregation is applied here, NOT at load time, so caches
stay `(n_prompts, d_model)` instead of `(n_prompts, seq_len, d_model)`.

Row schema (flexible):
  - dict with `system` + `question` + `response` keys → canonical role-rollout shape.
    Response-token span is identified by tokenizing the (system + question) prefix
    and (system + question + response) full sequence; response tokens = [prefix_len:].
  - dict with `conversation: [{role, content}, ...]` → arbitrary chat; the LAST
    assistant turn is the response.
  - dict with `input_text` + `response_text` → standard PER_PROMPT_COLUMNS shape.
  - str → user-only (no response, mean-pools over the entire user prompt; only
    sensible for the lmsys-chat-1m norm caching path).
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
import yaml
from safetensors.torch import save_file as _safe_save

from src.extraction.types import ActivationCache, ExtractionConfig

# Make external/assistant-axis importable.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_AA_PATH = _PROJECT_ROOT / "external" / "assistant-axis"
if str(_AA_PATH) not in sys.path:
    sys.path.insert(0, str(_AA_PATH))


def _git_sha() -> str:
    try:
        import subprocess

        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except Exception:
        return "unknown"


def _load_subjects() -> dict:
    return yaml.safe_load((Path("configs/subjects.yaml")).read_text())


def _normalize_row(row: Any, tokenizer: Any, chat_template_kwargs: dict) -> tuple[str, int, int]:
    """Return (full_chat_text, prefix_token_count, full_token_count) for a row.

    `prefix_token_count` is the number of tokens BEFORE the assistant response
    (so response tokens = [prefix_token_count : full_token_count]). For user-only
    rows (no response), prefix=0 and we mean-pool over the entire sequence.
    """
    if isinstance(row, str):
        msgs = [{"role": "user", "content": row}]
        full = tokenizer.apply_chat_template(
            msgs,
            tokenize=False,
            add_generation_prompt=False,
            **chat_template_kwargs,
        )
        n = len(tokenizer(full, return_tensors=None, add_special_tokens=False)["input_ids"])
        return full, 0, n  # mean over entire sequence

    if not isinstance(row, dict):
        raise TypeError(f"Unsupported row type: {type(row)}")

    if "conversation" in row:
        conv = row["conversation"]
        # Strip the final assistant turn to get the prefix
        if conv and conv[-1].get("role") in ("assistant", "model"):
            prefix_msgs = conv[:-1]
            response_text = conv[-1]["content"]
        else:
            prefix_msgs = conv
            response_text = ""
    elif "input_text" in row and "response_text" in row:
        prefix_msgs = [{"role": "user", "content": row["input_text"]}]
        response_text = row["response_text"]
    elif "question" in row:
        prefix_msgs = []
        if row.get("system"):
            prefix_msgs.append({"role": "system", "content": row["system"]})
        prefix_msgs.append({"role": "user", "content": row["question"]})
        response_text = row.get("response", "")
    else:
        raise ValueError(f"Cannot extract response span from row keys: {list(row.keys())}")

    # Build prefix string with assistant generation prompt appended (so the prefix
    # ends right where the assistant turn would begin).
    prefix_text = tokenizer.apply_chat_template(
        prefix_msgs,
        tokenize=False,
        add_generation_prompt=True,
        **chat_template_kwargs,
    )
    if response_text:
        full_text = prefix_text + response_text
    else:
        full_text = prefix_text  # no response; mean-pool over entire sequence

    prefix_n = len(tokenizer(prefix_text, return_tensors=None, add_special_tokens=False)["input_ids"])
    full_n = len(tokenizer(full_text, return_tensors=None, add_special_tokens=False)["input_ids"])
    if not response_text:
        prefix_n = 0  # mean over whole sequence

    return full_text, prefix_n, full_n


def _aggregate(
    per_token_acts: torch.Tensor,  # (seq_len, d_model)
    response_start: int,
    response_end: int,
    aggregation: str,
) -> torch.Tensor:
    if aggregation == "all":
        return per_token_acts
    if aggregation == "last":
        return per_token_acts[-1]

    # mean_response (default)
    if response_end <= response_start:
        # Degenerate; mean over the whole sequence to avoid empty-pool error.
        slice_ = per_token_acts
    else:
        slice_ = per_token_acts[response_start:response_end]
    return slice_.mean(dim=0)


def slice_response_tokens_left_padded(
    activations: torch.Tensor,
    attention_mask: torch.Tensor,
    full_token_lens: list[int],
    prefix_token_lens: list[int],
) -> list[torch.Tensor]:
    """Mean-pool response tokens from a left-padded batch forward output.

    Args:
        activations: (B, S, D) — per-token activations from one layer hook.
        attention_mask: (B, S) — 1 where token is real, 0 where padding.
        full_token_lens: list[B] — total real-token count per row (=
            number of 1s in attention_mask[i]).
        prefix_token_lens: list[B] — token count of the prefix (everything
            up to but not including the response). 0 means "no response;
            mean over all real tokens".

    Returns: list[B] of (D,) tensors (mean-pooled response-token activations).

    With LEFT padding: the last `full_token_lens[i]` positions are real;
    the last `full_token_lens[i] - prefix_token_lens[i]` positions are the
    response. Equivalently, response tokens occupy `[S - response_n : S]`.
    """
    B, S, _ = activations.shape
    if attention_mask.shape != (B, S):
        raise ValueError(
            f"attention_mask shape {attention_mask.shape} != activations shape ({B},{S})"
        )
    out = []
    for bi in range(B):
        full_n = int(full_token_lens[bi])
        prefix_n = int(prefix_token_lens[bi])
        actual = int(attention_mask[bi].sum().item())
        # Truncation: tokenizer may have capped at max_length; the actual real tokens
        # is min(full_n, actual). Response tokens are the LAST (actual - prefix_n)
        # positions, but only if prefix_n was actually retained.
        if prefix_n == 0:
            # No response-vs-prefix split: pool over all real tokens.
            real_start = S - actual
            real_end = S
        else:
            response_n = max(actual - prefix_n, 0)
            if response_n == 0:
                # Truncation ate the response; fall back to pooling all real tokens
                # so we don't crash, but log this as a "degenerate row" upstream.
                real_start = S - actual
                real_end = S
            else:
                real_start = S - response_n
                real_end = S
        slice_acts = activations[bi, real_start:real_end, :]
        # fp32 reduction for numerical stability
        out.append(slice_acts.float().mean(dim=0))
    return out


def extract_via_hf(
    model_or_id: Any,
    prompts: Sequence[dict[str, str]] | Sequence[str],
    cfg: ExtractionConfig,
    *,
    dataset: str,
    cache_root: str | Path = "data/cache",
    use_cache: bool = True,
    batch_size: int = 8,
    max_seq_len: int = 2048,
) -> dict[int, ActivationCache]:
    """Batched HF forward-hook extraction over `prompts`.

    Returns {layer: ActivationCache}. Rows can be str / dict; see module
    docstring for supported shapes.

    Implementation:
      1. Per-row CPU work: build full chat-templated text + count prefix /
         full token lengths.
      2. Sort by full_token_len descending (tighter padding within batches).
      3. For each batch: tokenize with left-padding, forward pass with hooks
         on every requested layer, slice each row's response tokens, mean-pool.
      4. Re-sort to original row order before saving.

    Sequential `extractor.full_conversation` was the Stage 1 stub; this
    batched version is ~3-5× faster on bf16 27B + 4-GPU device_map="auto".
    """
    cache_root = Path(cache_root)
    layers = list(cfg.layers)

    # ── Cache short-circuit ────────────────────────────────────────────────
    if use_cache:
        all_cached = True
        cached: dict[int, ActivationCache] = {}
        for layer in layers:
            stem = ActivationCache.cache_path(cfg.model_id, dataset, layer, cache_root)
            if stem.with_suffix(".safetensors").exists() and stem.with_suffix(".meta.json").exists():
                cached[layer] = ActivationCache.load(stem)
            else:
                all_cached = False
                break
        if all_cached:
            return cached

    # ── Resolve subject + load model ───────────────────────────────────────
    from assistant_axis.internals.model import ProbingModel

    if isinstance(model_or_id, str):
        subjects = _load_subjects()
        if model_or_id in subjects:
            hf_id = subjects[model_or_id]["hf_id"]
            chat_template_kwargs = subjects[model_or_id].get("chat_template_kwargs") or {}
            trust_remote_code = subjects[model_or_id].get("trust_remote_code", False)
        else:
            hf_id = model_or_id
            chat_template_kwargs = {}
            trust_remote_code = False

        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=trust_remote_code)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        # Truncation_side="left" preserves the response (which lives at the END of the
        # full chat-templated text) when seq exceeds max_seq_len. The prefix
        # (persona preamble + question) gets cut from the start.
        tokenizer.truncation_side = "left"
        model = AutoModelForCausalLM.from_pretrained(
            hf_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="sdpa",
            trust_remote_code=trust_remote_code,
        )
        # Gemma 2's checkpoint config silently falls back to eager despite the
        # attn_implementation="sdpa" request (because attn_logit_softcapping=50.0).
        # Force-flip post-load — empirically verified to make 0 eager calls in a
        # forward (2026-04-25). PyTorch 2.10's sdpa handles softcap correctly.
        model.config._attn_implementation = "sdpa"
        for sub in model.modules():
            if hasattr(sub, "config") and hasattr(sub.config, "_attn_implementation"):
                sub.config._attn_implementation = "sdpa"
        model.eval()
        probing_model = ProbingModel.from_existing(model, tokenizer, model_name=hf_id)
    else:
        probing_model = ProbingModel.from_existing(model_or_id, model_or_id.tokenizer)
        chat_template_kwargs = {}
        tokenizer = probing_model.tokenizer
        tokenizer.padding_side = "left"
        model = probing_model.model

    layer_modules = probing_model.get_layers()
    # IMPORTANT: bypass the LM head. For Gemma 2 (and any model with
    # final_logit_softcapping), calling model(input_ids=...) materializes a full
    # (B, S, vocab=256000) fp32 logits tensor for the softcap division, which
    # OOMs at long contexts (e.g. 12 GB transient at B=8, S=5120). Hooks fire
    # on layer modules during model.model.forward — we only need the residual
    # stream, not the logits, so call the base model directly.
    if hasattr(model, "model") and hasattr(model.model, "forward"):
        forward_target = model.model
    else:
        forward_target = model

    # ── Per-row CPU prep ───────────────────────────────────────────────────
    prep: list[tuple[int, str, int, int, str]] = []  # (orig_idx, full_text, prefix_n, full_n, prompt_id)
    for idx, row in enumerate(prompts):
        if isinstance(row, dict) and "prompt_id" in row:
            pid = str(row["prompt_id"])
        else:
            pid = f"row_{idx}"
        full_text, prefix_n, full_n = _normalize_row(row, tokenizer, chat_template_kwargs)
        prep.append((idx, full_text, prefix_n, full_n, pid))

    # Sort by full_n descending so each batch has near-uniform padding.
    sort_order = sorted(range(len(prep)), key=lambda i: -prep[i][3])

    # ── Batched forward ────────────────────────────────────────────────────
    pooled_per_layer_per_row: dict[int, list[torch.Tensor | None]] = {
        layer: [None] * len(prep) for layer in layers
    }

    # Capture buffers (overwritten each batch)
    captured: dict[int, torch.Tensor | None] = {layer: None for layer in layers}

    def _make_hook(layer_idx: int):
        def hook(module, inp, out):  # noqa: ANN001
            t = out[0] if isinstance(out, tuple) else out
            captured[layer_idx] = t

        return hook

    handles = []
    for layer in layers:
        handles.append(layer_modules[layer].register_forward_hook(_make_hook(layer)))

    try:
        for chunk_start in range(0, len(sort_order), batch_size):
            chunk_orig = sort_order[chunk_start : chunk_start + batch_size]
            texts = [prep[i][1] for i in chunk_orig]
            prefix_ns = [prep[i][2] for i in chunk_orig]
            full_ns = [prep[i][3] for i in chunk_orig]

            tok_out = tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_seq_len,
                add_special_tokens=False,
            )
            input_ids = tok_out["input_ids"].to(next(model.parameters()).device)
            attention_mask = tok_out["attention_mask"].to(input_ids.device)

            for layer in layers:
                captured[layer] = None

            with torch.inference_mode():
                _ = forward_target(input_ids=input_ids, attention_mask=attention_mask)

            # Slice + mean-pool
            for layer in layers:
                acts = captured[layer]
                if acts is None:
                    raise RuntimeError(f"Hook on layer {layer} did not fire")
                pooled = slice_response_tokens_left_padded(
                    acts.cpu(),  # move once for slicing + reduction
                    attention_mask.cpu(),
                    full_ns,
                    prefix_ns,
                )
                for bi, orig in enumerate(chunk_orig):
                    pooled_per_layer_per_row[layer][orig] = pooled[bi].to(torch.bfloat16)
    finally:
        for h in handles:
            h.remove()

    # ── Stack in original order + save ─────────────────────────────────────
    sha = _git_sha()
    prompt_ids = [p[4] for p in prep]  # already in original order
    out_caches: dict[int, ActivationCache] = {}
    for layer in layers:
        rows = pooled_per_layer_per_row[layer]
        if any(r is None for r in rows):
            raise RuntimeError(f"Layer {layer} has missing rows after batched extraction")
        tensor = torch.stack(rows, dim=0)  # (n, d_model)  # type: ignore[arg-type]
        cache = ActivationCache(
            model_id=cfg.model_id,
            dataset=dataset,
            layer=layer,
            tensor=tensor.contiguous(),
            prompt_ids=prompt_ids,
            token_aggregation=cfg.token_aggregation,
            dtype=cfg.dtype,
            seed=cfg.seed,
            git_sha=sha,
        )
        stem = ActivationCache.cache_path(cfg.model_id, dataset, layer, cache_root)
        cache.save(stem)
        out_caches[layer] = cache

    if isinstance(model_or_id, str):
        del model
        torch.cuda.empty_cache()

    return out_caches


def cache_path_for(cfg: ExtractionConfig, dataset: str, layer: int, cache_root: str | Path) -> Path:
    """Canonical cache path stem (no extension)."""
    return ActivationCache.cache_path(cfg.model_id, dataset, layer, cache_root)


def aggregate_response_tokens(
    activations: torch.Tensor,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    """Mean-pool `activations[batch, seq, d]` over `response_mask[batch, seq]==1`.

    Kept for backwards-compat with Stage 1 import sites; the main extraction
    path uses `_aggregate` above.
    """
    if activations.ndim != 3 or response_mask.ndim != 2:
        raise ValueError(
            f"Expected activations (B, S, D) and mask (B, S), got "
            f"{tuple(activations.shape)} and {tuple(response_mask.shape)}"
        )
    mask = response_mask.to(dtype=activations.dtype).unsqueeze(-1)
    counts = mask.sum(dim=1).clamp(min=1)
    if (response_mask.sum(dim=1) == 0).any():
        raise ValueError("aggregate_response_tokens received an all-zero mask row")
    return (activations * mask).sum(dim=1) / counts
