"""HuggingFace forward-hook extraction backend.

Wraps `external/assistant-axis/assistant_axis/internals/activations.py
::ActivationExtractor` so we get the paper's exact hook semantics for free.
This backend loads bf16 weights via plain HF transformers (no vLLM); use it
for the PCA fitting pass in Stage 3 T3.1 (~hundreds of prompts), not for
high-throughput rollouts.

Mean-response-token aggregation is applied here, NOT at load time, so caches
stay `(n_prompts, d_model)` instead of `(n_prompts, seq_len, d_model)`.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from src.extraction.types import ActivationCache, ExtractionConfig


def extract_via_hf(
    model_or_id: Any,
    prompts: Sequence[dict[str, str]] | Sequence[str],
    cfg: ExtractionConfig,
    *,
    dataset: str,
    cache_root: str | Path = "data/cache",
    use_cache: bool = True,
) -> dict[int, ActivationCache]:
    """Stage 1 stub. Stage 2 T2.2 fills the actual implementation.

    Implementation contract (Stage 2):
    1. If `model_or_id` is a string, load via `transformers.AutoModelForCausalLM`
       at `cfg.dtype` (bf16 only for now). Otherwise use it directly.
    2. Build an `ActivationExtractor` from `external/assistant-axis`.
    3. For each prompt, identify the response-token span (where the assistant
       reply lives in the chat-templated sequence). Run a single forward pass
       with hooks on `cfg.layers`.
    4. Aggregate per `cfg.token_aggregation`:
       - "mean_response": mean over response tokens (paper line 96)
       - "last": final-token activation
       - "all": (n_prompts, seq_len, d_model) — only for ad-hoc debugging
    5. Return `{layer: ActivationCache}` and write each to
       `data/cache/activations/{model_id}/{dataset}/L{layer}` per the contract.
    6. When `use_cache=True`, skip layers that already exist on disk.
    """
    raise NotImplementedError(
        "Stage 1 ships interfaces only. Stage 2 T2.2 implements the HF backend "
        "by wrapping external/assistant-axis ActivationExtractor."
    )


def cache_path_for(cfg: ExtractionConfig, dataset: str, layer: int, cache_root: str | Path) -> Path:
    """Canonical cache path stem (no extension)."""
    return ActivationCache.cache_path(cfg.model_id, dataset, layer, cache_root)


def aggregate_response_tokens(
    activations: torch.Tensor,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    """Mean-pool `activations[batch, seq, d]` over `response_mask[batch, seq]==1`.

    Returns `(batch, d)`. Empty masks raise — paper's >=10-responses-per-category
    filter means we should never see one in production, but we surface it loudly.
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
