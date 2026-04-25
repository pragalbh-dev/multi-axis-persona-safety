"""Top-level activation-extraction dispatcher.

Two backends:
- HF / TransformerLens — clean forward hooks on bf16 weights, used by the PCA
  fitting pass in Stage 3 T3.1. See `src.extraction.backend_hf`.
- vLLM — uses vLLM's hidden-states return path (or nnsight as fallback) so
  high-throughput rollouts (Stage 3 T3.1 default Assistant rollouts, Stage 4
  attack rollouts) can extract activations during generation. See
  `src.extraction.backend_vllm`. **Stage 1 stub only**; Stage 3 T3.1 fills
  empirical hook paths.

Storage layout: `data/cache/activations/{model_id}/{dataset}/L{layer}.{safetensors,meta.json}`.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

from src.extraction.types import ActivationCache, ExtractionConfig

Backend = Literal["hf", "vllm"]


def extract_activations(
    model_or_id: Any,
    prompts: Sequence[dict[str, str]] | Sequence[str],
    cfg: ExtractionConfig,
    *,
    dataset: str,
    backend: Backend = "hf",
    cache_root: str | Path = "data/cache",
    use_cache: bool = True,
) -> dict[int, ActivationCache]:
    """Extract activations for `prompts` at every layer in `cfg.layers`.

    Returns a dict mapping `layer_idx -> ActivationCache`. When `use_cache` is
    True (default), already-cached layers are loaded from disk and skipped.

    Args:
        model_or_id: An already-loaded HF model OR a HF model id string (which
            backend_hf will load itself).
        prompts: Either a list of strings (treated as user-only) or a list of
            chat-message dicts (`{"role": "user", "content": "..."}`).
        cfg: Extraction config (layers, hook_point template, aggregation, dtype).
        dataset: Stable string used in the cache path.
        backend: "hf" for forward-hook bf16 extraction; "vllm" for rollout-path
            extraction. Stage 1 only ships "hf".
        cache_root: Root directory for cache files.
        use_cache: When True, load cached layers from disk and only extract
            missing ones. When False, force re-extraction.

    Returns:
        dict[layer_idx, ActivationCache]. Each cache has tensor shape
        `(len(prompts), d_model)` after token aggregation.
    """
    if backend == "hf":
        from src.extraction.backend_hf import extract_via_hf

        return extract_via_hf(
            model_or_id,
            prompts,
            cfg,
            dataset=dataset,
            cache_root=cache_root,
            use_cache=use_cache,
        )
    if backend == "vllm":
        raise NotImplementedError(
            "vLLM extraction backend is a Stage 3 T3.1 deliverable; the contract "
            "lives in src/extraction/backend_vllm.py"
        )
    raise ValueError(f"Unknown backend: {backend!r}")
