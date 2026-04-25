"""Activation-extraction shared types.

`ActivationCache` is the on-disk + in-memory contract for cached activations.
Storage layout (locked Stage 1 T1.2 + CONVENTIONS):

    data/cache/activations/{model_id}/{dataset}/L{layer}.safetensors
    data/cache/activations/{model_id}/{dataset}/L{layer}.meta.json

The safetensors holds aggregated activations as a single tensor of shape
`(n_prompts, d_model)` — aggregation (mean over response tokens by default)
runs at extract time so caches stay small. The .meta.json sidecar carries
shape, dtype, token_aggregation, prompt_id ordering, seed, and git SHA.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import torch
from safetensors.torch import load_file as _safe_load
from safetensors.torch import save_file as _safe_save

TokenAggregation = Literal["mean_response", "last", "all"]
DType = Literal["bf16", "fp16", "fp32"]


@dataclass
class ExtractionConfig:
    """How to extract — handed to a backend (HF/TL or vLLM)."""

    model_id: str
    layers: list[int]
    hook_point: str  # templated, e.g. "blocks.{L}.hook_resid_post"
    token_aggregation: TokenAggregation = "mean_response"
    dtype: DType = "bf16"
    batch_size: int = 8
    seed: int = 42


@dataclass
class ActivationCache:
    """In-memory representation of one (model, dataset, layer) tensor cache.

    `tensor` shape is `(n_prompts, d_model)` after aggregation. `prompt_ids`
    is a parallel list of stable string IDs.
    """

    model_id: str
    dataset: str
    layer: int
    tensor: torch.Tensor
    prompt_ids: list[str]
    token_aggregation: TokenAggregation = "mean_response"
    dtype: DType = "bf16"
    seed: int = 42
    git_sha: str = "unknown"
    extra_meta: dict[str, str] = field(default_factory=dict)

    # ── IO ────────────────────────────────────────────────────────────────
    def save(self, path: str | Path) -> None:
        """Write `<path>.safetensors` + `<path>.meta.json` (atomic-ish)."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        st_path = p.with_suffix(".safetensors")
        meta_path = p.with_suffix(".meta.json")
        _safe_save({"activations": self.tensor.contiguous()}, str(st_path))
        meta = {
            "model_id": self.model_id,
            "dataset": self.dataset,
            "layer": self.layer,
            "shape": list(self.tensor.shape),
            "dtype": self.dtype,
            "token_aggregation": self.token_aggregation,
            "prompt_ids": self.prompt_ids,
            "seed": self.seed,
            "git_sha": self.git_sha,
            **self.extra_meta,
        }
        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> ActivationCache:
        """Inverse of `save`."""
        p = Path(path)
        st_path = p.with_suffix(".safetensors")
        meta_path = p.with_suffix(".meta.json")
        tensors = _safe_load(str(st_path))
        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        return cls(
            model_id=meta["model_id"],
            dataset=meta["dataset"],
            layer=int(meta["layer"]),
            tensor=tensors["activations"],
            prompt_ids=list(meta["prompt_ids"]),
            token_aggregation=meta.get("token_aggregation", "mean_response"),
            dtype=meta.get("dtype", "bf16"),
            seed=int(meta.get("seed", 42)),
            git_sha=meta.get("git_sha", "unknown"),
            extra_meta={
                k: v
                for k, v in meta.items()
                if k
                not in {
                    "model_id",
                    "dataset",
                    "layer",
                    "shape",
                    "dtype",
                    "token_aggregation",
                    "prompt_ids",
                    "seed",
                    "git_sha",
                }
            },
        )

    @staticmethod
    def cache_path(model_id: str, dataset: str, layer: int, root: str | Path) -> Path:
        """Canonical cache path under data/cache/activations/."""
        return Path(root) / "activations" / model_id / dataset / f"L{layer}"
