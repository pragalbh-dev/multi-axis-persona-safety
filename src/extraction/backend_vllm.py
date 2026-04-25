"""vLLM-side activation extraction (Stage 1 stub; Stage 3 T3.1 fills).

Two viable paths and the pick is empirical:

1. **vLLM hidden_states return path.** vLLM 0.19+ supports returning hidden
   states from a custom logits processor or via the `--hidden-states` server
   flag. We can request a list of layer indices and get back per-token
   hidden states alongside the generated tokens. Cleanest if it works on
   Blackwell + multimodal (Gemma 4, Qwen 3.6) at TP=4.

2. **nnsight on a separate bf16 process.** Run vLLM for high-throughput
   generation; replay generated sequences through nnsight on a parallel
   bf16 model to extract activations. Cleaner separation, but doubles
   memory + compute on the extraction sample.

Stage 3 T3.1 must:
- Pick path 1 vs path 2 based on a smoke test of both on Gemma 2 27B and
  Gemma 4 31B (the multimodal arch). Log the choice in `decisions.md`.
- Fill in the empirical hook paths for `model.layers[L]` vs
  `model.language_model.layers[L]` per family (the templates live in
  `configs/model_hooks.yaml` but need a forward-pass to confirm).
- Implement response-token-span identification (where does the assistant
  reply start in the generated token stream?) for each chat template.

Until then, this module exists only to advertise the contract. Calling
`extract_during_generate` raises NotImplementedError.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from src.extraction.types import ActivationCache, ExtractionConfig


def extract_during_generate(
    llm: Any,
    prompts: Sequence[dict[str, str]] | Sequence[str],
    cfg: ExtractionConfig,
    *,
    dataset: str,
) -> tuple[dict[int, ActivationCache], list[str]]:
    """Run rollouts AND capture activations in one pass.

    Returns `(caches, response_texts)`.

    Stage 3 T3.1 deliverable; Stage 1 ships only the contract.
    """
    raise NotImplementedError(
        "vLLM extraction backend is a Stage 3 T3.1 deliverable. "
        "See module docstring for the two candidate paths."
    )
