"""Subprocess work-module: HF activation extraction.

Loads a subject via HF transformers (`device_map="auto"`, bf16), runs forward
hooks on the requested layers, mean-pools over response tokens, writes
safetensors caches per locked schema, exits.

Invoked by `src.utils.model_runner.run_in_subprocess("src.extraction.run_extract", args, output_path)`.

Args JSON schema:
{
  "model_id":         str,        # key in configs/subjects.yaml
  "rows":             list[dict], # each row: {prompt_id, system, question, response} or
                                  #          {prompt_id, conversation: [{role, content}, ...]}
  "layers":           list[int],  # which layers to capture
  "dataset":          str,        # cache namespace (e.g. "plan_b_role_rollouts")
  "token_aggregation": str,       # "mean_response" | "last" | "all"
  "cache_root":       str,        # default "data/cache"
  "seed":             int
}

Output JSON:
{
  "status": "ok",
  "elapsed_seconds": float,
  "peak_vram_per_gpu": [4 floats, GiB],
  "artifacts": [list of safetensors file paths produced],
  "n_rows": int,
  "n_layers": int,
  "d_model": int
}
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

# Pin GPUs before torch import.
from src.utils import env as _env  # noqa: F401  side-effect import


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--args-json", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    cli = parser.parse_args()

    args = json.loads(cli.args_json.read_text())
    t0 = time.time()

    # Lazy imports — keep the parent process clean of torch.
    from src.extraction.backend_hf import extract_via_hf
    from src.extraction.types import ExtractionConfig

    cfg = ExtractionConfig(
        model_id=args["model_id"],
        layers=args["layers"],
        token_aggregation=args.get("token_aggregation", "mean_response"),
        seed=args.get("seed", 42),
    )

    rows = args["rows"]
    caches = extract_via_hf(
        model_or_id=args["model_id"],
        prompts=rows,
        cfg=cfg,
        dataset=args["dataset"],
        cache_root=args.get("cache_root", "data/cache"),
        use_cache=args.get("use_cache", True),
        batch_size=args.get("batch_size", 8),
        max_seq_len=args.get("max_seq_len", 2048),
    )

    artifacts = []
    n_layers = len(caches)
    d_model = 0
    for layer, cache in caches.items():
        stem = Path(args.get("cache_root", "data/cache")) / "activations" / args["model_id"] / args["dataset"] / f"L{layer}"
        artifacts.append(str(stem) + ".safetensors")
        artifacts.append(str(stem) + ".meta.json")
        if d_model == 0 and cache.tensor.ndim >= 2:
            d_model = cache.tensor.shape[-1]

    elapsed = time.time() - t0

    from src.utils.model_runner import peak_vram_per_gpu_gib, write_result

    write_result(
        cli.output,
        {
            "status": "ok",
            "elapsed_seconds": round(elapsed, 2),
            "peak_vram_per_gpu": peak_vram_per_gpu_gib(),
            "artifacts": artifacts,
            "n_rows": len(rows),
            "n_layers": n_layers,
            "d_model": int(d_model),
        },
    )


if __name__ == "__main__":
    main()
