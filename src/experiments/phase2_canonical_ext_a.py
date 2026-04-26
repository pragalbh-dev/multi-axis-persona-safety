"""Phase 2 — Canonical Ext A on merged Plan B baseline (500 DAN) + extended baseline (1707).

Loads both activation caches + parquets, concatenates into a unified n≈2207 baseline,
runs `compute_harm_direction`, writes results to
`results/plan_b_gemma2_27b/extensions/harm_direction_merged.json` plus a summary
markdown.

CPU-only. Fast (<5 min including IO). Run after Phase 1 completes.

Usage:
  uv run python -m src.experiments.phase2_canonical_ext_a
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from safetensors.torch import load_file

from src.analysis.harm_direction import _format_summary, compute_harm_direction
from src.extraction.types import ActivationCache


def _load_baseline(details_parquet: Path, cache_stem: Path, dataset_tag: str | None = None) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load (activations, harm_binary, dataset_per_row) for one baseline source.

    `details_parquet` schema must include condition_id, prompt_id, harm_binary,
    optionally `dataset` (defaults to dataset_tag if missing).
    `cache_stem` is the path WITHOUT .safetensors suffix; uses ActivationCache.load.
    """
    df = pd.read_parquet(details_parquet)
    df = df[df["condition_id"] == "baseline"].copy()
    if df.empty:
        raise RuntimeError(f"no baseline rows in {details_parquet}")

    cache = ActivationCache.load(cache_stem)
    acts = cache.tensor.float().numpy()
    pid_to_row = {pid: i for i, pid in enumerate(cache.prompt_ids)}

    keep_idx, harm, ds = [], [], []
    for _, row in df.iterrows():
        key = f"{row['prompt_id']}::{row['condition_id']}"
        if key not in pid_to_row:
            continue
        keep_idx.append(pid_to_row[key])
        harm.append(int(row["harm_binary"]))
        ds.append(str(row.get("dataset", dataset_tag or "unknown")))

    if not keep_idx:
        raise RuntimeError(f"zero rows aligned between {details_parquet} and {cache_stem}")

    return acts[np.asarray(keep_idx, dtype=np.int64)], np.asarray(harm, dtype=np.int32), ds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--plan-b-details",
        type=Path,
        default=Path("results/plan_b_gemma2_27b/details.parquet"),
        help="Plan B's details.parquet (n=500 DAN baseline + steered conditions)",
    )
    ap.add_argument(
        "--plan-b-cache",
        type=Path,
        default=Path("data/cache/activations/gemma_2_27b/plan_b_per_prompt_L21/L21"),
        help="Plan B's per-prompt activation cache stem",
    )
    ap.add_argument(
        "--ext-details",
        type=Path,
        default=Path("results/plan_b_gemma2_27b/extensions/baseline_extended.parquet"),
        help="Phase 1 extended baseline parquet",
    )
    ap.add_argument(
        "--ext-cache",
        type=Path,
        default=Path("data/cache/activations/gemma_2_27b/baseline_extended_L21/L21"),
        help="Phase 1 extended activation cache stem",
    )
    ap.add_argument(
        "--extraction-dir",
        type=Path,
        default=Path("results/plan_b_gemma2_27b/extraction"),
        help="dir holding aa.safetensors + pcs.safetensors (Plan B extraction artifacts)",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=Path("results/plan_b_gemma2_27b/extensions/harm_direction_merged.json"),
    )
    ap.add_argument(
        "--summary",
        type=Path,
        default=Path("results/plan_b_gemma2_27b/extensions/harm_direction_merged_summary.md"),
    )
    args = ap.parse_args()

    # Plan B baseline (500 DAN). Plan B's parquet doesn't tag dataset explicitly
    # — its baseline IS DAN — so we set dataset_tag.
    print(f"Loading Plan B baseline from {args.plan_b_details} + cache {args.plan_b_cache}...")
    acts_b, harm_b, ds_b = _load_baseline(args.plan_b_details, args.plan_b_cache, dataset_tag="dan")
    print(f"  n={len(harm_b)}, harm={int(harm_b.sum())}, datasets={set(ds_b)}")

    print(f"Loading extended baseline from {args.ext_details} + cache {args.ext_cache}...")
    acts_e, harm_e, ds_e = _load_baseline(args.ext_details, args.ext_cache)
    print(f"  n={len(harm_e)}, harm={int(harm_e.sum())}, datasets={set(ds_e)}")

    activations = np.concatenate([acts_b, acts_e], axis=0)
    harm_binary = np.concatenate([harm_b, harm_e], axis=0)
    dataset_per_row = ds_b + ds_e
    print(f"Merged: n={len(harm_binary)}, harm={int(harm_binary.sum())}, "
          f"datasets={ {k: dataset_per_row.count(k) for k in set(dataset_per_row)} }")

    aa = load_file(str(args.extraction_dir / "aa.safetensors"))["aa_at_lstar"].float().numpy()
    pcs_d = load_file(str(args.extraction_dir / "pcs.safetensors"))
    pcs = pcs_d["pcs_at_lstar"].float().numpy()
    pca_mean = pcs_d["pca_mean_at_lstar"].float().numpy()

    res = compute_harm_direction(
        activations=activations,
        harm_binary=harm_binary,
        aa=aa,
        pcs=pcs,
        pca_mean=pca_mean,
        layer=21,
        dataset_per_row=dataset_per_row,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(res.to_dict(), indent=2))

    summary = _format_summary(res)
    args.summary.write_text(
        "# Ext A — DiffMean v_harm diagnostic (CANONICAL, merged baseline)\n\n"
        f"**Sources:** Plan B baseline (n=500 DAN) + Phase 1 extended baseline (n={len(harm_e)} DAN unused + Shah).\n\n"
        f"**Total:** n={len(harm_binary)} ({int(harm_binary.sum())} harm / {int((1-harm_binary).sum())} safe).\n\n"
        "```\n" + summary + "\n```\n"
    )

    print(summary)
    print(f"\n→ JSON: {args.output}")
    print(f"→ Summary: {args.summary}")


if __name__ == "__main__":
    main()
