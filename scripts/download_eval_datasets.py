"""Download evaluation datasets for the multi-axis-persona-safety project.

Stage 0 T0.10. Writes each dataset to `data/eval/<name>/` as a JSONL file
plus a sibling `.meta.json` with HF ID, revision, row count, and schema.

Shah et al. persona-jailbreak (1,100 prompts) is intentionally skipped —
not locatable on HF via search on 2026-04-24; deferred to Stage 2 T2.0.

Usage:
    uv run python scripts/download_eval_datasets.py
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

# Ensure project root is importable so we pick up CUDA pinning (even though
# this script doesn't touch CUDA, it's harmless and keeps patterns uniform).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils import env  # noqa: F401  pins CUDA_VISIBLE_DEVICES=2,3

from datasets import load_dataset  # noqa: E402
from huggingface_hub import HfApi  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = PROJECT_ROOT / "data" / "eval"
SEED = 42


def _write(rows: list[dict], name: str, hf_id: str, revision: str, schema: list[str]) -> None:
    out_dir = OUT_ROOT / name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "prompts.jsonl"
    with out_file.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    meta = {
        "hf_id": hf_id,
        "revision": revision,
        "row_count": len(rows),
        "schema": schema,
        "seed": SEED if "subsample" in name or "1400" in name or "1000" in name else None,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"  wrote {out_file} ({len(rows)} rows)  HF={hf_id}@{revision[:7]}")


def _revision(hf_id: str) -> str:
    api = HfApi()
    info = api.dataset_info(hf_id)
    return info.sha or "unknown"


def download_ifeval() -> None:
    hf_id = "google/IFEval"
    ds = load_dataset(hf_id, split="train")
    rev = _revision(hf_id)
    rows = [{"id": i, "prompt": r["prompt"], "instruction_id_list": r.get("instruction_id_list"),
             "kwargs": r.get("kwargs")} for i, r in enumerate(ds)]
    _write(rows, "ifeval", hf_id, rev, schema=["id", "prompt", "instruction_id_list", "kwargs"])


def download_mmlu_pro_1400() -> None:
    hf_id = "TIGER-Lab/MMLU-Pro"
    ds = load_dataset(hf_id, split="test")
    rev = _revision(hf_id)
    rng = random.Random(SEED)
    idx = sorted(rng.sample(range(len(ds)), 1400))
    rows = [{"id": int(i), "question": ds[i]["question"], "options": ds[i]["options"],
             "answer": ds[i]["answer"], "category": ds[i]["category"]} for i in idx]
    _write(rows, "mmlu_pro_1400", hf_id, rev,
           schema=["id", "question", "options", "answer", "category"])


def download_gsm8k_1000() -> None:
    hf_id = "openai/gsm8k"
    ds = load_dataset(hf_id, "main", split="test")
    rev = _revision(hf_id)
    rng = random.Random(SEED)
    idx = sorted(rng.sample(range(len(ds)), 1000))
    rows = [{"id": int(i), "question": ds[i]["question"], "answer": ds[i]["answer"]} for i in idx]
    _write(rows, "gsm8k_1000", hf_id, rev, schema=["id", "question", "answer"])


def download_eq_bench() -> None:
    hf_id = "pbevan11/EQ-Bench"
    ds = load_dataset(hf_id, split="validation")
    rev = _revision(hf_id)
    rows = [{"id": i, **{k: r[k] for k in r.keys()}} for i, r in enumerate(ds)]
    _write(rows, "eq_bench", hf_id, rev, schema=list(ds.features.keys()))


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    print("Downloading eval datasets → data/eval/")
    print("  (Shah persona-jailbreak deferred to Stage 2 T2.0 — not on HF)")
    download_ifeval()
    download_mmlu_pro_1400()
    download_gsm8k_1000()
    download_eq_bench()
    print("done.")


if __name__ == "__main__":
    main()
