"""End-to-end smoke for the new --backend sglang path in run_subject_rollouts.

Runs 5 prompts on Gemma 2 27B with a single-axis cap (using the spike's
axis_0.safetensors at L*=22, tau=0). Verifies:
  1. _run_sglang launches the SGLang server via subprocess
  2. --forward-hooks JSON is built correctly from the steering dict
  3. /v1/chat/completions returns text + completion_tokens
  4. Output parquet has 5 rows with non-empty response_text
  5. tokens_per_second > 50 (sanity floor — spike measured 280+ for Gemma 2 27B)

Usage:
  source .venv-sglang/bin/activate    # SGLang env required
  python -m tests.integration.sglang_rollouts_smoke
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

_REPO = Path(__file__).resolve().parents[2]
_OUT = _REPO / "results" / "sglang_spike" / "rollouts_smoke"


def _build_prompts() -> Path:
    _OUT.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({
        "prompt_id": [0, 1, 2, 3, 4],
        "dataset": ["smoke"] * 5,
        "input_text": [
            "Explain in two sentences why the sky is blue.",
            "List three differences between a violin and a cello.",
            "Write a haiku about the ocean.",
            "What is 17 multiplied by 23? Show your work.",
            "Give me a one-sentence definition of recursion.",
        ],
    })
    p = _OUT / "prompts.parquet"
    df.to_parquet(p, index=False)
    return p


def main() -> int:
    prompts = _build_prompts()
    output_parquet = _OUT / "rollouts.parquet"
    if output_parquet.exists():
        output_parquet.unlink()

    vec_path = _REPO / "data" / "cache" / "sglang_spike_vectors" / "axis_0.safetensors"
    if not vec_path.exists():
        print(f"FAIL: spike vector not at {vec_path}; run sglang_hooks_smoke setup first", file=sys.stderr)
        return 2

    args = {
        "model_id": "gemma_2_27b",
        "backend": "sglang",
        "prompts_path": str(prompts),
        "output_path": str(output_parquet),
        "condition_id": "smoke_capping_axis0",
        "seed": 42,
        "max_new_tokens": 64,
        "temperature": 0.0,
        "steering": {
            "mode": "capping",
            "vectors": [str(vec_path)],
            "cap_thresholds": [0.0],
            "layers": [[22, 22]],
            "positions": "all",
        },
    }
    args_path = _OUT / "args.json"
    args_path.write_text(json.dumps(args, indent=2))
    work_path = _OUT / "work.json"

    print(f"[smoke] launching run_subject_rollouts --backend sglang …")
    t0 = time.time()
    cmd = [
        sys.executable, "-m", "src.evaluation.run_subject_rollouts",
        "--args-json", str(args_path), "--output", str(work_path),
    ]
    res = subprocess.run(cmd, cwd=str(_REPO), capture_output=False)
    elapsed = time.time() - t0
    if res.returncode != 0:
        print(f"FAIL: subprocess returned {res.returncode}", file=sys.stderr)
        return res.returncode

    work = json.loads(work_path.read_text())
    print(f"[smoke] work result: {json.dumps(work, indent=2)}")

    if work["status"] != "ok":
        print("FAIL: status != ok", file=sys.stderr); return 3
    if work["n_rows"] != 5:
        print(f"FAIL: n_rows={work['n_rows']}, expected 5", file=sys.stderr); return 4

    df = pd.read_parquet(output_parquet)
    if len(df) != 5:
        print(f"FAIL: parquet len={len(df)}", file=sys.stderr); return 5
    empty = df["response_text"].apply(lambda s: not s or len(s) < 5).sum()
    if empty:
        print(f"FAIL: {empty} responses are empty/too short", file=sys.stderr); return 6

    tps = work["tokens_per_second"]
    if tps < 50:
        print(f"FAIL: tps={tps} below sanity floor (50)", file=sys.stderr); return 7

    print(f"PASS — n_rows={work['n_rows']} tps={tps} elapsed={elapsed:.1f}s")
    print(f"      rollout columns: {list(df.columns)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
