"""Tiny no-GPU work-module used by tests/unit/test_model_runner.py.

Reads the args JSON, echoes it back into the output JSON. Sets
status="ok" on success or raises RuntimeError if args.force_failure
is truthy (to verify error propagation).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--args-json", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    cli = parser.parse_args()

    args = json.loads(cli.args_json.read_text())

    if args.get("force_failure"):
        raise RuntimeError("force_failure=True; deliberate test failure")

    t0 = time.time()
    # Simulate trivial work
    time.sleep(0.05)
    elapsed = time.time() - t0

    result = {
        "status": "ok",
        "elapsed_seconds": round(elapsed, 4),
        "peak_vram_per_gpu": [],  # no GPU in this fixture
        "echoed_args": args,
        "artifacts": [],
    }
    cli.output.parent.mkdir(parents=True, exist_ok=True)
    cli.output.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
