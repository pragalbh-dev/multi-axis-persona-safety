"""Subprocess-isolated model runner.

CRITICAL: vLLM's TP teardown leaks ~25 GB/GPU + 6 semaphores per call (Stage 0
finding, persists at TP=4 bf16). Sequential model loads in one Python process
OOM by the 4th-5th model. Every model load in production code MUST go through
`run_in_subprocess` so the parent never imports torch/vllm/transformers.

Each work module exposes a `python -m <module>` entrypoint with stable args:
  --args-json <path>   input JSON file (parsed and passed to main())
  --output    <path>   output JSON file (written by the child on success)

The child writes a single JSON object to `--output`, with at minimum:
  {
    "status": "ok",
    "elapsed_seconds": float,
    "peak_vram_per_gpu": [4 floats, GiB],
    "artifacts": [list of file paths produced]
  }
plus any work-module-specific fields.

On failure: child raises, subprocess.run(check=True) propagates non-zero exit
to the parent.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

__all__ = ["run_in_subprocess", "WorkModuleError"]


class WorkModuleError(RuntimeError):
    """Raised when a work-module subprocess exits non-zero or returns malformed JSON."""


def _build_child_env() -> dict[str, str]:
    """Construct env for the child process.

    Pass through the parent's env (so HF_TOKEN, OPENAI_API_KEY etc. are available)
    and force-set the GPU + allocator + transfer flags every child needs.
    """
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = env.get("CUDA_VISIBLE_DEVICES", "0,1,2,3")
    # Mitigate fragmentation across long-lived runs (matches the grid-search wrapper).
    env["PYTORCH_CUDA_ALLOC_CONF"] = env.get(
        "PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"
    )
    env["HF_HUB_ENABLE_HF_TRANSFER"] = env.get("HF_HUB_ENABLE_HF_TRANSFER", "1")
    return env


def run_in_subprocess(
    work_module: str,
    args: dict,
    output_path: Path | str | None = None,
    *,
    timeout_seconds: int | None = None,
    extra_env: dict[str, str] | None = None,
) -> dict:
    """Run a work-module in a fresh Python subprocess and return its JSON output.

    Args:
        work_module: e.g. "src.evaluation.run_judge". Must be invokable as
            `python -m <work_module>` and accept `--args-json` + `--output` flags.
        args: dict serialized to a tempfile JSON; passed to the child via
            `--args-json`. Should contain everything the work module needs to
            do its job (model_id, input parquet path, etc.).
        output_path: where the child writes its result JSON. If None, a tempfile
            is used and removed on return.
        timeout_seconds: subprocess timeout. None = wait forever.
        extra_env: overrides on top of the standard child env.

    Returns:
        The parsed JSON object the child wrote to `output_path`.

    Raises:
        WorkModuleError: child exit code != 0, or output JSON malformed/missing.
    """
    env = _build_child_env()
    if extra_env:
        env.update(extra_env)

    # Materialize args JSON to a temp file (or sibling of output_path).
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        args_path = output_path.with_suffix(".args.json")
        cleanup_args = False
        cleanup_output = False
    else:
        tmp_args = tempfile.NamedTemporaryFile(
            mode="w", suffix=".args.json", delete=False
        )
        args_path = Path(tmp_args.name)
        tmp_args.close()
        tmp_out = tempfile.NamedTemporaryFile(suffix=".out.json", delete=False)
        output_path = Path(tmp_out.name)
        tmp_out.close()
        cleanup_args = True
        cleanup_output = True

    args_path.write_text(json.dumps(args, indent=2, default=str))

    cmd = [
        sys.executable,
        "-m",
        work_module,
        "--args-json",
        str(args_path),
        "--output",
        str(output_path),
    ]

    t0 = time.time()
    try:
        completed = subprocess.run(
            cmd,
            env=env,
            check=False,
            timeout=timeout_seconds,
            capture_output=False,  # let stdout/stderr stream to console for live monitoring
        )
    except subprocess.TimeoutExpired as e:
        raise WorkModuleError(
            f"work_module={work_module} timed out after {timeout_seconds}s"
        ) from e
    elapsed = time.time() - t0

    if completed.returncode != 0:
        raise WorkModuleError(
            f"work_module={work_module} exited with code {completed.returncode} "
            f"after {elapsed:.1f}s. Args at {args_path}; output expected at {output_path}."
        )

    if not output_path.exists():
        raise WorkModuleError(
            f"work_module={work_module} exited 0 but did not write {output_path}."
        )

    try:
        result = json.loads(output_path.read_text())
    except json.JSONDecodeError as e:
        raise WorkModuleError(
            f"work_module={work_module} wrote invalid JSON to {output_path}: {e}"
        ) from e

    # Best-effort cleanup of tempfiles.
    if cleanup_args:
        args_path.unlink(missing_ok=True)
    if cleanup_output:
        # We've already read the JSON; safe to remove.
        output_path.unlink(missing_ok=True)

    return result


# ---------- helpers child work-modules use ----------


def write_result(output_path: Path | str, result: dict) -> None:
    """Helper for child work-modules to write their final JSON output atomically."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_suffix(output_path.suffix + ".tmp")
    tmp.write_text(json.dumps(result, indent=2, default=str))
    tmp.replace(output_path)


def peak_vram_per_gpu_gib() -> list[float]:
    """Return current per-GPU peak memory in GiB (driver-level via pynvml).

    Used by work-modules to record VRAM into their result JSON.
    Returns [] if pynvml is unavailable or no GPUs visible.
    """
    try:
        import pynvml  # type: ignore[import-not-found]

        pynvml.nvmlInit()
        n = pynvml.nvmlDeviceGetCount()
        out: list[float] = []
        for i in range(n):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            out.append(round(mem.used / (1024**3), 3))
        pynvml.nvmlShutdown()
        return out
    except Exception:
        return []
